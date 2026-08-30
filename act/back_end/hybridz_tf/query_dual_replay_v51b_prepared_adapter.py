# ===- query_dual_replay_v51b_prepared_adapter.py - Prepared adapter ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===-------------------------------------------------------------------===#
"""One-shot prepared-material adapter for the frozen V5.1b private kernel.

This module is an isolated, non-authoritative integration candidate.  It
does not weaken or edit the frozen private numeric kernel.  Instead, it
validates the frozen factory's closure ABI, copies its sealed dependency
tuples, and enables the appended prepared-mode ABI with factory-local
one-shot dispatch functions and private admission-result wrapper types.

An admission arms exactly one registry-produced ``DenseV51Support`` or
``DenseConvV51Plan``.  The frozen kernel snapshots the raw arrays, calls the
one-shot dispatch, independently rebuilds every numerical field, compares
the prepared value field by field, and retains only its rebuilt private
core.  The adapter never treats the raw-binding manifest as proof that its
hashes describe the supplied arrays: that provenance check belongs to the
future integrated registry.  Here the manifest is an exact immutable value
bound to the adapter locator, generation, kind, and call.

The adapter returns opaque factory-local locators and the frozen kernel's
exact tuple/bytes execution result.  ``stats()`` returns counts only.  No
prepared object, ndarray, frozen locator, or proof-authoritative value is
placed in a public receipt.

This candidate remains non-authoritative.  The prepared-mode kernel path does
not construct or retain the public Conv layer/box/deadline classes.  Public
prepared-result identities are admitted through weak-reference exact-type
gates, copied field by field into bytes-backed factory-private wrappers, and
then independently checked by the frozen numeric kernel.
"""

from __future__ import annotations

import builtins as _builtins_module
import _thread as _thread_module
import math as _math_module
import os as _os_module
import time as _time_module
import types as _types_module
import weakref as _weakref_module
from builtins import (
    BaseException as _BaseExceptionModule,
    Exception as _ExceptionModule,
    MemoryError as _MemoryErrorModule,
    TypeError as _TypeErrorModule,
    ValueError as _ValueErrorModule,
    id as _id_module,
    len as _len_module,
    memoryview as _memoryview_module,
    property as _property_module,
    range as _range_module,
    str as _str_module,
)
from types import MappingProxyType as _MappingProxyTypeModule
from typing import Any, NoReturn

import numpy as _np_module

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as _private_module,
)


SCHEMA = "act.query_dual_replay_v51b_prepared_numeric_adapter.v1"
NUMERIC_PROTOCOL = "one_shot_registry_prepared_dispatch_v51b"
RAW_BINDING_TAG = b"act.v51b.prepared.raw-binding.v1"
ZERO_SHA256 = "0" * 64
BRANCH_DENSE = "DENSE"
BRANCH_CONV_DENSE = "CONV2D_DENSE"

_ExactBytesModule = b"".__class__
_ExactIntModule = (0).__class__
_ExactFloatModule = (0.0).__class__
_ExactBoolModule = (False).__class__
_ExactTupleModule = ().__class__
_ExactStrModule = "".__class__
_ExactDictModule = {}.__class__
_ExactTypeModule = _ExactTupleModule.__class__
_ExactObjectModule = _ExactTupleModule.__base__
_MappingProxyTypeTypeModule = _MappingProxyTypeModule({}).__class__
_ModuleTypeModule = _types_module.ModuleType
_ModuleGetattributeModule = _ModuleTypeModule.__getattribute__
_CodeTypeModule = _types_module.CodeType
_ObjectGetattributeModule = _ExactObjectModule.__getattribute__
_ObjectSetattrModule = _ExactObjectModule.__setattr__
_NDArrayTobytesModule = _np_module.ndarray.tobytes
_NDArrayReshapeModule = _np_module.ndarray.reshape
_NpFrombufferModule = _np_module.frombuffer


class PreparedNumericAdapterError(RuntimeError):
    """Stable fail-closed error for the prepared-material adapter."""

    def __init__(self, code: str, message: str):
        self.code = "{}".format(code)
        self.args = ("{}: {}".format(self.code, message),)


class PreparedNumericAdapterTimeout(PreparedNumericAdapterError):
    """The adapter's one fixed absolute deadline expired."""

    def __init__(self) -> None:
        self.code = "DEADLINE_EXPIRED"
        self.args = (
            "DEADLINE_EXPIRED: V5.1b prepared-adapter deadline expired",
        )


def _create_prepared_numeric_adapter_impl(
    *,
    deadline: float,
    _sealed_dependencies: Any,
) -> Any:
    """Build one dependency-sealed prepared-material adapter."""

    _ExactBytes = b"".__class__
    _ExactInt = (0).__class__
    _ExactFloat = (0.0).__class__
    _ExactBool = (False).__class__
    _ExactTuple = ().__class__
    _ExactStr = "".__class__
    _ExactDict = {}.__class__
    _ExactType = _ExactTuple.__class__
    _ExactObject = _ExactTuple.__base__

    (
        _module_gates,
        _trusted_builtins,
        _direct_dependencies,
    ) = _sealed_dependencies
    (
        _AdapterError,
        _AdapterTimeout,
        _monotonic,
        _isfinite,
        _getpid,
        _get_ident,
        _allocate_lock,
        _RLock,
        _weakref_ref,
        _MappingProxyType,
        _MappingProxyTypeType,
        _ModuleType,
        _module_getattribute,
        _FunctionType,
        _CodeType,
        _frozen_factory,
        _expected_factory_code,
        _expected_factory_impl_spec,
        _expected_outer,
        _expected_direct,
        _expected_dense_prepare,
        _expected_conv_prepare,
        _ndarray_type,
        _object_getattribute,
        _object_setattr,
        _ndarray_tobytes,
        _ndarray_reshape,
        _np_frombuffer,
        _schema,
        _raw_binding_tag,
        _zero_sha,
        _branch_dense,
        _branch_conv,
    ) = _direct_dependencies

    _builtins = _MappingProxyType(_ExactDict(_trusted_builtins))
    _BaseException = _builtins["BaseException"]
    _Exception = _builtins["Exception"]
    _MemoryError = _builtins["MemoryError"]
    _TypeError = _builtins["TypeError"]
    _ValueError = _builtins["ValueError"]
    _id = _builtins["id"]
    _len = _builtins["len"]
    _memoryview = _builtins["memoryview"]
    _property = _builtins["property"]
    _range = _builtins["range"]
    _str = _builtins["str"]

    def _raise(code: str, message: str) -> NoReturn:
        raise _AdapterError(code, message)

    # Gate every dependency used to construct the factory before invoking
    # one of them.  Runtime operations use only the captured closure values.
    _sentinel = _ExactObject()

    def _gate_modules() -> None:
        for _module, _bindings in _module_gates:
            if _ExactType(_module) is not _ModuleType:
                _raise(
                    "DEPENDENCY_SUBSTITUTION",
                    "trusted dependency changed from an exact module",
                )
            _module_dict = _module_getattribute(_module, "__dict__")
            if _ExactType(_module_dict) is not _ExactDict:
                _raise(
                    "DEPENDENCY_SUBSTITUTION",
                    "trusted dependency module dictionary changed type",
                )
            for _name, _expected in _bindings:
                if (
                    _ExactDict.get(_module_dict, _name, _sentinel)
                    is not _expected
                ):
                    _raise(
                        "DEPENDENCY_SUBSTITUTION",
                        "trusted dependency binding was substituted",
                    )

    _gate_modules()

    if (
        _ExactType(deadline) is not _ExactFloat
        or not _isfinite(deadline)
    ):
        _raise(
            "INVALID_DEADLINE",
            "deadline must be an exact finite float monotonic timestamp",
        )
    if _monotonic() >= deadline:
        raise _AdapterTimeout()

    # Validate the complete frozen wrapper/closure shape and the identities
    # captured when this module was imported.  A tuple is copied below; the
    # frozen tuples are never mutated.
    if (
        _ExactType(_frozen_factory) is not _FunctionType
        or _frozen_factory.__code__ is not _expected_factory_code
        or _frozen_factory.__code__.co_freevars
        != ("implementation", "sealed_dependencies")
    ):
        _raise(
            "FROZEN_ABI_MISMATCH",
            "private-kernel factory function ABI changed",
        )
    _factory_closure = _frozen_factory.__closure__
    if (
        _ExactType(_factory_closure) is not _ExactTuple
        or _len(_factory_closure) != 2
        or _factory_closure[0].cell_contents
        is not _expected_factory_impl_spec
        or _factory_closure[1].cell_contents is not _expected_outer
    ):
        _raise(
            "FROZEN_ABI_MISMATCH",
            "private-kernel factory closure changed",
        )
    _implementation_spec = _factory_closure[0].cell_contents
    _outer = _factory_closure[1].cell_contents
    if (
        _ExactType(_implementation_spec) is not _ExactTuple
        or _implementation_spec is not _expected_factory_impl_spec
        or _len(_implementation_spec) != 7
        or _ExactType(_implementation_spec[0]) is not _CodeType
        or _implementation_spec[0].co_freevars != ()
        or _ExactType(_implementation_spec[1]) is not _ExactStr
        or _implementation_spec[2] is not None
        or _implementation_spec[3] is not None
        or _implementation_spec[4] is not _FunctionType
        or _implementation_spec[5] is not _ExactDict
        or _ExactType(_implementation_spec[6]) is not _ExactStr
        or _ExactType(_outer) is not _ExactTuple
        or _outer is not _expected_outer
        or _len(_outer) != 5
        or _ExactType(_outer[4]) is not _ExactTuple
        or _outer[4] is not _expected_direct
        or _len(_outer[4]) != 44
        or _outer[4][8] is not _expected_dense_prepare
        or _outer[4][9] is not _expected_conv_prepare
        or _ExactType(_outer[4][43]) is not _ExactBool
        or _outer[4][43] is not False
    ):
        _raise(
            "FROZEN_ABI_MISMATCH",
            "private-kernel sealed dependency ABI changed",
        )
    _direct = _outer[4]
    _PrivateError = _direct[0]
    _PrivateTimeout = _direct[1]
    _ArrayMemoryError = _direct[2]
    _DenseSupportPublicRef = _weakref_ref(_direct[16])
    _ConvPlanPublicRef = _weakref_ref(_direct[17])
    _ConvOffsetPublicRef = _weakref_ref(_direct[18])
    _F64 = _direct[20]
    _BOOL = _direct[21]
    _I64 = _direct[22]

    _DenseFields = (
        "support_upper",
        "box_mass_upper",
        "weight_shape",
        "weight_sha256",
        "max_abs_sha256",
        "support_sha256",
        "binding",
        "weight_exponent_min",
        "weight_exponent_max",
        "support_exponent_min",
        "support_exponent_max",
        "max_abs_exponent_min",
        "max_abs_exponent_max",
        "global_underflow_risk",
        "global_subnormal_operand",
        "disjoint_box_mass",
        "diagnostics",
        "proof_authority",
    )
    _ConvFields = (
        "layer_id",
        "input_shape",
        "output_shape",
        "stride",
        "padding",
        "dilation",
        "groups",
        "weight",
        "support",
        "offsets",
        "manifest",
        "proof_authority",
    )
    _OffsetFields = (
        "group",
        "kh",
        "kw",
        "co_start",
        "co_end",
        "ci_start",
        "ci_end",
        "output_h_indices",
        "output_w_indices",
        "targets",
        "support_flat",
        "channel_support_flat",
        "support_activity_flat",
        "support_sum_upper",
    )
    _PrivateDenseFields = (
        "support_upper",
        "box_mass_upper",
        "weight_shape",
        "weight_exponent_min",
        "weight_exponent_max",
        "support_exponent_min",
        "support_exponent_max",
        "max_abs_exponent_min",
        "max_abs_exponent_max",
        "global_underflow_risk",
        "global_subnormal_operand",
        "disjoint_box_mass",
        "proof_authority",
    )
    _PrivateConvFields = (
        "layer_id",
        "input_shape",
        "output_shape",
        "stride",
        "padding",
        "dilation",
        "groups",
        "weight",
        "support",
        "offsets",
        "proof_authority",
    )
    _PrivateOffsetFields = _OffsetFields
    _CONV_DISPATCH_TAG = (
        b"act.v51b.private.prepared-conv-dispatch.v1"
    )

    # These exact factory-private types replace every public object-bearing
    # dependency in direct[10:19].  The first three and exception slots are
    # deliberately inert on the prepared path.
    _UnusedLayerType = _ExactType(
        "_PreparedUnusedLayer", (_ExactObject,), {"__slots__": ()}
    )
    _UnusedBoxType = _ExactType(
        "_PreparedUnusedBox", (_ExactObject,), {"__slots__": ()}
    )
    _UnusedDeadlineType = _ExactType(
        "_PreparedUnusedDeadline", (_ExactObject,), {"__slots__": ()}
    )
    _UnusedReplayTimeout = _ExactType(
        "_PreparedUnusedReplayTimeout", (_Exception,), {"__slots__": ()}
    )
    _PrivateDenseAdmissionError = _ExactType(
        "_PreparedDenseAdmissionError", (_Exception,), {"__slots__": ()}
    )
    _UnusedReplayError = _ExactType(
        "_PreparedUnusedReplayError", (_Exception,), {"__slots__": ()}
    )
    _PrivateDenseSupport = _ExactType(
        "_PreparedDenseSupport",
        (_ExactObject,),
        {"__slots__": _PrivateDenseFields},
    )
    _PrivateConvPlan = _ExactType(
        "_PreparedConvPlan",
        (_ExactObject,),
        {"__slots__": _PrivateConvFields},
    )
    _PrivateConvOffset = _ExactType(
        "_PreparedConvOffset",
        (_ExactObject,),
        {"__slots__": _PrivateOffsetFields},
    )

    def _copy_exact_int_tuple(
        value: Any,
        *,
        length: int,
        name: str,
    ) -> Any:
        if _ExactType(value) is not _ExactTuple or _len(value) != length:
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} must be an exact length-{} tuple".format(
                    name, length
                ),
            )
        _items = []
        for _item in value:
            if _ExactType(_item) is not _ExactInt:
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "{} entries must be exact integers".format(name),
                )
            _items.append(_item)
        return _ExactTuple(_items)

    def _snapshot_public_dict(
        value: Any,
        expected_ref: Any,
        fields: Any,
        name: str,
    ) -> Any:
        _expected_type = expected_ref()
        _type_matches = (
            _expected_type is not None
            and _ExactType(value) is _expected_type
        )
        # Do not leave a temporary strong public-class reference in an
        # exception traceback or any returned operation graph.
        _expected_type = None
        if not _type_matches:
            _raise(
                "INVALID_PREPARED_TYPE",
                "{} has the wrong exact public type identity".format(name),
            )
        _source = _object_getattribute(value, "__dict__")
        if (
            _ExactType(_source) is not _ExactDict
            or _len(_source) != _len(fields)
        ):
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} instance dictionary changed shape".format(name),
            )
        _values = []
        for _field in fields:
            _item = _ExactDict.get(_source, _field, _sentinel)
            if _item is _sentinel:
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "{} omitted field {}".format(name, _field),
                )
            _values.append(_item)
        return (_source, _ExactTuple(_values))

    def _finish_public_snapshot(
        value: Any,
        source: Any,
        fields: Any,
        values: Any,
        name: str,
    ) -> None:
        _after = _object_getattribute(value, "__dict__")
        if _after is not source or _len(_after) != _len(fields):
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} changed its instance dictionary during copy".format(
                    name
                ),
            )
        for _index in _range(_len(fields)):
            if (
                _ExactDict.get(_after, fields[_index], _sentinel)
                is not values[_index]
            ):
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "{} field changed during copy".format(name),
                )

    # Admission must remain metadata-first.  No source-proportional copy is
    # permitted until every prepared ndarray in the material has passed its
    # exact-type, immutable-storage, deadline, offset-count, and aggregate
    # resource checks.
    _MAX_PREPARED_ARRAY_BYTES = 1 << 30
    _MAX_PREPARED_RAW_BYTES = 1 << 28
    _MAX_PREPARED_OFFSETS = 65536

    def _prepared_copy_gate() -> None:
        if _getpid() != owner_pid:
            _raise(
                "FORKED_PROCESS",
                "prepared adapter cannot be used after fork",
            )
        if _monotonic() >= owner_deadline:
            raise _AdapterTimeout()

    def _prepared_array_bytes_root(value: Any) -> Any:
        _base = value
        while _ExactType(_base) is _ndarray_type:
            _base = _base.base
        return _base

    def _checked_prepared_nbytes(
        shape: Any,
        itemsize: Any,
        name: str,
    ) -> int:
        if (
            _ExactType(shape) is not _ExactTuple
            or _ExactType(itemsize) is not _ExactInt
            or itemsize <= 0
        ):
            _raise(
                "FROZEN_ABI_MISMATCH",
                "prepared dtype or shape has invalid size metadata",
            )
        _has_zero_extent = False
        for _extent in shape:
            if _ExactType(_extent) is not _ExactInt or _extent < 0:
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "{} has a malformed shape".format(name),
                )
            if _extent == 0:
                _has_zero_extent = True
        if _has_zero_extent:
            return 0
        _element_limit = _MAX_PREPARED_ARRAY_BYTES // itemsize
        _count = 1
        for _extent in shape:
            if _extent != 0 and _count > _element_limit // _extent:
                _raise(
                    "RESOURCE_LIMIT",
                    "{} exceeds the per-array prepared-copy budget".format(
                        name
                    ),
                )
            _count *= _extent
        return _count * itemsize

    def _scan_private_array(
        value: Any,
        *,
        dtype: Any,
        ndim: int,
        name: str,
    ) -> Any:
        _prepared_copy_gate()
        if _ExactType(value) is not _ndarray_type:
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} must be an exact ndarray".format(name),
            )
        _value_dtype = value.dtype
        _value_shape = value.shape
        _value_flags = value.flags
        _value_nbytes = value.nbytes
        if (
            _value_dtype is not dtype
            or not _value_dtype.isnative
            or _ExactType(_value_shape) is not _ExactTuple
            or value.ndim != ndim
            or not _value_flags.c_contiguous
            or _value_flags.writeable
            or _value_flags.owndata
            or _ExactType(_value_nbytes) is not _ExactInt
        ):
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} has the wrong dtype, rank, layout, or storage".format(
                    name
                ),
            )
        _shape_items = []
        for _extent in _value_shape:
            if _ExactType(_extent) is not _ExactInt or _extent < 0:
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "{} has a malformed shape".format(name),
                )
            _shape_items.append(_extent)
        _shape = _ExactTuple(_shape_items)
        _itemsize = dtype.itemsize
        _expected_nbytes = _checked_prepared_nbytes(
            _shape, _itemsize, name
        )
        if _value_nbytes != _expected_nbytes:
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} has inconsistent byte metadata".format(name),
            )
        if _expected_nbytes > _MAX_PREPARED_ARRAY_BYTES:
            _raise(
                "RESOURCE_LIMIT",
                "{} exceeds the per-array prepared-copy budget".format(
                    name
                ),
            )
        _root = _prepared_array_bytes_root(value)
        if _ExactType(_root) is not _ExactBytes:
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} must have exact immutable bytes-rooted storage".format(
                    name
                ),
            )
        if (
            value.dtype is not _value_dtype
            or value.shape != _value_shape
            or value.nbytes != _value_nbytes
            or not value.flags.c_contiguous
            or value.flags.writeable
            or value.flags.owndata
            or _prepared_array_bytes_root(value) is not _root
        ):
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} changed during metadata scan".format(name),
            )
        _prepared_copy_gate()
        # value, expected dtype, exact shape, byte count, immutable root, name
        return (
            value,
            dtype,
            _shape,
            _expected_nbytes,
            _root,
            name,
        )

    def _check_private_array_ticket(ticket: Any) -> None:
        (
            _value,
            _dtype,
            _shape,
            _expected_nbytes,
            _root,
            _name,
        ) = ticket
        if (
            _ExactType(_value) is not _ndarray_type
            or _value.dtype is not _dtype
            or not _value.dtype.isnative
            or _value.shape != _shape
            or _value.ndim != _len(_shape)
            or _value.nbytes != _expected_nbytes
            or not _value.flags.c_contiguous
            or _value.flags.writeable
            or _value.flags.owndata
            or _prepared_array_bytes_root(_value) is not _root
        ):
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} changed after metadata admission".format(_name),
            )

    def _account_private_array(
        total_nbytes: int,
        ticket: Any,
    ) -> int:
        _nbytes = ticket[3]
        if total_nbytes > _MAX_PREPARED_RAW_BYTES - _nbytes:
            _raise(
                "RESOURCE_LIMIT",
                "prepared material exceeds the aggregate copy budget",
            )
        return total_nbytes + _nbytes

    def _copy_ticketed_array(ticket: Any) -> Any:
        _prepared_copy_gate()
        _check_private_array_ticket(ticket)
        (
            _value,
            _dtype,
            _shape,
            _expected_nbytes,
            _root,
            _name,
        ) = ticket
        _view = None
        _payload = None
        _failure = None
        try:
            _view = _memoryview(_value)
        except (_MemoryError, _ArrayMemoryError):
            _failure = "RESOURCE_LIMIT"
        except (_TypeError, _ValueError):
            _failure = "INVALID_PREPARED_VALUE"
        if _failure is None:
            if (
                _view.nbytes != _expected_nbytes
                or not _view.c_contiguous
                or not _view.readonly
            ):
                _failure = "INVALID_PREPARED_VALUE"
            else:
                try:
                    _payload = _ExactBytes(_view)
                except (_MemoryError, _ArrayMemoryError):
                    _failure = "RESOURCE_LIMIT"
                except (_TypeError, _ValueError):
                    _failure = "INVALID_PREPARED_VALUE"
        if _view is not None:
            _view.release()
        _view = None
        if _failure is not None:
            _value = None
            ticket = None
            _root = None
            if _failure == "RESOURCE_LIMIT":
                _raise(
                    "RESOURCE_LIMIT",
                    "prepared array copy allocation failed",
                )
            _raise(
                "INVALID_PREPARED_VALUE",
                "prepared array could not be copied",
            )
        if (
            _ExactType(_payload) is not _ExactBytes
            or _len(_payload) != _expected_nbytes
        ):
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} produced an inconsistent private payload".format(
                    _name
                ),
            )
        _prepared_copy_gate()
        _check_private_array_ticket(ticket)
        _result = None
        _failure = None
        try:
            _result = _np_frombuffer(_payload, dtype=_dtype)
            _result = _ndarray_reshape(_result, _shape)
        except (_MemoryError, _ArrayMemoryError):
            _failure = "RESOURCE_LIMIT"
        except (_TypeError, _ValueError):
            _failure = "INVALID_PREPARED_VALUE"
        if _failure is not None:
            _value = None
            ticket = None
            _root = None
            _payload = None
            _result = None
            if _failure == "RESOURCE_LIMIT":
                _raise(
                    "RESOURCE_LIMIT",
                    "prepared array reconstruction allocation failed",
                )
            _raise(
                "INVALID_PREPARED_VALUE",
                "prepared array could not be reconstructed",
            )
        if (
            _ExactType(_result) is not _ndarray_type
            or _result.dtype is not _dtype
            or _result.shape != _shape
            or _result.flags.writeable
            or _result.flags.owndata
        ):
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} did not produce immutable private storage".format(
                    _name
                ),
            )
        _base = _prepared_array_bytes_root(_result)
        if _ExactType(_base) is not _ExactBytes or _base is not _payload:
            _raise(
                "INVALID_PREPARED_VALUE",
                "{} private storage is not payload-rooted".format(_name),
            )
        _prepared_copy_gate()
        return _result

    def _set_private(target: Any, name: str, value: Any) -> None:
        _object_setattr(target, name, value)

    def _copy_dense_support(value: Any) -> Any:
        _prepared_copy_gate()
        _source, _values = _snapshot_public_dict(
            value,
            _DenseSupportPublicRef,
            _DenseFields,
            "prepared Dense support",
        )
        _support_ticket = _scan_private_array(
            _values[0],
            dtype=_F64,
            ndim=1,
            name="prepared Dense support_upper",
        )
        if _ExactType(_values[1]) is not _ExactFloat:
            _raise(
                "INVALID_PREPARED_VALUE",
                "prepared Dense box_mass_upper must be an exact float",
            )
        _weight_shape = _copy_exact_int_tuple(
            _values[2],
            length=2,
            name="prepared Dense weight_shape",
        )
        for _index in _range(7, 13):
            if (
                _values[_index] is not None
                and _ExactType(_values[_index]) is not _ExactInt
            ):
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "prepared Dense exponent metadata changed type",
                )
        for _index in _range(13, 16):
            if _ExactType(_values[_index]) is not _ExactBool:
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "prepared Dense boolean metadata changed type",
                )
        if _ExactType(_values[17]) is not _ExactBool:
            _raise(
                "INVALID_PREPARED_VALUE",
                "prepared Dense authority bit changed type",
            )
        _account_private_array(0, _support_ticket)
        _prepared_copy_gate()
        _support = _copy_ticketed_array(_support_ticket)
        _private = _PrivateDenseSupport()
        _set_private(_private, "support_upper", _support)
        _set_private(_private, "box_mass_upper", _values[1])
        _set_private(_private, "weight_shape", _weight_shape)
        _set_private(_private, "weight_exponent_min", _values[7])
        _set_private(_private, "weight_exponent_max", _values[8])
        _set_private(_private, "support_exponent_min", _values[9])
        _set_private(_private, "support_exponent_max", _values[10])
        _set_private(_private, "max_abs_exponent_min", _values[11])
        _set_private(_private, "max_abs_exponent_max", _values[12])
        _set_private(_private, "global_underflow_risk", _values[13])
        _set_private(_private, "global_subnormal_operand", _values[14])
        _set_private(_private, "disjoint_box_mass", _values[15])
        _set_private(_private, "proof_authority", _values[17])
        _finish_public_snapshot(
            value,
            _source,
            _DenseFields,
            _values,
            "prepared Dense support",
        )
        _prepared_copy_gate()
        return _private

    def _scan_conv_offset(value: Any) -> Any:
        _prepared_copy_gate()
        _source, _values = _snapshot_public_dict(
            value,
            _ConvOffsetPublicRef,
            _OffsetFields,
            "prepared Conv offset",
        )
        for _index in _range(7):
            if _ExactType(_values[_index]) is not _ExactInt:
                _raise(
                    "INVALID_PREPARED_VALUE",
                    "prepared Conv offset integer metadata changed type",
                )
        _tickets = (
            _scan_private_array(
                _values[7],
                dtype=_I64,
                ndim=1,
                name="prepared Conv output-h indices",
            ),
            _scan_private_array(
                _values[8],
                dtype=_I64,
                ndim=1,
                name="prepared Conv output-w indices",
            ),
            _scan_private_array(
                _values[9],
                dtype=_I64,
                ndim=1,
                name="prepared Conv targets",
            ),
            _scan_private_array(
                _values[10],
                dtype=_F64,
                ndim=1,
                name="prepared Conv offset support",
            ),
            _scan_private_array(
                _values[11],
                dtype=_F64,
                ndim=1,
                name="prepared Conv channel support",
            ),
            _scan_private_array(
                _values[12],
                dtype=_BOOL,
                ndim=1,
                name="prepared Conv support activity",
            ),
        )
        if _ExactType(_values[13]) is not _ExactFloat:
            _raise(
                "INVALID_PREPARED_VALUE",
                "prepared Conv support_sum_upper changed type",
            )
        _prepared_copy_gate()
        return (value, _source, _values, _tickets)

    def _copy_scanned_conv_offset(scanned: Any) -> Any:
        _prepared_copy_gate()
        _value, _source, _values, _tickets = scanned
        _output_h = _copy_ticketed_array(_tickets[0])
        _output_w = _copy_ticketed_array(_tickets[1])
        _targets = _copy_ticketed_array(_tickets[2])
        _support = _copy_ticketed_array(_tickets[3])
        _channel_support = _copy_ticketed_array(_tickets[4])
        _activity = _copy_ticketed_array(_tickets[5])
        _private = _PrivateConvOffset()
        for _index in _range(7):
            _set_private(
                _private, _PrivateOffsetFields[_index], _values[_index]
            )
        _set_private(_private, "output_h_indices", _output_h)
        _set_private(_private, "output_w_indices", _output_w)
        _set_private(_private, "targets", _targets)
        _set_private(_private, "support_flat", _support)
        _set_private(
            _private, "channel_support_flat", _channel_support
        )
        _set_private(
            _private, "support_activity_flat", _activity
        )
        _set_private(
            _private, "support_sum_upper", _values[13]
        )
        _finish_public_snapshot(
            _value,
            _source,
            _OffsetFields,
            _values,
            "prepared Conv offset",
        )
        _prepared_copy_gate()
        return _private

    def _copy_conv_plan(value: Any) -> Any:
        _prepared_copy_gate()
        _source, _values = _snapshot_public_dict(
            value,
            _ConvPlanPublicRef,
            _ConvFields,
            "prepared Conv plan",
        )
        if (
            _ExactType(_values[0]) is not _ExactInt
            or _ExactType(_values[6]) is not _ExactInt
            or _ExactType(_values[11]) is not _ExactBool
        ):
            _raise(
                "INVALID_PREPARED_VALUE",
                "prepared Conv scalar metadata changed type",
            )
        _input_shape = _copy_exact_int_tuple(
            _values[1], length=3, name="prepared Conv input_shape"
        )
        _output_shape = _copy_exact_int_tuple(
            _values[2], length=3, name="prepared Conv output_shape"
        )
        _stride = _copy_exact_int_tuple(
            _values[3], length=2, name="prepared Conv stride"
        )
        _padding = _copy_exact_int_tuple(
            _values[4], length=2, name="prepared Conv padding"
        )
        _dilation = _copy_exact_int_tuple(
            _values[5], length=2, name="prepared Conv dilation"
        )
        _weight_ticket = _scan_private_array(
            _values[7],
            dtype=_F64,
            ndim=4,
            name="prepared Conv weight",
        )
        _support_ticket = _scan_private_array(
            _values[8],
            dtype=_F64,
            ndim=1,
            name="prepared Conv support",
        )
        if _ExactType(_values[9]) is not _ExactTuple:
            _raise(
                "INVALID_PREPARED_VALUE",
                "prepared Conv offsets must be an exact tuple",
            )
        if _len(_values[9]) > _MAX_PREPARED_OFFSETS:
            _raise(
                "RESOURCE_LIMIT",
                "prepared Conv plan exceeds the offset-count budget",
            )
        _total_nbytes = 0
        _total_nbytes = _account_private_array(
            _total_nbytes, _weight_ticket
        )
        _total_nbytes = _account_private_array(
            _total_nbytes, _support_ticket
        )
        _scanned_offsets = []
        for _offset in _values[9]:
            _prepared_copy_gate()
            _scanned = _scan_conv_offset(_offset)
            for _ticket in _scanned[3]:
                _total_nbytes = _account_private_array(
                    _total_nbytes, _ticket
                )
            _scanned_offsets.append(_scanned)
            _prepared_copy_gate()
        # Only this point begins source-proportional copying: all arrays and
        # offsets in the complete plan have already passed admission.
        _prepared_copy_gate()
        _weight = _copy_ticketed_array(_weight_ticket)
        _support = _copy_ticketed_array(_support_ticket)
        _offsets = []
        for _scanned in _scanned_offsets:
            _offsets.append(_copy_scanned_conv_offset(_scanned))
        _offset_tuple = _ExactTuple(_offsets)
        _private = _PrivateConvPlan()
        _set_private(_private, "layer_id", _values[0])
        _set_private(_private, "input_shape", _input_shape)
        _set_private(_private, "output_shape", _output_shape)
        _set_private(_private, "stride", _stride)
        _set_private(_private, "padding", _padding)
        _set_private(_private, "dilation", _dilation)
        _set_private(_private, "groups", _values[6])
        _set_private(_private, "weight", _weight)
        _set_private(_private, "support", _support)
        _set_private(_private, "offsets", _offset_tuple)
        _set_private(_private, "proof_authority", _values[11])
        _finish_public_snapshot(
            value,
            _source,
            _ConvFields,
            _values,
            "prepared Conv plan",
        )
        _prepared_copy_gate()
        return _private

    owner_pid = _getpid()
    owner_deadline = deadline
    operation_lock = _allocate_lock()
    lifecycle_lock = _RLock()
    phase = ["OPEN"]
    generation = [0]
    consumed_generation = [0]
    last_post_failure_generation = [0]
    poison_epoch = [0]
    # state, kind, owner thread, generation, private prepared value,
    # copied raw binding, exact expected dispatch call
    slot = ["IDLE", None, None, 0, None, None, None]
    records = {}
    locator_refs = {}
    physical_keys = {}
    frozen_close_ref = [None]
    frozen_stats_ref = [None]
    frozen_snapshot = {
        "material_count": 0,
        "locator_count": 0,
        "dense_materials": 0,
        "conv_materials": 0,
        "dense_admissions": 0,
        "conv_admissions": 0,
        "dense_executions": 0,
        "conv_executions": 0,
    }
    frozen_snapshot_names = _ExactTuple(frozen_snapshot)
    counters = {
        "dispatch_arms": 0,
        "dispatch_consumes": 0,
        "rejected_operations": 0,
        "post_consume_failures": 0,
        "dense_admissions": 0,
        "conv_admissions": 0,
        "dense_executions": 0,
        "conv_executions": 0,
    }

    def _pid_gate() -> None:
        # This check must precede every adapter lock acquisition.
        if _getpid() != owner_pid:
            _raise(
                "FORKED_PROCESS",
                "a prepared-adapter capability cannot cross a fork",
            )

    def _clear_slot_locked() -> None:
        slot[0] = "IDLE"
        slot[1] = None
        slot[2] = None
        slot[3] = 0
        slot[4] = None
        slot[5] = None
        slot[6] = None

    def _drop_records_locked() -> None:
        records.clear()
        locator_refs.clear()
        physical_keys.clear()

    def _close_frozen_no_raise() -> bool:
        _close = frozen_close_ref[0]
        if _close is not None:
            try:
                _close()
            except _BaseException:
                return False
            return True
        return False

    def _zero_frozen_material_snapshot_locked() -> None:
        frozen_snapshot["material_count"] = 0
        frozen_snapshot["locator_count"] = 0
        frozen_snapshot["dense_materials"] = 0
        frozen_snapshot["conv_materials"] = 0

    def _read_frozen_snapshot() -> Any:
        _stats = frozen_stats_ref[0]
        if _stats is None:
            _raise(
                "FROZEN_STATS_MISMATCH",
                "frozen stats entry point is unavailable",
            )
        try:
            _current = _stats()
        except _BaseException as _exc:
            raise _AdapterError(
                "FROZEN_STATS_MISMATCH",
                "frozen stats call failed",
            ) from None
        if _ExactType(_current) is not _MappingProxyTypeType:
            _raise(
                "FROZEN_STATS_MISMATCH",
                "frozen stats changed exact mapping type",
            )
        _values = []
        try:
            for _name in frozen_snapshot_names:
                _value = _current[_name]
                if _ExactType(_value) is not _ExactInt or _value < 0:
                    _raise(
                        "FROZEN_STATS_MISMATCH",
                        "frozen stats count changed representation",
                    )
                _values.append(_value)
        except _AdapterError:
            raise
        except _BaseException as _exc:
            raise _AdapterError(
                "FROZEN_STATS_MISMATCH",
                "frozen stats fields changed",
            ) from None
        return _ExactTuple(_values)

    def _apply_frozen_snapshot_locked(values: Any) -> None:
        if (
            _ExactType(values) is not _ExactTuple
            or _len(values) != _len(frozen_snapshot_names)
            or phase[0] != "OPEN"
        ):
            _raise(
                "FROZEN_STATS_MISMATCH",
                "frozen stats cannot commit outside the open phase",
            )
        for _index in _range(_len(frozen_snapshot_names)):
            _value = values[_index]
            if _ExactType(_value) is not _ExactInt or _value < 0:
                _raise(
                    "FROZEN_STATS_MISMATCH",
                    "frozen stats commit changed representation",
                )
        for _index in _range(_len(frozen_snapshot_names)):
            _name = frozen_snapshot_names[_index]
            _value = values[_index]
            frozen_snapshot[_name] = _value

    def _poison(
        *,
        rejected: bool,
        post_consume: bool,
    ) -> None:
        _pid_gate()
        _changed = False
        with lifecycle_lock:
            if phase[0] == "OPEN":
                if rejected:
                    counters["rejected_operations"] += 1
                if post_consume:
                    counters["post_consume_failures"] += 1
                    last_post_failure_generation[0] = (
                        consumed_generation[0]
                    )
                poison_epoch[0] += 1
                _clear_slot_locked()
                _drop_records_locked()
                phase[0] = "POISONED"
                _zero_frozen_material_snapshot_locked()
                _changed = True
        if _changed:
            _close_frozen_no_raise()

    def _deadline_gate(*, post_consume: bool = False) -> None:
        _pid_gate()
        if _monotonic() >= owner_deadline:
            _poison(
                rejected=True,
                post_consume=post_consume,
            )
            raise _AdapterTimeout()

    def _live_gate() -> None:
        _pid_gate()
        with lifecycle_lock:
            _current = phase[0]
        if _current != "OPEN":
            _raise(
                "CLOSED" if _current == "CLOSED" else "POISONED",
                "prepared numeric adapter is not open",
            )
        _deadline_gate()

    def _acquire_operation() -> None:
        _pid_gate()
        if not operation_lock.acquire(False):
            with lifecycle_lock:
                if phase[0] == "OPEN":
                    counters["rejected_operations"] += 1
            _raise(
                "CONCURRENT_OPERATION",
                "prepared adapter operation already in progress",
            )

    def _valid_sha(value: Any) -> bool:
        if _ExactType(value) is not _ExactStr or _len(value) != 64:
            return False
        for _character in value:
            if _character not in "0123456789abcdef":
                return False
        return True

    def _binding_copy(value: Any, kind: str, layer_id: Any) -> Any:
        if _ExactType(value) is not _ExactTuple or _len(value) != 13:
            _raise(
                "INVALID_RAW_BINDING",
                "raw_binding must be the exact 13-field tuple ABI",
            )
        if (
            _ExactType(value[0]) is not _ExactBytes
            or value[0] != _raw_binding_tag
            or _ExactType(value[4]) is not _ExactStr
            or _ExactType(value[5]) is not _ExactInt
            or value[5] < 0
            or _ExactType(value[6]) is not _ExactInt
            or value[6] < 0
        ):
            _raise(
                "INVALID_RAW_BINDING",
                "raw_binding tag, branch, or locator fields are malformed",
            )
        for _index in (1, 2, 3, 7, 8, 9, 10, 11, 12):
            if not _valid_sha(value[_index]):
                _raise(
                    "INVALID_RAW_BINDING",
                    "raw_binding SHA fields must be exact lowercase SHA-256",
                )
        if kind == _branch_dense:
            if (
                value[4] != _branch_dense
                or value[8] != _zero_sha
                or value[9] != _zero_sha
                or value[10] != _zero_sha
            ):
                _raise(
                    "RAW_BINDING_KIND_MISMATCH",
                    "Dense raw_binding has wrong branch or nonzero absent fields",
                )
        elif (
            value[4] != _branch_conv
            or value[8] == _zero_sha
            or value[9] == _zero_sha
            or value[10] == _zero_sha
            or _ExactType(layer_id) is not _ExactInt
            or layer_id < 0
            or value[5] != layer_id
        ):
            _raise(
                "RAW_BINDING_KIND_MISMATCH",
                "Conv raw_binding does not bind the exact admission layer",
            )
        # Construct a distinct exact tuple; do not retain or mutate a
        # registry-owned tuple container.
        return _ExactTuple(_item for _item in value)

    def _copy_array_shape(value: Any, name: str) -> Any:
        if _ExactType(value) is not _ndarray_type:
            _raise(
                "INVALID_RAW_CALL",
                "{} must be an exact ndarray".format(name),
            )
        _shape = value.shape
        if _ExactType(_shape) is not _ExactTuple:
            _raise(
                "INVALID_RAW_CALL",
                "{} shape changed exact type".format(name),
            )
        _items = []
        for _extent in _shape:
            if _ExactType(_extent) is not _ExactInt or _extent < 0:
                _raise(
                    "INVALID_RAW_CALL",
                    "{} shape is malformed".format(name),
                )
            _items.append(_extent)
        return _ExactTuple(_items)

    def _conv_expected_call(
        *,
        layer_id: Any,
        weight: Any,
        predecessor_lb: Any,
        predecessor_ub: Any,
        input_shape: Any,
        output_shape: Any,
        stride: Any,
        padding: Any,
        dilation: Any,
        groups: Any,
    ) -> Any:
        if (
            _ExactType(layer_id) is not _ExactInt
            or _ExactType(groups) is not _ExactInt
        ):
            _raise(
                "INVALID_RAW_CALL",
                "Conv layer_id and groups must be exact integers",
            )
        _input = _copy_exact_int_tuple(
            input_shape, length=3, name="raw Conv input_shape"
        )
        _output = _copy_exact_int_tuple(
            output_shape, length=3, name="raw Conv output_shape"
        )
        _stride = _copy_exact_int_tuple(
            stride, length=2, name="raw Conv stride"
        )
        _padding = _copy_exact_int_tuple(
            padding, length=2, name="raw Conv padding"
        )
        _dilation = _copy_exact_int_tuple(
            dilation, length=2, name="raw Conv dilation"
        )
        _packet = (
            _CONV_DISPATCH_TAG,
            layer_id,
            _input,
            _output,
            _stride,
            _padding,
            _dilation,
            groups,
        )
        _shapes = (
            _copy_array_shape(weight, "raw Conv weight"),
            _copy_array_shape(
                predecessor_lb, "raw Conv predecessor lower"
            ),
            _copy_array_shape(
                predecessor_ub, "raw Conv predecessor upper"
            ),
        )
        return (_packet, _shapes)

    def _same_exact_int_tuple(value: Any, expected: Any) -> bool:
        if (
            _ExactType(value) is not _ExactTuple
            or _len(value) != _len(expected)
        ):
            return False
        for _index in _range(_len(expected)):
            if (
                _ExactType(value[_index]) is not _ExactInt
                or value[_index] != expected[_index]
            ):
                return False
        return True

    def _same_conv_dispatch_call(
        packet: Any,
        shapes: Any,
        expected_call: Any,
    ) -> bool:
        if (
            _ExactType(expected_call) is not _ExactTuple
            or _len(expected_call) != 2
            or _ExactType(packet) is not _ExactTuple
            or _len(packet) != 8
            or _ExactType(shapes) is not _ExactTuple
            or _len(shapes) != 3
        ):
            return False
        _expected_packet = expected_call[0]
        _expected_shapes = expected_call[1]
        if (
            _ExactType(packet[0]) is not _ExactBytes
            or packet[0] != _CONV_DISPATCH_TAG
            or _ExactType(packet[1]) is not _ExactInt
            or packet[1] != _expected_packet[1]
            or _ExactType(packet[7]) is not _ExactInt
            or packet[7] != _expected_packet[7]
        ):
            return False
        for _index in _range(2, 7):
            if not _same_exact_int_tuple(
                packet[_index], _expected_packet[_index]
            ):
                return False
        for _index in _range(3):
            if not _same_exact_int_tuple(
                shapes[_index], _expected_shapes[_index]
            ):
                return False
        return True

    def _slot_consumed(wanted_generation: int) -> bool:
        _pid_gate()
        with lifecycle_lock:
            return consumed_generation[0] == wanted_generation

    def _record_post_consume_failure(
        wanted_generation: int,
    ) -> None:
        _pid_gate()
        with lifecycle_lock:
            if (
                consumed_generation[0] == wanted_generation
                and last_post_failure_generation[0]
                != wanted_generation
            ):
                counters["post_consume_failures"] += 1
                last_post_failure_generation[0] = wanted_generation

    def _publication_epoch() -> int:
        _pid_gate()
        with lifecycle_lock:
            if phase[0] != "OPEN":
                _raise(
                    "POISONED",
                    "operation cannot publish after adapter poison",
                )
            return poison_epoch[0]

    def _publication_gate(wanted_epoch: int) -> None:
        _pid_gate()
        with lifecycle_lock:
            if (
                phase[0] != "OPEN"
                or poison_epoch[0] != wanted_epoch
            ):
                _raise(
                    "CONCURRENT_POISON",
                    "operation cannot publish across a poison epoch",
                )

    def _commit_execution(
        *,
        kind: str,
        wanted_epoch: int,
        frozen_values: Any,
    ) -> None:
        _pid_gate()
        with lifecycle_lock:
            if (
                phase[0] != "OPEN"
                or poison_epoch[0] != wanted_epoch
            ):
                _raise(
                    "CONCURRENT_POISON",
                    "execution cannot commit across a terminal epoch",
                )
            if _monotonic() >= owner_deadline:
                raise _AdapterTimeout()
            _apply_frozen_snapshot_locked(frozen_values)
            if kind == _branch_dense:
                counters["dense_executions"] += 1
            elif kind == _branch_conv:
                counters["conv_executions"] += 1
            else:
                _raise(
                    "BRANCH_MISMATCH",
                    "execution commit branch changed",
                )

    def _arm(
        *,
        kind: str,
        prepared: Any,
        raw_binding: Any,
        expected_call: Any,
    ) -> int:
        _pid_gate()
        with lifecycle_lock:
            if phase[0] != "OPEN" or slot[0] != "IDLE":
                _slot_invalid = True
            else:
                _slot_invalid = False
                generation[0] += 1
                _current_generation = generation[0]
                slot[0] = "ARMED"
                slot[1] = kind
                slot[2] = _get_ident()
                slot[3] = _current_generation
                slot[4] = prepared
                slot[5] = raw_binding
                slot[6] = expected_call
                counters["dispatch_arms"] += 1
        if _slot_invalid:
            _poison(rejected=True, post_consume=False)
            _raise(
                "SLOT_NOT_IDLE",
                "prepared dispatch slot is not idle",
            )
        return _current_generation

    def _consume(
        *,
        kind: str,
        args: Any,
        kwargs: Any,
    ) -> Any:
        _pid_gate()
        if (
            _ExactType(args) is not _ExactTuple
            or _len(args) != 2
            or _ExactType(kwargs) is not _ExactDict
            or _len(kwargs) != 1
            or "deadline" not in kwargs
        ):
            _poison(rejected=True, post_consume=False)
            _raise(
                "DISPATCH_CALL_MISMATCH",
                "frozen preparer call signature changed",
            )
        _deadline_gate()
        _thread_id = _get_ident()
        with lifecycle_lock:
            if (
                phase[0] != "OPEN"
                or slot[0] != "ARMED"
                or slot[1] != kind
                or slot[2] != _thread_id
                or slot[3] <= 0
                or slot[3] != generation[0]
                or slot[5] is None
            ):
                _invalid_slot = True
                _prepared = None
            else:
                _invalid_slot = False
                _prepared = slot[4]
                _expected_call = slot[6]
        if _invalid_slot:
            _poison(rejected=True, post_consume=False)
            _raise(
                "DISPATCH_SLOT_MISMATCH",
                "prepared dispatch kind, owner, or generation changed",
            )

        if kind == _branch_dense:
            _call_deadline = _ExactDict.get(kwargs, "deadline")
            _valid_call = (
                _ExactType(args[0]) is _ndarray_type
                and _ExactType(args[1]) is _ndarray_type
                and _ExactType(_call_deadline) is _ExactFloat
                and _call_deadline is owner_deadline
                and _ExactType(_prepared) is _PrivateDenseSupport
            )
        else:
            _call_deadline = _ExactDict.get(kwargs, "deadline")
            _valid_call = (
                _ExactType(_call_deadline) is _ExactFloat
                and _call_deadline is owner_deadline
                and _ExactType(_prepared) is _PrivateConvPlan
                and _same_conv_dispatch_call(
                    args[0], args[1], _expected_call
                )
            )
        if not _valid_call:
            _poison(rejected=True, post_consume=False)
            _raise(
                "DISPATCH_CALL_MISMATCH",
                "frozen preparer raw call or deadline changed",
            )

        with lifecycle_lock:
            if (
                phase[0] != "OPEN"
                or slot[0] != "ARMED"
                or slot[1] != kind
                or slot[2] != _thread_id
                or slot[3] != generation[0]
                or slot[4] is not _prepared
            ):
                _lost_slot = True
            else:
                slot[0] = "CONSUMED"
                slot[4] = None
                consumed_generation[0] = slot[3]
                counters["dispatch_consumes"] += 1
                _lost_slot = False
        if _lost_slot:
            _poison(rejected=True, post_consume=False)
            _raise(
                "DISPATCH_SLOT_MISMATCH",
                "prepared dispatch changed during consumption",
            )
        return _prepared

    def _dense_dispatch(*args: Any, **kwargs: Any) -> Any:
        return _consume(kind=_branch_dense, args=args, kwargs=kwargs)

    def _conv_dispatch(*args: Any, **kwargs: Any) -> Any:
        return _consume(kind=_branch_conv, args=args, kwargs=kwargs)

    # Exact tuple concatenation creates new direct/outer tuples.  Prepared
    # mode replaces every public object-bearing dependency at direct[8:19]
    # and flips only the appended exact-bool mode slot.
    _direct_copy = (
        _direct[:8]
        + (
            _dense_dispatch,
            _conv_dispatch,
            _UnusedLayerType,
            _UnusedBoxType,
            _UnusedDeadlineType,
            _UnusedReplayTimeout,
            _PrivateDenseAdmissionError,
            _UnusedReplayError,
            _PrivateDenseSupport,
            _PrivateConvPlan,
            _PrivateConvOffset,
        )
        + _direct[19:43]
        + (True,)
    )
    _outer_copy = _outer[:4] + (_direct_copy,)
    if (
        _direct_copy is _direct
        or _outer_copy is _outer
        or _ExactType(_direct_copy) is not _ExactTuple
        or _len(_direct_copy) != 44
        or _ExactType(_outer_copy) is not _ExactTuple
        or _len(_outer_copy) != 5
    ):
        _raise(
            "FROZEN_ABI_MISMATCH",
            "sealed dependency tuples were not copied exactly",
        )
    for _index in _range(44):
        if (
            _index
            not in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 43)
            and _direct_copy[_index] is not _direct[_index]
        ):
            _raise(
                "FROZEN_ABI_MISMATCH",
                "a retained frozen dependency changed identity",
            )
    if (
        _direct_copy[8] is not _dense_dispatch
        or _direct_copy[9] is not _conv_dispatch
        or _direct[8] is not _expected_dense_prepare
        or _direct[9] is not _expected_conv_prepare
        or _direct_copy[10] is not _UnusedLayerType
        or _direct_copy[11] is not _UnusedBoxType
        or _direct_copy[12] is not _UnusedDeadlineType
        or _direct_copy[13] is not _UnusedReplayTimeout
        or _direct_copy[14] is not _PrivateDenseAdmissionError
        or _direct_copy[15] is not _UnusedReplayError
        or _direct_copy[16] is not _PrivateDenseSupport
        or _direct_copy[17] is not _PrivateConvPlan
        or _direct_copy[18] is not _PrivateConvOffset
        or _direct_copy[43] is not True
        or _direct[43] is not False
    ):
        _raise(
            "FROZEN_ABI_MISMATCH",
            "prepared dispatch replacement changed the frozen tuple",
        )

    try:
        _implementation_globals = _ExactDict()
        _implementation_globals["__builtins__"] = _ExactDict()
        _implementation_globals["__name__"] = _implementation_spec[6]
        _fresh_implementation = _implementation_spec[4](
            _implementation_spec[0],
            _implementation_globals,
            _implementation_spec[1],
            _implementation_spec[2],
            _implementation_spec[3],
        )
        _frozen_port = _fresh_implementation(
            deadline=owner_deadline,
            _sealed_dependencies=_outer_copy,
        )
    except _PrivateTimeout as _exc:
        raise _AdapterTimeout() from None
    except _PrivateError as _exc:
        raise _AdapterError(
            "FROZEN_FACTORY_REJECTED",
            "frozen factory rejected configuration with exact type {}".format(
                _ExactType(_exc).__name__
            ),
        ) from None
    except _Exception as _exc:
        raise _AdapterError(
            "FROZEN_FACTORY_FAILURE",
            "unexpected frozen factory failure: {}".format(
                _ExactType(_exc).__name__
            ),
        ) from None

    # Re-run the exact-module descriptor gate after construction so a
    # concurrent substitution cannot hide between validation and use.
    _gate_modules()

    # Capture bound entry points once.  Later changes to the factory-local
    # heap class, including any added ``_check_*`` method, cannot redirect
    # these calls.
    _frozen_admit_dense = _frozen_port.admit_dense
    _frozen_admit_conv = _frozen_port.admit_conv
    _frozen_execute_dense = _frozen_port.execute_dense
    _frozen_execute_conv = _frozen_port.execute_conv
    _frozen_stats = _frozen_port.stats
    _frozen_close = _frozen_port.close
    frozen_close_ref[0] = _frozen_close
    frozen_stats_ref[0] = _frozen_stats

    locator_capability = _ExactObject()

    def _make_locator_type() -> Any:
        __slots__ = ("__weakref__",)

        def __new__(cls, capability: Any = None) -> Any:
            if capability is not locator_capability:
                _raise(
                    "LOCATOR_CONSTRUCTION",
                    "prepared adapter locators are factory-minted",
                )
            return _ExactObject.__new__(cls)

        def __copy__(self) -> NoReturn:
            _raise("COPY_FORBIDDEN", "locators cannot be copied")

        def __deepcopy__(self, memo: Any) -> NoReturn:
            del memo
            _raise("COPY_FORBIDDEN", "locators cannot be deep-copied")

        def __reduce__(self) -> NoReturn:
            _raise("COPY_FORBIDDEN", "locators cannot be serialised")

        def __repr__(self) -> str:
            return "<prepared-numeric-locator>"

        @_property
        def proof_authority(self) -> bool:
            return False

        return _ExactType(
            "_Locator",
            (_ExactObject,),
            {
                "__module__": _schema,
                "__slots__": __slots__,
                "__new__": __new__,
                "__copy__": __copy__,
                "__deepcopy__": __deepcopy__,
                "__reduce__": __reduce__,
                "__repr__": __repr__,
                "proof_authority": proof_authority,
            },
        )

    _Locator = _make_locator_type()

    def _drop_locator(
        reference: Any,
        locator_id: int,
        wanted_generation: int,
    ) -> None:
        if _getpid() != owner_pid:
            return
        with lifecycle_lock:
            _record = records.get(locator_id)
            if (
                locator_refs.get(locator_id) is reference
                and _record is not None
                and _record[2] == wanted_generation
            ):
                records.pop(locator_id, None)
                locator_refs.pop(locator_id, None)
                physical_keys.pop(_record[1][3], None)

    def _mint_locator(
        *,
        frozen_locator: Any,
        kind: str,
        raw_binding: Any,
        wanted_generation: int,
        wanted_epoch: int,
        frozen_values: Any,
    ) -> Any:
        _pid_gate()
        _locator = _Locator(locator_capability)
        _locator_id = _id(_locator)
        _record = (
            frozen_locator,
            raw_binding,
            wanted_generation,
            kind,
        )
        _reference = _weakref_ref(
            _locator,
            lambda _current,
            _wanted_id=_locator_id,
            _wanted_generation=wanted_generation: _drop_locator(
                _current, _wanted_id, _wanted_generation
            ),
        )
        with lifecycle_lock:
            if (
                phase[0] != "OPEN"
                or poison_epoch[0] != wanted_epoch
                or slot[0] != "CONSUMED"
                or slot[1] != kind
                or slot[2] != _get_ident()
                or slot[3] != wanted_generation
                or slot[5] is not raw_binding
                or raw_binding[3] in physical_keys
            ):
                _raise(
                    "POST_CONSUME_STATE_MISMATCH",
                    "consumed admission cannot publish a locator",
                )
            if _monotonic() >= owner_deadline:
                raise _AdapterTimeout()
            if (
                _locator_id in records
                or _locator_id in locator_refs
            ):
                _raise(
                    "LOCATOR_COLLISION",
                    "prepared locator identity unexpectedly collided",
                )
            try:
                records[_locator_id] = _record
                physical_keys[raw_binding[3]] = _locator_id
                locator_refs[_locator_id] = _reference
                _apply_frozen_snapshot_locked(frozen_values)
            except _BaseException:
                records.pop(_locator_id, None)
                locator_refs.pop(_locator_id, None)
                physical_keys.pop(raw_binding[3], None)
                raise
            _clear_slot_locked()
            if kind == _branch_dense:
                counters["dense_admissions"] += 1
            else:
                counters["conv_admissions"] += 1
            return _locator

    def _resolve_locator(locator: Any, kind: str) -> Any:
        _pid_gate()
        if _ExactType(locator) is not _Locator:
            _raise(
                "LOCATOR_MISMATCH",
                "locator belongs to another adapter or was transplanted",
            )
        with lifecycle_lock:
            _locator_id = _id(locator)
            _reference = locator_refs.get(_locator_id)
            _record = records.get(_locator_id)
            if (
                _reference is None
                or _reference() is not locator
                or _record is None
                or _record[3] != kind
                or _record[2] <= 0
                or _record[2] > generation[0]
                or _record[1][4] != kind
                or physical_keys.get(_record[1][3]) != _locator_id
            ):
                _raise(
                    "LOCATOR_MISMATCH",
                    "locator generation, kind, or raw binding changed",
                )
            return _record[0]

    port_capability = _ExactObject()
    port_reference = [None]

    def _check_port(value: Any) -> None:
        _pid_gate()
        if (
            _ExactType(value) is not _Port
            or port_reference[0] is None
            or port_reference[0]() is not value
        ):
            _poison(rejected=True, post_consume=False)
            _raise(
                "PORT_MISMATCH",
                "prepared adapter port identity changed",
            )

    def _admission_failure(
        exc: BaseException,
        wanted_generation: int,
    ) -> NoReturn:
        _consumed = _slot_consumed(wanted_generation)
        if _consumed:
            _record_post_consume_failure(wanted_generation)
        _poison(rejected=True, post_consume=False)
        if _ExactType(exc) is _AdapterTimeout or _ExactType(exc) is _PrivateTimeout:
            raise _AdapterTimeout() from None
        if _ExactType(exc) is _AdapterError:
            if _consumed:
                raise _AdapterError(
                    "POST_CONSUME_FAILURE",
                    "frozen admission failed after one-shot consume ({})".format(
                        exc.code
                    ),
                ) from None
            _descriptor = _describe_public_failure(exc)
            raise _AdapterError(
                _descriptor[1], _descriptor[2]
            ) from None
        _code = (
            "POST_CONSUME_FAILURE"
            if _consumed
            else "FROZEN_ADMISSION_REJECTED"
        )
        raise _AdapterError(
            _code,
            "frozen admission raised exact type {}".format(
                _ExactType(exc).__name__
            ),
        ) from None

    def _execution_failure(exc: BaseException) -> NoReturn:
        _poison(rejected=True, post_consume=False)
        if _ExactType(exc) is _AdapterTimeout or _ExactType(exc) is _PrivateTimeout:
            raise _AdapterTimeout() from None
        if _ExactType(exc) is _AdapterError:
            _descriptor = _describe_public_failure(exc)
            raise _AdapterError(
                _descriptor[1], _descriptor[2]
            ) from None
        raise _AdapterError(
            "FROZEN_EXECUTION_REJECTED",
            "frozen execution raised exact type {}".format(
                _ExactType(exc).__name__
            ),
        ) from None

    def _describe_public_failure(exc: BaseException) -> Any:
        if (
            _ExactType(exc) is _AdapterTimeout
            or _ExactType(exc) is _PrivateTimeout
        ):
            return ("TIMEOUT", "", "")
        if _ExactType(exc) is _AdapterError:
            _code = exc.code
            _args = exc.args
            if (
                _ExactType(_code) is _ExactStr
                and _ExactType(_args) is _ExactTuple
                and _len(_args) == 1
                and _ExactType(_args[0]) is _ExactStr
            ):
                _prefix = _code + ": "
                _full = _args[0]
                if _full[: _len(_prefix)] == _prefix:
                    return (
                        "ERROR",
                        _code,
                        _full[_len(_prefix) :],
                    )
            return (
                "ERROR",
                "ADAPTER_ERROR",
                "adapter error representation changed",
            )
        _name = _ExactType(exc).__name__
        if _ExactType(_name) is not _ExactStr:
            _name = "BaseException"
        return (
            "ERROR",
            "ADAPTER_OPERATION_FAILURE",
            "unexpected adapter failure of exact type {}".format(_name),
        )

    def _raise_public_failure(descriptor: Any) -> NoReturn:
        if descriptor[0] == "TIMEOUT":
            raise _AdapterTimeout() from None
        raise _AdapterError(
            descriptor[1], descriptor[2]
        ) from None

    def _new_public_failure(descriptor: Any) -> Any:
        if descriptor[0] == "TIMEOUT":
            return _AdapterTimeout()
        return _AdapterError(descriptor[1], descriptor[2])

    def _cleanup(reference: Any) -> None:
        del reference
        if _getpid() != owner_pid:
            return
        if not operation_lock.acquire(False):
            return
        try:
            with lifecycle_lock:
                if phase[0] == "OPEN":
                    _clear_slot_locked()
                    _drop_records_locked()
                    poison_epoch[0] += 1
                    phase[0] = "CLOSED"
                    _zero_frozen_material_snapshot_locked()
            _close_frozen_no_raise()
        finally:
            operation_lock.release()

    def _stats_mapping_locked() -> Any:
        return _MappingProxyType(
            {
                "state": phase[0],
                "slot_state": slot[0],
                "generation": generation[0],
                "locator_count": _len(records),
                **counters,
                "frozen_material_count": frozen_snapshot[
                    "material_count"
                ],
                "frozen_locator_count": frozen_snapshot[
                    "locator_count"
                ],
                "frozen_dense_materials": frozen_snapshot[
                    "dense_materials"
                ],
                "frozen_conv_materials": frozen_snapshot[
                    "conv_materials"
                ],
                "frozen_dense_admissions": frozen_snapshot[
                    "dense_admissions"
                ],
                "frozen_conv_admissions": frozen_snapshot[
                    "conv_admissions"
                ],
                "frozen_dense_executions": frozen_snapshot[
                    "dense_executions"
                ],
                "frozen_conv_executions": frozen_snapshot[
                    "conv_executions"
                ],
                "proof_authority": False,
            }
        )

    def _make_port_type() -> Any:
        __slots__ = ("__weakref__",)

        def __new__(cls, capability: Any = None) -> Any:
            if capability is not port_capability:
                _raise(
                    "PORT_CONSTRUCTION",
                    "prepared adapter ports are factory-minted",
                )
            return _ExactObject.__new__(cls)

        def __copy__(self) -> NoReturn:
            _raise("COPY_FORBIDDEN", "ports cannot be copied")

        def __deepcopy__(self, memo: Any) -> NoReturn:
            del memo
            _raise("COPY_FORBIDDEN", "ports cannot be deep-copied")

        def __reduce__(self) -> NoReturn:
            _raise("COPY_FORBIDDEN", "ports cannot be serialised")

        @_property
        def proof_authority(self) -> bool:
            return False

        @_property
        def schema(self) -> str:
            return _schema

        def _admit_dense_operation(
            self,
            *,
            weight: Any,
            predecessor_max_abs: Any,
            raw_binding: Any,
            prepared_support: Any,
            tile_width: int = 256,
        ) -> Any:
            _failure = None
            _binding = None
            _private_support = None
            _wanted_epoch = None
            _wanted_generation = None
            _frozen_locator = None
            _frozen_values = None
            _locator = None
            try:
                _pid_gate()
                _acquire_operation()
                try:
                    _check_port(self)
                    _live_gate()
                    try:
                        _binding = _binding_copy(
                            raw_binding, _branch_dense, None
                        )
                        _private_support = _copy_dense_support(
                            prepared_support
                        )
                    except _AdapterError:
                        _poison(rejected=True, post_consume=False)
                        raise
                    except (_MemoryError, _ArrayMemoryError):
                        _poison(rejected=True, post_consume=False)
                        raise _AdapterError(
                            "RESOURCE_LIMIT",
                            "Dense prepared metadata allocation failed",
                        ) from None
                    except _BaseException as _exc:
                        _poison(rejected=True, post_consume=False)
                        raise _AdapterError(
                            "INVALID_PREPARED_VALUE",
                            "Dense prepared value copy failed: {}".format(
                                _ExactType(_exc).__name__
                            ),
                        ) from None
                    _wanted_epoch = _publication_epoch()
                    _wanted_generation = _arm(
                        kind=_branch_dense,
                        prepared=_private_support,
                        raw_binding=_binding,
                        expected_call=None,
                    )
                    try:
                        _frozen_locator = _frozen_admit_dense(
                            weight=weight,
                            predecessor_max_abs=predecessor_max_abs,
                            tile_width=tile_width,
                        )
                        _deadline_gate(post_consume=True)
                        _frozen_values = _read_frozen_snapshot()
                        _deadline_gate(post_consume=True)
                        _locator = _mint_locator(
                            frozen_locator=_frozen_locator,
                            kind=_branch_dense,
                            raw_binding=_binding,
                            wanted_generation=_wanted_generation,
                            wanted_epoch=_wanted_epoch,
                            frozen_values=_frozen_values,
                        )
                        return _locator
                    except _BaseException as _exc:
                        _admission_failure(
                            _exc, _wanted_generation
                        )
                finally:
                    operation_lock.release()
            except _BaseException as _caught:
                _failure = _describe_public_failure(_caught)
            self = None
            weight = None
            predecessor_max_abs = None
            raw_binding = None
            prepared_support = None
            tile_width = 0
            _binding = None
            _private_support = None
            _wanted_epoch = None
            _wanted_generation = None
            _frozen_locator = None
            _frozen_values = None
            _locator = None
            _raise_public_failure(_failure)

        def _admit_conv_operation(
            self,
            *,
            layer_id: int,
            weight: Any,
            predecessor_lb: Any,
            predecessor_ub: Any,
            input_shape: Any,
            output_shape: Any,
            stride: Any,
            padding: Any,
            dilation: Any,
            groups: int,
            raw_binding: Any,
            prepared_plan: Any,
        ) -> Any:
            _failure = None
            _binding = None
            _expected_call = None
            _private_plan = None
            _wanted_epoch = None
            _wanted_generation = None
            _frozen_locator = None
            _frozen_values = None
            _locator = None
            try:
                _pid_gate()
                _acquire_operation()
                try:
                    _check_port(self)
                    _live_gate()
                    try:
                        _binding = _binding_copy(
                            raw_binding, _branch_conv, layer_id
                        )
                        _expected_call = _conv_expected_call(
                            layer_id=layer_id,
                            weight=weight,
                            predecessor_lb=predecessor_lb,
                            predecessor_ub=predecessor_ub,
                            input_shape=input_shape,
                            output_shape=output_shape,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                        )
                        _private_plan = _copy_conv_plan(prepared_plan)
                    except _AdapterError:
                        _poison(rejected=True, post_consume=False)
                        raise
                    except (_MemoryError, _ArrayMemoryError):
                        _poison(rejected=True, post_consume=False)
                        raise _AdapterError(
                            "RESOURCE_LIMIT",
                            "Conv prepared metadata allocation failed",
                        ) from None
                    except _BaseException as _exc:
                        _poison(rejected=True, post_consume=False)
                        raise _AdapterError(
                            "INVALID_PREPARED_VALUE",
                            "Conv prepared value copy failed: {}".format(
                                _ExactType(_exc).__name__
                            ),
                        ) from None
                    _wanted_epoch = _publication_epoch()
                    _wanted_generation = _arm(
                        kind=_branch_conv,
                        prepared=_private_plan,
                        raw_binding=_binding,
                        expected_call=_expected_call,
                    )
                    try:
                        _frozen_locator = _frozen_admit_conv(
                            layer_id=layer_id,
                            weight=weight,
                            predecessor_lb=predecessor_lb,
                            predecessor_ub=predecessor_ub,
                            input_shape=input_shape,
                            output_shape=output_shape,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                        )
                        _deadline_gate(post_consume=True)
                        _frozen_values = _read_frozen_snapshot()
                        _deadline_gate(post_consume=True)
                        _locator = _mint_locator(
                            frozen_locator=_frozen_locator,
                            kind=_branch_conv,
                            raw_binding=_binding,
                            wanted_generation=_wanted_generation,
                            wanted_epoch=_wanted_epoch,
                            frozen_values=_frozen_values,
                        )
                        return _locator
                    except _BaseException as _exc:
                        _admission_failure(
                            _exc, _wanted_generation
                        )
                finally:
                    operation_lock.release()
            except _BaseException as _caught:
                _failure = _describe_public_failure(_caught)
            self = None
            layer_id = 0
            weight = None
            predecessor_lb = None
            predecessor_ub = None
            input_shape = None
            output_shape = None
            stride = None
            padding = None
            dilation = None
            groups = 0
            raw_binding = None
            prepared_plan = None
            _binding = None
            _expected_call = None
            _private_plan = None
            _wanted_epoch = None
            _wanted_generation = None
            _frozen_locator = None
            _frozen_values = None
            _locator = None
            _raise_public_failure(_failure)

        def _execute_dense_operation(self, locator: Any, coefficients: Any) -> Any:
            _failure = None
            _wanted_epoch = None
            _frozen_locator = None
            _result = None
            _frozen_values = None
            try:
                _pid_gate()
                _acquire_operation()
                try:
                    _check_port(self)
                    _live_gate()
                    _wanted_epoch = _publication_epoch()
                    try:
                        _frozen_locator = _resolve_locator(
                            locator, _branch_dense
                        )
                        _result = _frozen_execute_dense(
                            _frozen_locator, coefficients
                        )
                        _deadline_gate()
                        _frozen_values = _read_frozen_snapshot()
                        _deadline_gate()
                        _commit_execution(
                            kind=_branch_dense,
                            wanted_epoch=_wanted_epoch,
                            frozen_values=_frozen_values,
                        )
                    except _BaseException as _exc:
                        _execution_failure(_exc)
                    return _result
                finally:
                    operation_lock.release()
            except _BaseException as _caught:
                _failure = _describe_public_failure(_caught)
            self = None
            locator = None
            coefficients = None
            _wanted_epoch = None
            _frozen_locator = None
            _result = None
            _frozen_values = None
            _raise_public_failure(_failure)

        def _execute_conv_operation(self, locator: Any, coefficients: Any) -> Any:
            _failure = None
            _wanted_epoch = None
            _frozen_locator = None
            _result = None
            _frozen_values = None
            try:
                _pid_gate()
                _acquire_operation()
                try:
                    _check_port(self)
                    _live_gate()
                    _wanted_epoch = _publication_epoch()
                    try:
                        _frozen_locator = _resolve_locator(
                            locator, _branch_conv
                        )
                        _result = _frozen_execute_conv(
                            _frozen_locator, coefficients
                        )
                        _deadline_gate()
                        _frozen_values = _read_frozen_snapshot()
                        _deadline_gate()
                        _commit_execution(
                            kind=_branch_conv,
                            wanted_epoch=_wanted_epoch,
                            frozen_values=_frozen_values,
                        )
                    except _BaseException as _exc:
                        _execution_failure(_exc)
                    return _result
                finally:
                    operation_lock.release()
            except _BaseException as _caught:
                _failure = _describe_public_failure(_caught)
            self = None
            locator = None
            coefficients = None
            _wanted_epoch = None
            _frozen_locator = None
            _result = None
            _frozen_values = None
            _raise_public_failure(_failure)

        def _stats_operation(self) -> Any:
            _failure = None
            _current_phase = None
            _frozen_values = None
            _stats_result = None
            try:
                _pid_gate()
                _acquire_operation()
                try:
                    _check_port(self)
                    with lifecycle_lock:
                        _current_phase = phase[0]
                    if _current_phase == "OPEN":
                        try:
                            _deadline_gate()
                            _frozen_values = _read_frozen_snapshot()
                            _deadline_gate()
                            with lifecycle_lock:
                                if phase[0] != "OPEN":
                                    _raise(
                                        "CONCURRENT_POISON",
                                        "stats crossed a terminal epoch",
                                    )
                                if _monotonic() >= owner_deadline:
                                    raise _AdapterTimeout()
                                _apply_frozen_snapshot_locked(
                                    _frozen_values
                                )
                                _stats_result = _stats_mapping_locked()
                        except _BaseException as _exc:
                            _execution_failure(_exc)
                        return _stats_result
                    with lifecycle_lock:
                        return _stats_mapping_locked()
                finally:
                    operation_lock.release()
            except _BaseException as _caught:
                _failure = _describe_public_failure(_caught)
            self = None
            _current_phase = None
            _frozen_values = None
            _stats_result = None
            _raise_public_failure(_failure)

        def _close_operation(self) -> None:
            _failure = None
            try:
                _pid_gate()
                if not operation_lock.acquire(False):
                    with lifecycle_lock:
                        if phase[0] == "OPEN":
                            counters["rejected_operations"] += 1
                    _raise(
                        "CONCURRENT_OPERATION",
                        "cannot close during an adapter operation",
                    )
                try:
                    _check_port(self)
                    with lifecycle_lock:
                        if phase[0] == "OPEN":
                            _clear_slot_locked()
                            _drop_records_locked()
                            poison_epoch[0] += 1
                            phase[0] = "CLOSED"
                            _zero_frozen_material_snapshot_locked()
                    _close_frozen_no_raise()
                    return
                finally:
                    operation_lock.release()
            except _BaseException as _caught:
                _failure = _describe_public_failure(_caught)
            self = None
            _raise_public_failure(_failure)

        # Public methods are deliberately closure-free.  Their sealed
        # operation/defaults are cleared from the failing frame before a new
        # scalar-only exception is raised, so a retained contender exception
        # cannot traverse a closure back into live winner materials.
        def admit_dense_prepared(
            self,
            _operation=_admit_dense_operation,
            _base_exception=_BaseException,
            _describe=_describe_public_failure,
            _new_failure=_new_public_failure,
            /,
            *,
            weight: Any,
            predecessor_max_abs: Any,
            raw_binding: Any,
            prepared_support: Any,
            tile_width: int = 256,
        ) -> Any:
            _descriptor = None
            _public_failure = None
            try:
                return _operation(
                    self,
                    weight=weight,
                    predecessor_max_abs=predecessor_max_abs,
                    raw_binding=raw_binding,
                    prepared_support=prepared_support,
                    tile_width=tile_width,
                )
            except _base_exception as _caught:
                _descriptor = _describe(_caught)
            _public_failure = _new_failure(_descriptor)
            self = None
            weight = None
            predecessor_max_abs = None
            raw_binding = None
            prepared_support = None
            tile_width = 0
            _operation = None
            _base_exception = None
            _describe = None
            _new_failure = None
            _descriptor = None
            raise _public_failure from None

        def admit_conv_prepared(
            self,
            _operation=_admit_conv_operation,
            _base_exception=_BaseException,
            _describe=_describe_public_failure,
            _new_failure=_new_public_failure,
            /,
            *,
            layer_id: int,
            weight: Any,
            predecessor_lb: Any,
            predecessor_ub: Any,
            input_shape: Any,
            output_shape: Any,
            stride: Any,
            padding: Any,
            dilation: Any,
            groups: int,
            raw_binding: Any,
            prepared_plan: Any,
        ) -> Any:
            _descriptor = None
            _public_failure = None
            try:
                return _operation(
                    self,
                    layer_id=layer_id,
                    weight=weight,
                    predecessor_lb=predecessor_lb,
                    predecessor_ub=predecessor_ub,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    raw_binding=raw_binding,
                    prepared_plan=prepared_plan,
                )
            except _base_exception as _caught:
                _descriptor = _describe(_caught)
            _public_failure = _new_failure(_descriptor)
            self = None
            layer_id = 0
            weight = None
            predecessor_lb = None
            predecessor_ub = None
            input_shape = None
            output_shape = None
            stride = None
            padding = None
            dilation = None
            groups = 0
            raw_binding = None
            prepared_plan = None
            _operation = None
            _base_exception = None
            _describe = None
            _new_failure = None
            _descriptor = None
            raise _public_failure from None

        def execute_dense(
            self,
            locator: Any,
            coefficients: Any,
            _operation=_execute_dense_operation,
            _base_exception=_BaseException,
            _describe=_describe_public_failure,
            _new_failure=_new_public_failure,
            /,
        ) -> Any:
            _descriptor = None
            _public_failure = None
            try:
                return _operation(self, locator, coefficients)
            except _base_exception as _caught:
                _descriptor = _describe(_caught)
            _public_failure = _new_failure(_descriptor)
            self = None
            locator = None
            coefficients = None
            _operation = None
            _base_exception = None
            _describe = None
            _new_failure = None
            _descriptor = None
            raise _public_failure from None

        def execute_conv(
            self,
            locator: Any,
            coefficients: Any,
            _operation=_execute_conv_operation,
            _base_exception=_BaseException,
            _describe=_describe_public_failure,
            _new_failure=_new_public_failure,
            /,
        ) -> Any:
            _descriptor = None
            _public_failure = None
            try:
                return _operation(self, locator, coefficients)
            except _base_exception as _caught:
                _descriptor = _describe(_caught)
            _public_failure = _new_failure(_descriptor)
            self = None
            locator = None
            coefficients = None
            _operation = None
            _base_exception = None
            _describe = None
            _new_failure = None
            _descriptor = None
            raise _public_failure from None

        def stats(
            self,
            _operation=_stats_operation,
            _base_exception=_BaseException,
            _describe=_describe_public_failure,
            _new_failure=_new_public_failure,
            /,
        ) -> Any:
            _descriptor = None
            _public_failure = None
            try:
                return _operation(self)
            except _base_exception as _caught:
                _descriptor = _describe(_caught)
            _public_failure = _new_failure(_descriptor)
            self = None
            _operation = None
            _base_exception = None
            _describe = None
            _new_failure = None
            _descriptor = None
            raise _public_failure from None

        def close(
            self,
            _operation=_close_operation,
            _base_exception=_BaseException,
            _describe=_describe_public_failure,
            _new_failure=_new_public_failure,
            /,
        ) -> None:
            _descriptor = None
            _public_failure = None
            try:
                return _operation(self)
            except _base_exception as _caught:
                _descriptor = _describe(_caught)
            _public_failure = _new_failure(_descriptor)
            self = None
            _operation = None
            _base_exception = None
            _describe = None
            _new_failure = None
            _descriptor = None
            raise _public_failure from None

        _object_setattr(
            admit_dense_prepared,
            "__wrapped__",
            _admit_dense_operation,
        )
        _object_setattr(
            admit_conv_prepared,
            "__wrapped__",
            _admit_conv_operation,
        )
        _object_setattr(
            execute_dense,
            "__wrapped__",
            _execute_dense_operation,
        )
        _object_setattr(
            execute_conv,
            "__wrapped__",
            _execute_conv_operation,
        )
        _object_setattr(stats, "__wrapped__", _stats_operation)
        _object_setattr(close, "__wrapped__", _close_operation)

        return _ExactType(
            "_Port",
            (_ExactObject,),
            {
                "__module__": _schema,
                "__slots__": __slots__,
                "__new__": __new__,
                "__copy__": __copy__,
                "__deepcopy__": __deepcopy__,
                "__reduce__": __reduce__,
                "proof_authority": proof_authority,
                "schema": schema,
                "admit_dense_prepared": admit_dense_prepared,
                "admit_conv_prepared": admit_conv_prepared,
                "execute_dense": execute_dense,
                "execute_conv": execute_conv,
                "stats": stats,
                "close": close,
            },
        )

    _Port = _make_port_type()
    _port = _Port(port_capability)
    port_reference[0] = _weakref_ref(_port, _cleanup)
    return _port


def _seal_prepared_adapter_factory(
    implementation: Any,
    module_gates: Any,
    trusted_builtins: Any,
    direct_dependencies: Any,
) -> Any:
    # Only immutable code/default metadata is retained.  Every factory call
    # executes that code with a fresh globals/builtins dictionary.  All
    # factory-local types are constructed through the sealed exact metaclass;
    # neither this wrapper nor the implementation needs LOAD_BUILD_CLASS.
    implementation_code = implementation.__code__
    implementation_name = implementation.__name__
    implementation_defaults = implementation.__defaults__
    implementation_closure = implementation.__closure__
    function_type = _types_module.FunctionType
    exact_dict = {}.__class__
    exact_tuple = ().__class__
    exact_str = "".__class__
    exact_type = exact_tuple.__class__
    adapter_error_type = PreparedNumericAdapterError
    adapter_timeout_type = PreparedNumericAdapterTimeout
    base_exception_type = _BaseExceptionModule
    builtin_len = _len_module
    module_name = __name__
    sealed_dependencies = (
        module_gates,
        trusted_builtins,
        direct_dependencies,
    )

    def create_prepared_numeric_adapter(*, deadline: float) -> Any:
        """Create one non-authoritative one-shot prepared adapter."""

        failure = None
        private_builtins = None
        private_globals = None
        fresh_implementation = None
        try:
            private_builtins = exact_dict(trusted_builtins)
            private_globals = {
                "__builtins__": private_builtins,
                "__name__": module_name,
            }
            fresh_implementation = function_type(
                implementation_code,
                private_globals,
                implementation_name,
                implementation_defaults,
                implementation_closure,
            )
            return fresh_implementation(
                deadline=deadline,
                _sealed_dependencies=sealed_dependencies,
            )
        except base_exception_type as caught:
            if exact_type(caught) is adapter_timeout_type:
                failure = ("TIMEOUT", "", "")
            elif exact_type(caught) is adapter_error_type:
                code = caught.code
                args = caught.args
                if (
                    exact_type(code) is exact_str
                    and exact_type(args) is exact_tuple
                    and builtin_len(args) == 1
                    and exact_type(args[0]) is exact_str
                ):
                    prefix = code + ": "
                    full = args[0]
                    if full[: builtin_len(prefix)] == prefix:
                        failure = (
                            "ERROR",
                            code,
                            full[builtin_len(prefix) :],
                        )
                if failure is None:
                    failure = (
                        "ERROR",
                        "ADAPTER_ERROR",
                        "adapter error representation changed",
                    )
            else:
                name = exact_type(caught).__name__
                if exact_type(name) is not exact_str:
                    name = "BaseException"
                failure = (
                    "ERROR",
                    "ADAPTER_FACTORY_FAILURE",
                    "unexpected adapter factory failure of exact type {}".format(
                        name
                    ),
                )
        private_builtins = None
        private_globals = None
        fresh_implementation = None
        deadline = 0.0
        if failure[0] == "TIMEOUT":
            raise adapter_timeout_type() from None
        raise adapter_error_type(
            failure[1], failure[2]
        ) from None

    return create_prepared_numeric_adapter


_FROZEN_FACTORY = _private_module.create_private_numeric_kernel
_FROZEN_FACTORY_CODE = _FROZEN_FACTORY.__code__
_FROZEN_FACTORY_CLOSURE = _FROZEN_FACTORY.__closure__
if (
    _ExactTypeModule(_FROZEN_FACTORY) is not _types_module.FunctionType
    or _FROZEN_FACTORY_CODE.co_freevars
    != ("implementation", "sealed_dependencies")
    or _ExactTypeModule(_FROZEN_FACTORY_CLOSURE) is not _ExactTupleModule
    or _len_module(_FROZEN_FACTORY_CLOSURE) != 2
):
    raise PreparedNumericAdapterError(
        "FROZEN_ABI_MISMATCH",
        "private-kernel factory ABI was invalid at adapter import",
    )
_FROZEN_IMPLEMENTATION_SPEC = _FROZEN_FACTORY_CLOSURE[0].cell_contents
_FROZEN_OUTER = _FROZEN_FACTORY_CLOSURE[1].cell_contents
if (
    _ExactTypeModule(_FROZEN_IMPLEMENTATION_SPEC) is not _ExactTupleModule
    or _len_module(_FROZEN_IMPLEMENTATION_SPEC) != 7
    or _ExactTypeModule(_FROZEN_IMPLEMENTATION_SPEC[0])
    is not _types_module.CodeType
    or _FROZEN_IMPLEMENTATION_SPEC[0].co_freevars != ()
    or _ExactTypeModule(_FROZEN_IMPLEMENTATION_SPEC[1])
    is not _ExactStrModule
    or _FROZEN_IMPLEMENTATION_SPEC[2] is not None
    or _FROZEN_IMPLEMENTATION_SPEC[3] is not None
    or _FROZEN_IMPLEMENTATION_SPEC[4] is not _types_module.FunctionType
    or _FROZEN_IMPLEMENTATION_SPEC[5] is not _ExactDictModule
    or _ExactTypeModule(_FROZEN_IMPLEMENTATION_SPEC[6])
    is not _ExactStrModule
    or _ExactTypeModule(_FROZEN_OUTER) is not _ExactTupleModule
    or _len_module(_FROZEN_OUTER) != 5
    or _ExactTypeModule(_FROZEN_OUTER[4]) is not _ExactTupleModule
    or _len_module(_FROZEN_OUTER[4]) != 44
    or _ExactTypeModule(_FROZEN_OUTER[4][43]) is not _ExactBoolModule
    or _FROZEN_OUTER[4][43] is not False
):
    raise PreparedNumericAdapterError(
        "FROZEN_ABI_MISMATCH",
        "private-kernel sealed ABI was invalid at adapter import",
    )
_FROZEN_DIRECT = _FROZEN_OUTER[4]
_FROZEN_DENSE_PREPARE = _FROZEN_DIRECT[8]
_FROZEN_CONV_PREPARE = _FROZEN_DIRECT[9]

_TRUSTED_BUILTINS = (
    ("BaseException", _BaseExceptionModule),
    ("Exception", _ExceptionModule),
    ("MemoryError", _MemoryErrorModule),
    ("TypeError", _TypeErrorModule),
    ("ValueError", _ValueErrorModule),
    ("bool", _ExactBoolModule),
    ("bytes", _ExactBytesModule),
    ("dict", _ExactDictModule),
    ("float", _ExactFloatModule),
    ("id", _id_module),
    ("int", _ExactIntModule),
    ("len", _len_module),
    ("memoryview", _memoryview_module),
    ("object", _ExactObjectModule),
    ("property", _property_module),
    ("range", _range_module),
    ("str", _str_module),
    ("tuple", _ExactTupleModule),
    ("type", _ExactTypeModule),
)

_MODULE_GATES = (
    (_builtins_module, _TRUSTED_BUILTINS),
    (
        _math_module,
        (("isfinite", _math_module.isfinite),),
    ),
    (
        _os_module,
        (("getpid", _os_module.getpid),),
    ),
    (
        _thread_module,
        (
            ("RLock", _thread_module.RLock),
            ("allocate_lock", _thread_module.allocate_lock),
            ("get_ident", _thread_module.get_ident),
        ),
    ),
    (
        _time_module,
        (("monotonic", _time_module.monotonic),),
    ),
    (
        _types_module,
        (
            ("CodeType", _types_module.CodeType),
            ("FunctionType", _types_module.FunctionType),
            ("MappingProxyType", _types_module.MappingProxyType),
            ("ModuleType", _types_module.ModuleType),
        ),
    ),
    (
        _np_module,
        (
            ("frombuffer", _np_module.frombuffer),
            ("ndarray", _np_module.ndarray),
        ),
    ),
    (
        _weakref_module,
        (("ref", _weakref_module.ref),),
    ),
    (
        _private_module,
        (("create_private_numeric_kernel", _FROZEN_FACTORY),),
    ),
)

_DIRECT_DEPENDENCIES = (
    PreparedNumericAdapterError,
    PreparedNumericAdapterTimeout,
    _time_module.monotonic,
    _math_module.isfinite,
    _os_module.getpid,
    _thread_module.get_ident,
    _thread_module.allocate_lock,
    _thread_module.RLock,
    _weakref_module.ref,
    _MappingProxyTypeModule,
    _MappingProxyTypeTypeModule,
    _ModuleTypeModule,
    _ModuleGetattributeModule,
    _types_module.FunctionType,
    _CodeTypeModule,
    _FROZEN_FACTORY,
    _FROZEN_FACTORY_CODE,
    _FROZEN_IMPLEMENTATION_SPEC,
    _FROZEN_OUTER,
    _FROZEN_DIRECT,
    _FROZEN_DENSE_PREPARE,
    _FROZEN_CONV_PREPARE,
    _np_module.ndarray,
    _ObjectGetattributeModule,
    _ObjectSetattrModule,
    _NDArrayTobytesModule,
    _NDArrayReshapeModule,
    _NpFrombufferModule,
    SCHEMA,
    RAW_BINDING_TAG,
    ZERO_SHA256,
    BRANCH_DENSE,
    BRANCH_CONV_DENSE,
)

create_prepared_numeric_adapter = _seal_prepared_adapter_factory(
    _create_prepared_numeric_adapter_impl,
    _MODULE_GATES,
    _TRUSTED_BUILTINS,
    _DIRECT_DEPENDENCIES,
)

del _create_prepared_numeric_adapter_impl
del _seal_prepared_adapter_factory


__all__ = [
    "BRANCH_CONV_DENSE",
    "BRANCH_DENSE",
    "NUMERIC_PROTOCOL",
    "PreparedNumericAdapterError",
    "PreparedNumericAdapterTimeout",
    "RAW_BINDING_TAG",
    "SCHEMA",
    "ZERO_SHA256",
    "create_prepared_numeric_adapter",
]
