# ===- query_dual_replay_v51b_private_decoder.py - Private decoder -----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===-------------------------------------------------------------------===#
"""Strict value-only decoder for the frozen V5.1b private numeric kernel.

This module is deliberately isolated from the replay/session path.  It
decodes only the direct exact-tuple return ABI of
``query_dual_replay_v51b_private_kernel`` and returns a distinct exact tuple
whose array fields are exact, read-only, bytes-backed ``numpy.ndarray``
objects.  The returned value is not a capability, has no proof authority, and
is never accepted as decoder input.

A future session must call the hidden numeric port and this decoder while
holding the same operation lock.  In particular, this decoder does not
authenticate caller-supplied values: the direct-return requirement is a
session ownership rule, not a claim made by this non-authoritative module.

All arithmetic, type, lifecycle, and storage dependencies are captured at
module import.  Factory construction first identity-gates those bindings and
then moves them into closure-private immutable mappings.  Factory-private
types are made through the literal-derived exact builtin metaclass.  The
sealed implementation and all nested code objects contain no ``LOAD_GLOBAL``
or ``LOAD_BUILD_CLASS`` instruction.
"""

from __future__ import annotations

import _thread as _thread_module
import builtins as _builtins_module
import ctypes as _ctypes_module
import math as _math_module
import os as _os_module
import sys as _sys_module
import time as _time_module
import types as _types_module
import weakref as _weakref_module
from builtins import (
    Exception as _ExceptionModule,
    FloatingPointError as _FloatingPointErrorModule,
    MemoryError as _MemoryErrorModule,
    OverflowError as _OverflowErrorModule,
    RuntimeError as _RuntimeErrorModule,
    TypeError as _TypeErrorModule,
    ValueError as _ValueErrorModule,
    id as _id_module,
    len as _len_module,
    property as _property_module,
    range as _range_module,
)
from types import MappingProxyType as _MappingProxyTypeModule
from typing import Any, NoReturn

import numpy as _np_module
from numpy._core import _exceptions as _np_exceptions_module
from numpy._core import _ufunc_config as _ufunc_config_module


SCHEMA = "act.query_dual_replay_v51b_private_numeric_decoder.v1"
DECODER_PROTOCOL = "direct_value_only_dense_conv_v51b"

_DENSE_RAW_TAG = b"act.v51b.private.dense-result.v1"
_CONV_RAW_TAG = b"act.v51b.private.conv-result.v1"
_DENSE_DECODED_TAG = b"act.v51b.private.decoded-dense-result.v1"
_CONV_DECODED_TAG = b"act.v51b.private.decoded-conv-result.v1"

_ExactBytesModule = b"".__class__
_ExactIntModule = (0).__class__
_ExactFloatModule = (0.0).__class__
_ExactBoolModule = (False).__class__
_ExactTupleModule = ().__class__
_ExactStrModule = "".__class__
_ExactDictModule = {}.__class__
_ExactTypeModule = _ExactTupleModule.__class__
_ExactObjectModule = _ExactTupleModule.__base__

_F64DtypeModule = _np_module.dtype(_np_module.float64)
_BoolDtypeModule = _np_module.dtype(_np_module.bool_)
_U64DtypeModule = _np_module.dtype(_np_module.uint64)
_F64TagModule = _F64DtypeModule.str.encode("ascii")
_BoolTagModule = _BoolDtypeModule.str.encode("ascii")
_F64EtaFloatModule = _ExactFloatModule.fromhex(
    "0x0.0000000000001p-1022"
)
_F64HalfModule = _ExactFloatModule.fromhex(
    "0x1.0000000000000p-53"
)
_F64HalfAboveModule = _ExactFloatModule.fromhex(
    "0x1.0000000000001p-53"
)
_MaxDecodeBytesModule = 1 << 30
_ArrayMemoryErrorModule = _np_exceptions_module._ArrayMemoryError
_MachineModule = _os_module.uname().machine
_ObjectGetattributeModule = _ExactObjectModule.__getattribute__
_ModuleTypeModule = _types_module.ModuleType
_ModuleDictGetModule = _ModuleTypeModule.__dict__["__dict__"].__get__
_CanonicalExtobjModule = _ufunc_config_module._make_extobj(
    divide="raise",
    over="raise",
    under="ignore",
    invalid="raise",
    bufsize=8192,
    call=None,
)
_ExtobjSetModule = _ufunc_config_module._extobj_contextvar.set
_ExtobjResetModule = _ufunc_config_module._extobj_contextvar.reset
_FenvLibraryModule = _ctypes_module.CDLL("libm.so.6")
_FegetenvModule = _FenvLibraryModule.fegetenv
_FesetenvModule = _FenvLibraryModule.fesetenv
_FenvBufferTypeModule = _ctypes_module.c_ubyte * 32
_CArrayNewModule = _ctypes_module.Array.__dict__["__new__"]
_CArrayLenModule = _ctypes_module.Array.__dict__["__len__"]
_CArrayGetitemModule = _ctypes_module.Array.__dict__["__getitem__"]
_CArraySetitemModule = _ctypes_module.Array.__dict__["__setitem__"]
_CFuncPtrCallModule = _ctypes_module._CFuncPtr.__dict__["__call__"]
_CFuncPtrRestypeModule = _ctypes_module._CFuncPtr.__dict__["restype"]
_CFuncPtrArgtypesModule = _ctypes_module._CFuncPtr.__dict__["argtypes"]
_CFuncPtrErrcheckModule = _ctypes_module._CFuncPtr.__dict__["errcheck"]
_FenvFunctionTypeModule = _ExactTypeModule(_FegetenvModule)
_GettraceModule = _sys_module.gettrace
_GetprofileModule = _sys_module.getprofile
_MonitoringModule = _sys_module.monitoring
_MonitoringGetToolModule = _MonitoringModule.get_tool
_GatePrimitivesModule = (
    _ObjectGetattributeModule,
    _ModuleTypeModule,
    _ModuleDictGetModule,
    _CanonicalExtobjModule,
    _ExtobjSetModule,
    _ExtobjResetModule,
    _F64EtaFloatModule,
    _F64HalfModule,
    _F64HalfAboveModule,
    _MaxDecodeBytesModule,
    _FenvLibraryModule,
    _FegetenvModule,
    _FesetenvModule,
    _FenvBufferTypeModule,
    _CArrayNewModule,
    _CArrayLenModule,
    _CArrayGetitemModule,
    _CArraySetitemModule,
    _CFuncPtrCallModule,
    _CFuncPtrRestypeModule,
    _CFuncPtrArgtypesModule,
    _CFuncPtrErrcheckModule,
    _FenvFunctionTypeModule,
    _ctypes_module._CFuncPtr,
    _ctypes_module.c_int,
    _ctypes_module.c_ubyte,
    _GettraceModule,
    _GetprofileModule,
    _MonitoringGetToolModule,
    _MachineModule,
)
_UfuncTypeModule = _ExactTypeModule(_np_module.logical_and)
_UfuncReduceDescriptorModule = _UfuncTypeModule.__dict__["reduce"]
_UfuncInstanceStatesModule = (
    (
        _np_module.logical_and,
        _ExactTupleModule(
            _ExactDictModule.items(_np_module.logical_and.__dict__)
        ),
    ),
    (
        _np_module.logical_or,
        _ExactTupleModule(
            _ExactDictModule.items(_np_module.logical_or.__dict__)
        ),
    ),
)
_LogicalAndReduceModule = _UfuncReduceDescriptorModule.__get__(
    _np_module.logical_and, _UfuncTypeModule
)
_LogicalOrReduceModule = _UfuncReduceDescriptorModule.__get__(
    _np_module.logical_or, _UfuncTypeModule
)


class PrivateNumericDecoderError(_RuntimeErrorModule):
    """Stable fail-closed error for the isolated result decoder."""

    __slots__ = ("code",)

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.args = (f"{code}: {message}",)


def _create_private_numeric_result_decoder_impl(
    *,
    deadline: float,
    _sealed_dependencies: Any,
) -> Any:
    """Create one factory-private, non-authoritative decoder port."""

    # Literal-derived exact types remain independent of mutable builtin names.
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
        _trusted_numpy,
        _trusted_math,
        _direct,
    ) = _sealed_dependencies
    (
        _DecoderError,
        _ArrayMemoryError,
        _monotonic,
        _getpid,
        _RLock,
        _weakref_ref,
        _MappingProxyType,
        _schema,
        _dense_raw_tag,
        _conv_raw_tag,
        _dense_decoded_tag,
        _conv_decoded_tag,
        _F64,
        _BOOL,
        _U64,
        _F64_TAG,
        _BOOL_TAG,
        _gate_primitives,
        _UfuncType,
        _ufunc_reduce_descriptor,
        _ufunc_instance_states,
        _logical_and_reduce,
        _logical_or_reduce,
    ) = _direct
    (
        _object_getattribute,
        _ModuleType,
        _module_dict_get,
        _canonical_extobj,
        _extobj_set,
        _extobj_reset,
        _F64_ETA_FLOAT,
        _F64_HALF,
        _F64_HALF_ABOVE,
        _MAX_DECODE_BYTES,
        _fenv_library,
        _fegetenv,
        _fesetenv,
        _FenvBufferType,
        _carray_new,
        _carray_len,
        _carray_getitem,
        _carray_setitem,
        _cfunc_call,
        _cfunc_restype,
        _cfunc_argtypes,
        _cfunc_errcheck,
        _FenvFunctionType,
        _CFuncPtr,
        _CInt,
        _CUByte,
        _gettrace,
        _getprofile,
        _monitoring_get_tool,
        _machine,
    ) = _gate_primitives

    # This branch precedes every call through a dependency.  ``dict.get`` is
    # invoked through the literal-derived exact dict class, so even a
    # substituted builtin is never called.
    _sentinel = _ExactObject()
    for _module, _bindings in _module_gates:
        # Reject a heap-subclassed module before any dynamic attribute lookup.
        # The native descriptor then obtains the real dictionary without
        # invoking a caller-provided ``__getattribute__`` or ``__getattr__``.
        if _ExactType(_module) is not _ModuleType:
            raise _DecoderError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted dependency module changed exact type",
            )
        _module_dict = _module_dict_get(_module, _ModuleType)
        if (
            _ExactType(_module) is not _ModuleType
            or _ExactType(_module_dict) is not _ExactDict
        ):
            raise _DecoderError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted dependency module or dictionary changed type",
            )
        for _name, _trusted in _bindings:
            if (
                _ExactDict.get(_module_dict, _name, _sentinel)
                is not _trusted
            ):
                raise _DecoderError(
                    "DEPENDENCY_SUBSTITUTION",
                    "trusted dependency binding was substituted",
                )

    # NumPy ufunc instances have mutable per-instance dictionaries.  Reject
    # any pre-construction state change and use only import-time
    # descriptor-bound reduce methods thereafter.
    _ufunc_type_dict = _object_getattribute(_UfuncType, "__dict__")
    if (
        _ExactType(_ufunc_type_dict) is not _MappingProxyType
        or _ufunc_type_dict["reduce"] is not _ufunc_reduce_descriptor
    ):
        raise _DecoderError(
            "DEPENDENCY_SUBSTITUTION",
            "trusted NumPy ufunc reduce descriptor was substituted",
        )
    for _ufunc, _trusted_state in _ufunc_instance_states:
        if _ExactType(_ufunc) is not _UfuncType:
            raise _DecoderError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted NumPy ufunc changed exact type",
            )
        _current_state = _object_getattribute(_ufunc, "__dict__")
        if (
            _ExactType(_current_state) is not _ExactDict
            or _ExactDict.__len__(_current_state)
            != _ExactTuple.__len__(_trusted_state)
        ):
            raise _DecoderError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted NumPy ufunc instance state changed",
            )
        for _state_name, _trusted_value in _trusted_state:
            if (
                _ExactDict.get(
                    _current_state, _state_name, _sentinel
                )
                is not _trusted_value
            ):
                raise _DecoderError(
                    "DEPENDENCY_SUBSTITUTION",
                    "trusted NumPy ufunc instance state changed",
                )

    _builtins = _MappingProxyType(_ExactDict(_trusted_builtins))
    _np = _MappingProxyType(_ExactDict(_trusted_numpy))
    _math = _MappingProxyType(_ExactDict(_trusted_math))

    _id = _builtins["id"]
    _len = _builtins["len"]
    _property = _builtins["property"]
    _range = _builtins["range"]
    _Exception = _builtins["Exception"]
    _FloatingPointError = _builtins["FloatingPointError"]
    _MemoryError = _builtins["MemoryError"]
    _TypeError = _builtins["TypeError"]
    _ValueError = _builtins["ValueError"]
    _OverflowError = _builtins["OverflowError"]
    _numpy_version = _np["__version__"]

    _frombuffer = _np["frombuffer"]
    _asarray = _np["asarray"]
    _ascontiguousarray = _np["ascontiguousarray"]
    _zeros = _np["zeros"]
    _isfinite = _np["isfinite"]
    _minimum = _np["minimum"]
    _nextafter = _np["nextafter"]
    _ndarray = _np["ndarray"]
    _ndarray_copy = _ndarray.copy
    _ndarray_reshape = _ndarray.reshape
    _ndarray_tobytes = _ndarray.tobytes
    _ndarray_view = _ndarray.view
    _float64 = _np["float64"]
    _longdouble = _np["longdouble"]
    _inf = _math["inf"]
    _isfinite_scalar = _math["isfinite"]

    def _raise(code: str, message: str) -> NoReturn:
        raise _DecoderError(code, message)

    _fenv_buffer_type_dict = _object_getattribute(
        _FenvBufferType, "__dict__"
    )
    _fenv_function_type_dict = _object_getattribute(
        _FenvFunctionType, "__dict__"
    )
    if (
        _ExactType(_fenv_buffer_type_dict) is not _MappingProxyType
        or _fenv_buffer_type_dict["_length_"] != 32
        or _fenv_buffer_type_dict["_type_"] is not _CUByte
        or _ExactType(_fegetenv) is not _FenvFunctionType
        or _ExactType(_fesetenv) is not _FenvFunctionType
        or _ExactType(_fenv_function_type_dict)
        is not _MappingProxyType
        or _fenv_function_type_dict["_flags_"] != 1
        or _fenv_function_type_dict["_restype_"] is not _CInt
    ):
        _raise(
            "DEPENDENCY_SUBSTITUTION",
            "native floating-environment reader changed representation",
        )

    def _check_fenv_control() -> None:
        if (
            _ExactType(_fegetenv) is not _FenvFunctionType
            or _ExactType(_fesetenv) is not _FenvFunctionType
            or _ExactType(_fenv_buffer_type_dict)
            is not _MappingProxyType
            or _fenv_buffer_type_dict["_length_"] != 32
            or _fenv_buffer_type_dict["_type_"] is not _CUByte
            or _ExactType(_fenv_function_type_dict)
            is not _MappingProxyType
            or _fenv_function_type_dict["_flags_"] != 1
            or _fenv_function_type_dict["_restype_"] is not _CInt
            or _cfunc_restype.__get__(
                _fegetenv, _FenvFunctionType
            )
            is not _CInt
            or _cfunc_restype.__get__(
                _fesetenv, _FenvFunctionType
            )
            is not _CInt
            or _cfunc_argtypes.__get__(
                _fegetenv, _FenvFunctionType
            )
            is not None
            or _cfunc_argtypes.__get__(
                _fesetenv, _FenvFunctionType
            )
            is not None
            or _cfunc_errcheck.__get__(
                _fegetenv, _FenvFunctionType
            )
            is not None
            or _cfunc_errcheck.__get__(
                _fesetenv, _FenvFunctionType
            )
            is not None
        ):
            _raise(
                "DEPENDENCY_SUBSTITUTION",
                "native floating-environment reader state changed",
            )
        try:
            fenv_buffer = _carray_new(_FenvBufferType)
        except _Exception as exc:
            raise _DecoderError(
                "NUMERIC_PLATFORM",
                "could not allocate the native floating-environment record",
            ) from exc
        if (
            _ExactType(fenv_buffer) is not _FenvBufferType
            or _carray_len(fenv_buffer) != 32
        ):
            _raise(
                "DEPENDENCY_SUBSTITUTION",
                "native floating-environment record changed representation",
            )
        try:
            read_status = _cfunc_call(_fegetenv, fenv_buffer)
            if read_status == 0:
                control_word = (
                    _carray_getitem(fenv_buffer, 0)
                    | (_carray_getitem(fenv_buffer, 1) << 8)
                )
                status_word = (
                    _carray_getitem(fenv_buffer, 4)
                    | (_carray_getitem(fenv_buffer, 5) << 8)
                )
                mxcsr = (
                    _carray_getitem(fenv_buffer, 28)
                    | (_carray_getitem(fenv_buffer, 29) << 8)
                    | (_carray_getitem(fenv_buffer, 30) << 16)
                    | (_carray_getitem(fenv_buffer, 31) << 24)
                )
                valid = (
                    control_word & 0x003F == 0x003F
                    and control_word & 0x0C00 == 0
                    and control_word & 0x0300 == 0x0300
                    and mxcsr & 0x1F80 == 0x1F80
                    and mxcsr & 0x6000 == 0
                    and mxcsr & 0x8040 == 0
                )
                pending_unmasked = (
                    status_word & (~control_word) & 0x003F
                )
                mxcsr_pending_unmasked = (
                    (mxcsr & 0x003F)
                    & (~(mxcsr >> 7))
                    & 0x003F
                )
                if (
                    not valid
                    and (
                        pending_unmasked
                        or mxcsr_pending_unmasked
                    )
                ):
                    safe_status = status_word & ~0x80FF
                    _carray_setitem(
                        fenv_buffer, 4, safe_status & 0xFF
                    )
                    _carray_setitem(
                        fenv_buffer, 5, (safe_status >> 8) & 0xFF
                    )
                    safe_mxcsr = mxcsr & ~mxcsr_pending_unmasked
                    _carray_setitem(
                        fenv_buffer, 28, safe_mxcsr & 0xFF
                    )
                    _carray_setitem(
                        fenv_buffer,
                        29,
                        (safe_mxcsr >> 8) & 0xFF,
                    )
                    _carray_setitem(
                        fenv_buffer,
                        30,
                        (safe_mxcsr >> 16) & 0xFF,
                    )
                    _carray_setitem(
                        fenv_buffer,
                        31,
                        (safe_mxcsr >> 24) & 0xFF,
                    )
                    sanitize_status = _cfunc_call(
                        _fesetenv, fenv_buffer
                    )
                else:
                    sanitize_status = 0
            else:
                control_word = 0
                status_word = 0
                mxcsr = 0
                valid = False
                pending_unmasked = 0
                mxcsr_pending_unmasked = 0
                sanitize_status = 0
        except _Exception as exc:
            raise _DecoderError(
                "NUMERIC_PLATFORM",
                "could not read the native floating environment",
            ) from exc
        if (
            _ExactType(read_status) is not _ExactInt
            or read_status != 0
            or _ExactType(control_word) is not _ExactInt
            or _ExactType(status_word) is not _ExactInt
            or _ExactType(mxcsr) is not _ExactInt
            or _ExactType(valid) is not _ExactBool
            or _ExactType(pending_unmasked) is not _ExactInt
            or _ExactType(mxcsr_pending_unmasked)
            is not _ExactInt
            or _ExactType(sanitize_status) is not _ExactInt
            or sanitize_status != 0
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "native floating-environment read failed",
            )
        if not valid:
            _raise(
                "NUMERIC_PLATFORM",
                "x87/MXCSR masks, precision, rounding, or underflow changed",
            )

    def _check_instrumentation() -> None:
        try:
            trace_callback = _gettrace()
            profile_callback = _getprofile()
            monitoring_active = False
            for tool_id in _range(6):
                if _monitoring_get_tool(tool_id) is not None:
                    monitoring_active = True
        except _Exception as exc:
            raise _DecoderError(
                "NUMERIC_PLATFORM",
                "could not inspect Python numeric instrumentation",
            ) from exc
        if (
            trace_callback is not None
            or profile_callback is not None
            or monitoring_active
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "mutable Python instrumentation is active",
            )

    def _enter_numeric_environment() -> Any:
        try:
            return _extobj_set(_canonical_extobj)
        except _Exception as exc:
            raise _DecoderError(
                "NUMERIC_ENVIRONMENT",
                "failed to install the canonical NumPy numeric policy",
            ) from exc

    def _leave_numeric_environment(token: Any) -> None:
        try:
            _extobj_reset(token)
        except _Exception as exc:
            raise _DecoderError(
                "NUMERIC_ENVIRONMENT",
                "failed to restore the caller NumPy numeric policy",
            ) from exc

    def _check_runtime_numeric_environment() -> None:
        _check_fenv_control()
        one = _ExactFloat(1.0)
        if (
            _ExactFloat.hex(one + _F64_HALF)
            != "0x1.0000000000000p+0"
            or _ExactFloat.hex(one + _F64_HALF_ABOVE)
            != "0x1.0000000000001p+0"
            or _ExactFloat.hex(_F64_ETA_FLOAT * one)
            != "0x0.0000000000001p-1022"
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "caller thread is not binary64 RN with gradual underflow",
            )

    if (
        _ExactType(deadline) is not _ExactFloat
        or not _isfinite_scalar(deadline)
    ):
        _raise(
            "INVALID_DEADLINE",
            "deadline must be an exact finite float monotonic timestamp",
        )
    if (
        _ExactType(_schema) is not _ExactStr
        or _ExactType(_numpy_version) is not _ExactStr
        or _numpy_version != "2.3.5"
        or _ExactType(_dense_raw_tag) is not _ExactBytes
        or _ExactType(_conv_raw_tag) is not _ExactBytes
        or _ExactType(_dense_decoded_tag) is not _ExactBytes
        or _ExactType(_conv_decoded_tag) is not _ExactBytes
        or _ExactType(_F64_TAG) is not _ExactBytes
        or _ExactType(_BOOL_TAG) is not _ExactBytes
        or _ExactType(_F64_ETA_FLOAT) is not _ExactFloat
        or _ExactType(_F64_HALF) is not _ExactFloat
        or _ExactType(_F64_HALF_ABOVE) is not _ExactFloat
        or _ExactType(_MAX_DECODE_BYTES) is not _ExactInt
        or _MAX_DECODE_BYTES != 1073741824
        or _ExactType(_machine) is not _ExactStr
        or _machine != "x86_64"
        or not _F64.isnative
        or not _BOOL.isnative
    ):
        _raise(
            "DEPENDENCY_SUBSTITUTION",
            "fixed decoder ABI constants changed representation",
        )

    _check_instrumentation()
    _factory_numeric_token = _enter_numeric_environment()
    try:
        _check_runtime_numeric_environment()
        _check_instrumentation()
        if (
            _ExactFloat.hex(_F64_ETA_FLOAT)
            != "0x0.0000000000001p-1022"
            or _ExactFloat.hex(_F64_HALF)
            != "0x1.0000000000000p-53"
            or _ExactFloat.hex(_F64_HALF_ABOVE)
            != "0x1.0000000000001p-53"
        ):
            _raise(
                "DEPENDENCY_SUBSTITUTION",
                "fixed binary64 probe constants changed representation",
            )
    finally:
        _leave_numeric_environment(_factory_numeric_token)

    _owner_pid = _getpid()
    _end = deadline
    _state = ["OPEN"]
    _operation_lock = _RLock()
    _port_reference = [None]

    def _expire() -> NoReturn:
        _state[0] = "EXPIRED"
        _raise("DEADLINE_EXPIRED", "private decoder deadline expired")

    def _check_live() -> None:
        # PID is checked before acquiring the inherited lock in every public
        # method, so a forked child cannot wait on a parent-owned lock.
        if _getpid() != _owner_pid:
            _raise(
                "FORKED_PROCESS",
                "a private decoder port cannot cross a fork",
            )
        if _state[0] == "EXPIRED":
            _raise("DEADLINE_EXPIRED", "private decoder deadline expired")
        if _state[0] != "OPEN":
            _raise("CLOSED", "private decoder is closed")
        _check_fenv_control()
        if _monotonic() >= _end:
            _expire()

    def _array_all(value: Any) -> bool:
        flat = _ndarray_reshape(
            _asarray(value), (-1,)
        )
        return _ExactBool(
            _logical_and_reduce(flat, axis=None, initial=True)
        )

    def _array_any(value: Any) -> bool:
        flat = _ndarray_reshape(
            _asarray(value), (-1,)
        )
        return _ExactBool(
            _logical_or_reduce(flat, axis=None, initial=False)
        )

    def _same_bits(left: Any, right: Any) -> bool:
        return _ExactBool(
            left.shape == right.shape
            and left.dtype == right.dtype
            and _ndarray_tobytes(left, order="C")
            == _ndarray_tobytes(right, order="C")
        )

    def _bytes_backed_readonly(value: Any) -> bool:
        current = value
        depth = 0
        while _ExactType(current) is _ndarray:
            if current.flags.writeable or depth > 4:
                return False
            current = current.base
            depth += 1
        return _ExactType(current) is _ExactBytes

    def _validate_expectation(
        expected_rows: Any, expected_width: Any
    ) -> tuple[int, int]:
        if (
            _ExactType(expected_rows) is not _ExactInt
            or _ExactType(expected_width) is not _ExactInt
            or expected_rows <= 0
            or expected_width <= 0
        ):
            _raise(
                "INVALID_EXPECTATION",
                "expected rows and width must be exact positive integers",
            )
        return expected_rows, expected_width

    def _decode_frame(
        frame: Any,
        *,
        expected_shape: Any,
        dtype: Any,
        dtype_tag: Any,
        name: str,
    ) -> Any:
        # Exact outer type precedes len, indexing, attribute access, payload
        # iteration, or any conversion protocol.
        if _ExactType(frame) is not _ExactTuple:
            _raise("INVALID_FRAME", f"{name} frame type is not exact tuple")
        if _len(frame) != 3:
            _raise("INVALID_FRAME", f"{name} frame length is not three")
        payload = frame[0]
        shape = frame[1]
        tag = frame[2]
        if (
            _ExactType(payload) is not _ExactBytes
            or _ExactType(shape) is not _ExactTuple
            or _ExactType(tag) is not _ExactBytes
        ):
            _raise(
                "INVALID_FRAME",
                f"{name} frame fields are not exact builtin values",
            )
        if _len(shape) != _len(expected_shape):
            _raise("INVALID_FRAME", f"{name} shape does not match session")
        for _extent in shape:
            if _ExactType(_extent) is not _ExactInt or _extent <= 0:
                _raise(
                    "INVALID_FRAME",
                    f"{name} shape contains an invalid extent",
                )
        # Tuple equality is intentionally delayed until every element is an
        # exact int; otherwise a foreign element could run ``__eq__``.
        if shape != expected_shape:
            _raise("INVALID_FRAME", f"{name} shape does not match session")
        if tag != dtype_tag:
            _raise("INVALID_FRAME", f"{name} dtype tag is invalid")
        _count = 1
        for _extent in shape:
            _count *= _extent
        _byte_count = _count * dtype.itemsize
        if _byte_count > _MAX_DECODE_BYTES:
            _raise(
                "RESOURCE_LIMIT",
                f"{name} exceeds the fixed decode byte budget",
            )
        if _len(payload) != _byte_count:
            _raise("INVALID_FRAME", f"{name} payload length is invalid")
        if dtype == _BOOL:
            for _byte in payload:
                if _byte != 0 and _byte != 1:
                    _raise(
                        "INVALID_FRAME",
                        f"{name} boolean payload is not canonical",
                    )
        try:
            _raw = _frombuffer(payload, dtype=dtype)
            _value = _ndarray_reshape(_raw, shape)
        except (_TypeError, _ValueError, _OverflowError) as _exc:
            raise _DecoderError(
                "INVALID_FRAME",
                f"{name} payload could not be reconstructed",
            ) from _exc
        if (
            _ExactType(_raw) is not _ndarray
            or _ExactType(_value) is not _ndarray
            or _value.dtype != dtype
            or _value.shape != expected_shape
            or (dtype == _F64 and not _value.dtype.isnative)
            or not _bytes_backed_readonly(_value)
        ):
            _raise(
                "INVALID_FRAME",
                f"{name} did not reconstruct as immutable native storage",
            )
        _check_live()
        return _value

    def _decode_envelope(
        result: Any,
        *,
        raw_tag: Any,
    ) -> None:
        # A decoded tuple has a distinct tag and therefore fails this gate.
        if _ExactType(result) is not _ExactTuple:
            _raise("INVALID_ENVELOPE", "result type is not exact tuple")
        if _len(result) != 9:
            _raise("INVALID_ENVELOPE", "result length is not nine")
        if (
            _ExactType(result[0]) is not _ExactBytes
            or result[0] != raw_tag
            or result[1] is not False
        ):
            _raise(
                "INVALID_ENVELOPE",
                "result tag or authority field is invalid",
            )

    def _require_finite(value: Any, name: str) -> None:
        if not _array_all(_isfinite(value)):
            _raise("NONFINITE", f"{name} contains a non-finite value")

    def _require_nonnegative(value: Any, name: str) -> None:
        _require_finite(value, name)
        if _array_any(value < 0.0):
            _raise("NEGATIVE_GUARD", f"{name} contains a negative value")

    def _require_positive_zero(
        value: Any, inactive: Any, name: str
    ) -> None:
        _bits = _ndarray_view(value, _U64)
        if _array_any(_bits[inactive] != 0):
            _raise(
                "SEMANTIC_MISMATCH",
                f"{name} is not exact positive zero when inactive",
            )

    def _require_positive_when_active(
        value: Any, active: Any, name: str
    ) -> None:
        if _array_any(active & (value == 0.0)):
            _raise(
                "SEMANTIC_MISMATCH",
                f"{name} is not strictly positive when active",
            )

    def _decode_dense(
        result: Any,
        expected_rows: Any,
        expected_width: Any,
    ) -> Any:
        _rows, _width = _validate_expectation(
            expected_rows, expected_width
        )
        if (
            _rows * _width * 8 + _rows * (4 * 8 + 2)
            > _MAX_DECODE_BYTES
        ):
            _raise(
                "RESOURCE_LIMIT",
                "Dense result exceeds the aggregate decode budget",
            )
        _decode_envelope(result, raw_tag=_dense_raw_tag)
        _row_shape = (_rows,)
        _nominal = _decode_frame(
            result[2],
            expected_shape=(_rows, _width),
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Dense nominal",
        )
        _support = _decode_frame(
            result[3],
            expected_shape=_row_shape,
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Dense support mass",
        )
        _wide = _decode_frame(
            result[4],
            expected_shape=_row_shape,
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Dense wide guard",
        )
        _streamed = _decode_frame(
            result[5],
            expected_shape=_row_shape,
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Dense streamed guard",
        )
        _final = _decode_frame(
            result[6],
            expected_shape=_row_shape,
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Dense final guard",
        )
        _active = _decode_frame(
            result[7],
            expected_shape=_row_shape,
            dtype=_BOOL,
            dtype_tag=_BOOL_TAG,
            name="Dense active mask",
        )
        _fallback = _decode_frame(
            result[8],
            expected_shape=_row_shape,
            dtype=_BOOL,
            dtype_tag=_BOOL_TAG,
            name="Dense fallback mask",
        )

        _require_finite(_nominal, "Dense nominal")
        _require_nonnegative(_support, "Dense support mass")
        _require_nonnegative(_wide, "Dense wide guard")
        _require_nonnegative(_streamed, "Dense streamed guard")
        _require_nonnegative(_final, "Dense final guard")
        if not _array_all(_active == (_support != 0.0)):
            _raise(
                "SEMANTIC_MISMATCH",
                "Dense active mask does not match support mass",
            )
        if _array_any(_fallback & ~_active):
            _raise(
                "SEMANTIC_MISMATCH",
                "Dense fallback mask is not a subset of active mask",
            )
        _inactive = ~_active
        _require_positive_zero(_support, _inactive, "Dense support mass")
        _require_positive_zero(_wide, _inactive, "Dense wide guard")
        _require_positive_zero(
            _streamed, _inactive, "Dense streamed guard"
        )
        _require_positive_zero(_final, _inactive, "Dense final guard")
        # The frozen kernel's active predicate witnesses a nonzero
        # coefficient/support product.  Its directed gamma/tau construction
        # and, on fallback rows, its directed tiled radius therefore produce
        # strictly positive guards.  Enforcing this invariant prevents an
        # internally inconsistent active row from silently losing all error
        # allowance while retaining otherwise matching masks.
        _require_positive_when_active(
            _wide, _active, "Dense wide guard"
        )
        _require_positive_when_active(
            _streamed, _active, "Dense streamed guard"
        )
        _require_positive_when_active(
            _final, _active, "Dense final guard"
        )
        if not _same_bits(
            _streamed[~_fallback], _wide[~_fallback]
        ):
            _raise(
                "SEMANTIC_MISMATCH",
                "Dense non-fallback streamed guard does not match wide guard",
            )

        _expected_final = _ndarray_copy(_wide)
        if _array_any(_fallback):
            _expected_final[_fallback] = _minimum(
                _wide[_fallback], _streamed[_fallback]
            )
        _expected_final[_inactive] = 0.0
        if not _same_bits(_final, _expected_final):
            _raise(
                "SEMANTIC_MISMATCH",
                "Dense final guard does not match the fixed bit-level rule",
            )
        _check_live()
        return (
            _dense_decoded_tag,
            False,
            _nominal,
            _support,
            _wide,
            _streamed,
            _final,
            _active,
            _fallback,
        )

    def _ceil_f64(value: Any) -> Any:
        _wide = _asarray(value, dtype=_longdouble)
        if not _array_all(_isfinite(_wide)):
            _raise(
                "NONFINITE",
                "decoder wide directed sum is non-finite",
            )
        _nearest = _asarray(_wide, dtype=_float64)
        if not _array_all(_isfinite(_nearest)):
            _raise(
                "NONFINITE",
                "decoder directed sum does not fit in binary64",
            )
        _below = _asarray(_nearest, dtype=_longdouble) < _wide
        if _array_any(_below):
            _nearest = _ascontiguousarray(_nearest, dtype=_float64)
            _nearest[_below] = _nextafter(
                _nearest[_below], _float64(_inf)
            )
        if not _array_all(_isfinite(_nearest)):
            _raise(
                "NONFINITE",
                "decoder outward successor is non-finite",
            )
        return _ascontiguousarray(_nearest, dtype=_float64)

    def _zero_sum(left: Any, right: Any) -> Any:
        _left_active = left != 0.0
        _right_active = right != 0.0
        _both = _left_active & _right_active
        _result = _zeros(left.shape, dtype=_float64)
        _only_left = _left_active & ~_right_active
        _only_right = _right_active & ~_left_active
        _result[_only_left] = left[_only_left]
        _result[_only_right] = right[_only_right]
        if _array_any(_both):
            _wide = _nextafter(
                _asarray(left[_both], dtype=_longdouble)
                + _asarray(right[_both], dtype=_longdouble),
                _longdouble(_inf),
                dtype=_longdouble,
            )
            _result[_both] = _ceil_f64(_wide)
        return _result

    def _decode_conv(
        result: Any,
        expected_rows: Any,
        expected_width: Any,
    ) -> Any:
        _rows, _width = _validate_expectation(
            expected_rows, expected_width
        )
        if (
            _rows * _width * 8 + _rows * (3 * 8 + 3)
            > _MAX_DECODE_BYTES
        ):
            _raise(
                "RESOURCE_LIMIT",
                "Conv result exceeds the aggregate decode budget",
            )
        _decode_envelope(result, raw_tag=_conv_raw_tag)
        _row_shape = (_rows,)
        _coefficient = _decode_frame(
            result[2],
            expected_shape=(_rows, _width),
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Conv coefficient",
        )
        _scalar = _decode_frame(
            result[3],
            expected_shape=_row_shape,
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Conv scalar guard",
        )
        _channel = _decode_frame(
            result[4],
            expected_shape=_row_shape,
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Conv channel guard",
        )
        _accumulation = _decode_frame(
            result[5],
            expected_shape=_row_shape,
            dtype=_F64,
            dtype_tag=_F64_TAG,
            name="Conv accumulation guard",
        )
        _active = _decode_frame(
            result[6],
            expected_shape=_row_shape,
            dtype=_BOOL,
            dtype_tag=_BOOL_TAG,
            name="Conv active mask",
        )
        _channel_active = _decode_frame(
            result[7],
            expected_shape=_row_shape,
            dtype=_BOOL,
            dtype_tag=_BOOL_TAG,
            name="Conv channel active mask",
        )
        _accumulation_active = _decode_frame(
            result[8],
            expected_shape=_row_shape,
            dtype=_BOOL,
            dtype_tag=_BOOL_TAG,
            name="Conv accumulation active mask",
        )

        _require_finite(_coefficient, "Conv coefficient")
        _require_nonnegative(_scalar, "Conv scalar guard")
        _require_nonnegative(_channel, "Conv channel guard")
        _require_nonnegative(
            _accumulation, "Conv accumulation guard"
        )
        if not _array_all(_channel_active == (_channel != 0.0)):
            _raise(
                "SEMANTIC_MISMATCH",
                "Conv channel mask does not match its guard",
            )
        if not _array_all(
            _accumulation_active == (_accumulation != 0.0)
        ):
            _raise(
                "SEMANTIC_MISMATCH",
                "Conv accumulation mask does not match its guard",
            )
        if not _array_all(
            _active == (_channel_active | _accumulation_active)
        ):
            _raise(
                "SEMANTIC_MISMATCH",
                "Conv active mask is not the component-mask union",
            )
        _require_positive_zero(
            _scalar, ~_active, "Conv scalar guard"
        )
        _require_positive_zero(
            _channel, ~_channel_active, "Conv channel guard"
        )
        _require_positive_zero(
            _accumulation,
            ~_accumulation_active,
            "Conv accumulation guard",
        )
        _expected_scalar = _zero_sum(_channel, _accumulation)
        _expected_scalar[~_active] = 0.0
        if not _same_bits(_scalar, _expected_scalar):
            _raise(
                "SEMANTIC_MISMATCH",
                "Conv scalar guard does not match directed zero-sum order",
            )
        _check_live()
        return (
            _conv_decoded_tag,
            False,
            _coefficient,
            _scalar,
            _channel,
            _accumulation,
            _active,
            _channel_active,
            _accumulation_active,
        )

    _port_capability = _ExactObject()

    def _check_port_identity(value: Any) -> None:
        if (
            _ExactType(value) is not _Port
            or _port_reference[0] is None
            or _port_reference[0]() is not value
        ):
            _raise(
                "PORT_MISMATCH",
                "private decoder port identity changed",
            )
        if _getpid() != _owner_pid:
            _raise(
                "FORKED_PROCESS",
                "a private decoder port cannot cross a fork",
            )

    def _check_port(value: Any) -> None:
        _check_port_identity(value)
        _check_live()

    def _numeric_operation(operation: Any) -> Any:
        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            _check_port(self)
            _check_instrumentation()
            token = _enter_numeric_environment()
            try:
                result = operation(self, *args, **kwargs)
                _check_instrumentation()
                return result
            except _FloatingPointError as exc:
                if _ExactType(exc) is not _FloatingPointError:
                    raise
                raise _DecoderError(
                    "NONFINITE",
                    "decoder arithmetic raised a floating-point exception",
                ) from exc
            except _OverflowError as exc:
                if _ExactType(exc) is not _OverflowError:
                    raise
                raise _DecoderError(
                    "NONFINITE",
                    "decoder arithmetic exceeded a numeric platform limit",
                ) from exc
            except _MemoryError as exc:
                if (
                    _ExactType(exc) is not _MemoryError
                    and _ExactType(exc) is not _ArrayMemoryError
                ):
                    raise
                raise _DecoderError(
                    "RESOURCE_LIMIT",
                    "decoder exceeded the fixed memory budget",
                ) from exc
            finally:
                _leave_numeric_environment(token)

        return guarded

    def _make_port_type() -> Any:
        __slots__ = ("__weakref__",)

        def __new__(cls, capability: Any = None) -> Any:
            if capability is not _port_capability:
                _raise(
                    "PORT_CONSTRUCTION",
                    "private decoder ports are factory-minted",
                )
            return _ExactObject.__new__(cls)

        def __copy__(self) -> NoReturn:
            _raise("COPY_FORBIDDEN", "decoder ports cannot be copied")

        def __deepcopy__(self, memo: Any) -> NoReturn:
            del memo
            _raise(
                "COPY_FORBIDDEN",
                "decoder ports cannot be deep-copied",
            )

        def __reduce__(self) -> NoReturn:
            _raise(
                "COPY_FORBIDDEN",
                "decoder ports cannot be serialised",
            )

        def __repr__(self) -> str:
            return "<private-numeric-result-decoder>"

        @_property
        def proof_authority(self) -> bool:
            return False

        @_property
        def schema(self) -> str:
            return _schema

        def _check_self(self) -> None:
            _check_port(self)

        def decode_dense(
            self,
            result: Any,
            *,
            expected_rows: int,
            expected_width: int,
        ) -> Any:
            """Decode one direct Dense kernel return under the port lock."""

            _check_port(self)
            with _operation_lock:
                _check_port(self)
                _check_runtime_numeric_environment()
                _decoded = _decode_dense(
                    result, expected_rows, expected_width
                )
                _check_runtime_numeric_environment()
                _check_port(self)
                return _decoded

        def decode_conv(
            self,
            result: Any,
            *,
            expected_rows: int,
            expected_width: int,
        ) -> Any:
            """Decode one direct dense-Conv return under the port lock."""

            _check_port(self)
            with _operation_lock:
                _check_port(self)
                _check_runtime_numeric_environment()
                _decoded = _decode_conv(
                    result, expected_rows, expected_width
                )
                _check_runtime_numeric_environment()
                _check_port(self)
                return _decoded

        def close(self) -> None:
            _check_port_identity(self)
            with _operation_lock:
                _check_port_identity(self)
                if _state[0] == "OPEN":
                    _state[0] = "CLOSED"

        guarded_decode_dense = _numeric_operation(decode_dense)
        guarded_decode_conv = _numeric_operation(decode_conv)

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
                "__repr__": __repr__,
                "proof_authority": proof_authority,
                "schema": schema,
                "_check_self": _check_self,
                "decode_dense": guarded_decode_dense,
                "decode_conv": guarded_decode_conv,
                "close": close,
            },
        )

    _Port = _make_port_type()
    _port = _Port(_port_capability)
    _port_reference[0] = _weakref_ref(_port)
    _check_live()
    _check_instrumentation()
    return _port


def _seal_private_numeric_result_decoder(
    implementation: Any,
    module_gates: Any,
    trusted_builtins: Any,
    trusted_numpy: Any,
    trusted_math: Any,
    direct_dependencies: Any,
) -> Any:
    # Keep immutable implementation components only.  Every construction
    # receives a fresh function with fresh empty builtin/global dictionaries,
    # so the public factory closure exposes no long-lived mutable function.
    implementation = (
        implementation.__code__,
        implementation.__name__,
        implementation.__defaults__,
        implementation.__closure__,
        _ExactTypeModule(implementation),
        _ExactDictModule,
        implementation.__globals__["__name__"],
    )
    sealed_dependencies = (
        module_gates,
        trusted_builtins,
        trusted_numpy,
        trusted_math,
        direct_dependencies,
    )

    def create_private_numeric_result_decoder(*, deadline: float) -> Any:
        """Return one dependency-sealed private result decoder."""

        (
            _code,
            _name,
            _defaults,
            _closure,
            _FunctionType,
            _ExactDict,
            _module_name,
        ) = implementation
        _implementation_globals = _ExactDict()
        _implementation_globals["__builtins__"] = _ExactDict()
        _implementation_globals["__name__"] = _module_name
        _fresh_implementation = _FunctionType(
            _code,
            _implementation_globals,
            _name,
            _defaults,
            _closure,
        )
        return _fresh_implementation(
            deadline=deadline,
            _sealed_dependencies=sealed_dependencies,
        )

    return create_private_numeric_result_decoder


_TRUSTED_BUILTINS = (
    ("Exception", _ExceptionModule),
    ("FloatingPointError", _FloatingPointErrorModule),
    ("MemoryError", _MemoryErrorModule),
    ("OverflowError", _OverflowErrorModule),
    ("RuntimeError", _RuntimeErrorModule),
    ("TypeError", _TypeErrorModule),
    ("ValueError", _ValueErrorModule),
    ("bool", _ExactBoolModule),
    ("bytes", _ExactBytesModule),
    ("dict", _ExactDictModule),
    ("float", _ExactFloatModule),
    ("id", _id_module),
    ("int", _ExactIntModule),
    ("len", _len_module),
    ("object", _ExactObjectModule),
    ("property", _property_module),
    ("range", _range_module),
    ("str", _ExactStrModule),
    ("tuple", _ExactTupleModule),
    ("type", _ExactTypeModule),
)

_TRUSTED_NUMPY = (
    ("__version__", _np_module.__version__),
    ("asarray", _np_module.asarray),
    ("ascontiguousarray", _np_module.ascontiguousarray),
    ("bool_", _np_module.bool_),
    ("dtype", _np_module.dtype),
    ("float64", _np_module.float64),
    ("frombuffer", _np_module.frombuffer),
    ("isfinite", _np_module.isfinite),
    ("logical_and", _np_module.logical_and),
    ("logical_or", _np_module.logical_or),
    ("longdouble", _np_module.longdouble),
    ("minimum", _np_module.minimum),
    ("ndarray", _np_module.ndarray),
    ("nextafter", _np_module.nextafter),
    ("uint64", _np_module.uint64),
    ("zeros", _np_module.zeros),
)

_TRUSTED_MATH = (
    ("inf", _math_module.inf),
    ("isfinite", _math_module.isfinite),
)

_MODULE_GATES = (
    (_builtins_module, _TRUSTED_BUILTINS),
    (
        _ctypes_module,
        (
            ("Array", _ctypes_module.Array),
            ("CDLL", _ctypes_module.CDLL),
            ("_CFuncPtr", _ctypes_module._CFuncPtr),
            ("c_int", _ctypes_module.c_int),
            ("c_ubyte", _ctypes_module.c_ubyte),
        ),
    ),
    (_np_module, _TRUSTED_NUMPY),
    (
        _np_exceptions_module,
        (
            ("_ArrayMemoryError", _ArrayMemoryErrorModule),
        ),
    ),
    (_math_module, _TRUSTED_MATH),
    (
        _os_module,
        (
            ("getpid", _os_module.getpid),
            ("uname", _os_module.uname),
        ),
    ),
    (_thread_module, (("RLock", _thread_module.RLock),)),
    (
        _sys_module,
        (
            ("getprofile", _GetprofileModule),
            ("gettrace", _GettraceModule),
            ("monitoring", _MonitoringModule),
        ),
    ),
    (
        _MonitoringModule,
        (("get_tool", _MonitoringGetToolModule),),
    ),
    (_time_module, (("monotonic", _time_module.monotonic),)),
    (
        _types_module,
        (
            ("FunctionType", _types_module.FunctionType),
            ("MappingProxyType", _types_module.MappingProxyType),
            ("ModuleType", _types_module.ModuleType),
        ),
    ),
    (_weakref_module, (("ref", _weakref_module.ref),)),
    (
        _ufunc_config_module,
        (
            (
                "_extobj_contextvar",
                _ufunc_config_module._extobj_contextvar,
            ),
            ("_make_extobj", _ufunc_config_module._make_extobj),
        ),
    ),
)

_DIRECT_DEPENDENCIES = (
    PrivateNumericDecoderError,
    _ArrayMemoryErrorModule,
    _time_module.monotonic,
    _os_module.getpid,
    _thread_module.RLock,
    _weakref_module.ref,
    _MappingProxyTypeModule,
    SCHEMA,
    _DENSE_RAW_TAG,
    _CONV_RAW_TAG,
    _DENSE_DECODED_TAG,
    _CONV_DECODED_TAG,
    _F64DtypeModule,
    _BoolDtypeModule,
    _U64DtypeModule,
    _F64TagModule,
    _BoolTagModule,
    _GatePrimitivesModule,
    _UfuncTypeModule,
    _UfuncReduceDescriptorModule,
    _UfuncInstanceStatesModule,
    _LogicalAndReduceModule,
    _LogicalOrReduceModule,
)

create_private_numeric_result_decoder = (
    _seal_private_numeric_result_decoder(
        _create_private_numeric_result_decoder_impl,
        _MODULE_GATES,
        _TRUSTED_BUILTINS,
        _TRUSTED_NUMPY,
        _TRUSTED_MATH,
        _DIRECT_DEPENDENCIES,
    )
)

del _create_private_numeric_result_decoder_impl
del _seal_private_numeric_result_decoder


__all__ = [
    "DECODER_PROTOCOL",
    "PrivateNumericDecoderError",
    "SCHEMA",
    "create_private_numeric_result_decoder",
]
