# ===- query_dual_replay_v51b_private_kernel.py - Private V5.1b kernels ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===-------------------------------------------------------------------===#
"""Isolated, closure-owned V5.1b Dense and dense-Conv numeric kernels.

This module is deliberately not connected to the production replay/session
path.  It tests a narrow productisation claim: material that has already
passed the exhaustive V5.1a admission checks can be copied into one
factory-local, bytes-backed core and subsequently executed without consulting
the public V5.1a validators, hashes, manifests, receipts, or diagnostics.

The only public constructor returns a factory-local port.  Numeric cores are
exact closure-owned tuples, the locator type is lexical to that factory, no
``get_core``-style API exists, and an opaque locator is accepted only by the
port that minted it.  Exact ndarray type checks happen before any ndarray
attribute or content read.  Every external array is captured exactly once
with ``tobytes`` and all later validation and arithmetic use the resulting
immutable view.

Builtin, NumPy, math, clock, PID, lock, and weak-reference dependencies are
captured at module import, identity-gated before factory construction, and
used through closure-private bindings.  Factory-private types are constructed
through the literal-derived exact builtin metaclass; the sealed implementation
has neither ``LOAD_BUILD_CLASS`` nor a dependency on
``builtins.__build_class__``.  Hot arithmetic avoids NumPy's mutable Python
dispatch wrappers in favour of captured ufuncs, exact ndarray methods, and
builtin C entry points.

Every output is an exact builtin tuple containing only exact bytes, exact
shape/dtype tuples, and the literal ``False`` authority bit.  No ndarray or
mutable heap-class result object crosses the return boundary.  The tuple is a
non-authoritative pure value, never a capability and never valid as caller
input to a later trusted operation.  This remains a non-authoritative research
candidate; it cannot issue or authenticate a solver verdict.
"""

from __future__ import annotations

import builtins as _builtins_module
import ctypes as _ctypes_module
import _thread as _thread_module
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
    TypeError as _TypeErrorModule,
    ValueError as _ValueErrorModule,
    any as _any_module,
    id as _id_module,
    len as _len_module,
    max as _max_module,
    memoryview as _memoryview_module,
    min as _min_module,
    property as _property_module,
    range as _range_module,
    sum as _sum_module,
)
from types import MappingProxyType as _MappingProxyTypeModule
from typing import Any, NoReturn

import numpy as _np_module
from numpy._core import _exceptions as _np_exceptions_module
from numpy._core import _ufunc_config as _ufunc_config_module

from act.back_end.hybridz_tf import query_dual_replay as _frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as _conv_v51
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as _dense_v51


SCHEMA = "act.query_dual_replay_v51b_private_numeric_kernel.v1"
NUMERIC_PROTOCOL = "closure_owned_prevalidated_dense_conv_v51b"

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
_I64DtypeModule = _np_module.dtype(_np_module.int64)
_F64InfoCaptureModule = _np_module.finfo(_np_module.float64)
_WideInfoCaptureModule = _np_module.finfo(_np_module.longdouble)
_F64NmantModule = _ExactIntModule(_F64InfoCaptureModule.nmant)
_WideNmantModule = _ExactIntModule(_WideInfoCaptureModule.nmant)
_F64EpsModule = _ExactFloatModule(_F64InfoCaptureModule.eps)
_WideEpsModule = _ExactFloatModule(_WideInfoCaptureModule.eps)
_I32MaxModule = _ExactIntModule(
    _np_module.iinfo(_np_module.int32).max
)
_F64ResultTagModule = _F64DtypeModule.str.encode("ascii")
_BoolResultTagModule = _BoolDtypeModule.str.encode("ascii")
_F64UnitRoundoffModule = _np_module.float64(2.0**-53)
_F64EtaFloatModule = _ExactFloatModule.fromhex(
    "0x0.0000000000001p-1022"
)
_F64EtaModule = _np_module.float64(_F64EtaFloatModule)
_F64HalfAboveModule = _ExactFloatModule.fromhex(
    "0x1.0000000000001p-53"
)
_F64TinyModule = _ExactFloatModule(_F64InfoCaptureModule.tiny)
_I64MaxModule = (1 << 63) - 1
_ArrayMemoryErrorModule = _np_exceptions_module._ArrayMemoryError
_MachineModule = _os_module.uname().machine
_MaxSnapshotBytesModule = 1 << 30
_MaxConvAxisExtentModule = 1 << 20
_MaxConvElementsModule = 1 << 27
_MaxConvWorkspaceBytesModule = 2 << 30
_WideMantissaBitsModule = _ExactIntModule(
    _WideInfoCaptureModule.nmant
) + 1
del _F64InfoCaptureModule
del _WideInfoCaptureModule
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
    _F64HalfAboveModule,
    _I64MaxModule,
    _MaxSnapshotBytesModule,
    _MaxConvAxisExtentModule,
    _MaxConvElementsModule,
    _MaxConvWorkspaceBytesModule,
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
_UfuncTypeModule = _ExactTypeModule(_np_module.add)
_UfuncReduceDescriptorModule = _UfuncTypeModule.__dict__["reduce"]
_UfuncInstanceStatesModule = (
    (
        _np_module.add,
        _ExactTupleModule(
            _ExactDictModule.items(_np_module.add.__dict__)
        ),
    ),
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
    (
        _np_module.maximum,
        _ExactTupleModule(
            _ExactDictModule.items(_np_module.maximum.__dict__)
        ),
    ),
    (
        _np_module.minimum,
        _ExactTupleModule(
            _ExactDictModule.items(_np_module.minimum.__dict__)
        ),
    ),
)
_AddReduceModule = _UfuncReduceDescriptorModule.__get__(
    _np_module.add, _UfuncTypeModule
)
_LogicalAndReduceModule = _UfuncReduceDescriptorModule.__get__(
    _np_module.logical_and, _UfuncTypeModule
)
_LogicalOrReduceModule = _UfuncReduceDescriptorModule.__get__(
    _np_module.logical_or, _UfuncTypeModule
)
_MaximumReduceModule = _UfuncReduceDescriptorModule.__get__(
    _np_module.maximum, _UfuncTypeModule
)
_MinimumReduceModule = _UfuncReduceDescriptorModule.__get__(
    _np_module.minimum, _UfuncTypeModule
)


class PrivateNumericKernelError(RuntimeError):
    """Stable fail-closed error for the isolated private kernel."""

    def __init__(self, code: str, message: str):
        self.code = "{}".format(code)
        self.args = ("{}: {}".format(self.code, message),)


class PrivateNumericKernelTimeout(PrivateNumericKernelError):
    """The factory's one finite absolute deadline expired."""

    def __init__(self) -> None:
        self.code = "DEADLINE_EXPIRED"
        self.args = (
            "DEADLINE_EXPIRED: "
            "V5.1b private numeric-kernel deadline expired",
        )


def _create_private_numeric_kernel_impl(
    *,
    deadline: float,
    _sealed_dependencies: Any,
) -> Any:
    """Return one factory-local, non-authoritative private-kernel port.

    ``deadline`` is one finite absolute :func:`time.monotonic` timestamp.
    Dense and Conv material is admitted from exact native binary64 ndarrays.
    The returned port intentionally exposes no material accessor.
    """

    # Literal-derived exact types do not resolve mutable builtin names.
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
        _direct_dependencies,
    ) = _sealed_dependencies
    (
        _PrivateNumericKernelError,
        _PrivateNumericKernelTimeout,
        _ArrayMemoryError,
        _monotonic,
        _getpid,
        _RLock,
        _weakref_ref,
        _MappingProxyType,
        _prepare_dense_public,
        _prepare_conv_public,
        _FrozenLayer,
        _Box,
        _Deadline,
        _ReplayTimeout,
        _DenseAdmissionError,
        _ReplayError,
        _DenseSupport,
        _ConvPlan,
        _ConvOffset,
        _schema,
        _F64,
        _BOOL,
        _I64,
        _F64_NMANT,
        _WIDE_NMANT,
        _F64_EPS,
        _WIDE_EPS,
        _I32_MAX,
        _F64_RESULT_TAG,
        _BOOL_RESULT_TAG,
        _F64_U,
        _F64_ETA,
        _F64_TINY,
        _WIDE_MANTISSA_BITS,
        _gate_primitives,
        _UfuncType,
        _ufunc_reduce_descriptor,
        _ufunc_instance_states,
        _add_reduce,
        _logical_and_reduce,
        _logical_or_reduce,
        _maximum_reduce,
        _minimum_reduce,
        _prepared_mode,
    ) = _direct_dependencies
    (
        _object_getattribute,
        _ModuleType,
        _module_dict_get,
        _canonical_extobj,
        _extobj_set,
        _extobj_reset,
        _F64_ETA_FLOAT,
        _F64_HALF_ABOVE,
        _I64_MAX,
        _MAX_SNAPSHOT_BYTES,
        _MAX_CONV_AXIS_EXTENT,
        _MAX_CONV_ELEMENTS,
        _MAX_CONV_WORKSPACE_BYTES,
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

    # Reject substitutions before invoking any dependency.  All later numeric
    # calls resolve through closure-private mappings built from import-time
    # function objects, never through a mutable module attribute.
    _dependency_sentinel = _ExactObject()
    for _module, _bindings in _module_gates:
        # ``module.__dict__`` is itself a dynamic attribute lookup when a
        # caller has changed the module object's heap subclass.  Exact-type
        # rejection must therefore precede that lookup, and the dictionary is
        # read only through the import-time bound native descriptor.
        if _ExactType(_module) is not _ModuleType:
            raise _PrivateNumericKernelError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted dependency module changed exact type",
            )
        _module_dict = _module_dict_get(_module, _ModuleType)
        if (
            _ExactType(_module) is not _ModuleType
            or _ExactType(_module_dict) is not _ExactDict
        ):
            raise _PrivateNumericKernelError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted dependency module or dictionary changed type",
            )
        for _name, _trusted in _bindings:
            if (
                _ExactDict.get(
                    _module_dict, _name, _dependency_sentinel
                )
                is not _trusted
            ):
                raise _PrivateNumericKernelError(
                    "DEPENDENCY_SUBSTITUTION",
                    "trusted dependency binding was substituted",
                )

    # A NumPy ufunc is an exact builtin instance, but its per-instance
    # ``__dict__`` is mutable.  In particular, assigning ``ufunc.reduce``
    # leaves the module binding unchanged.  Fingerprint that instance state
    # before construction and execute only import-time descriptor-bound
    # reduce methods, so post-construction instance changes are inert.
    _ufunc_type_dict = _object_getattribute(_UfuncType, "__dict__")
    if (
        _ExactType(_ufunc_type_dict) is not _MappingProxyType
        or _ufunc_type_dict["reduce"] is not _ufunc_reduce_descriptor
    ):
        raise _PrivateNumericKernelError(
            "DEPENDENCY_SUBSTITUTION",
            "trusted NumPy ufunc reduce descriptor was substituted",
        )
    for _ufunc, _trusted_state in _ufunc_instance_states:
        if _ExactType(_ufunc) is not _UfuncType:
            raise _PrivateNumericKernelError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted NumPy ufunc changed exact type",
            )
        _current_state = _object_getattribute(_ufunc, "__dict__")
        if (
            _ExactType(_current_state) is not _ExactDict
            or _ExactDict.__len__(_current_state)
            != _ExactTuple.__len__(_trusted_state)
        ):
            raise _PrivateNumericKernelError(
                "DEPENDENCY_SUBSTITUTION",
                "trusted NumPy ufunc instance state changed",
            )
        for _state_name, _trusted_value in _trusted_state:
            if (
                _ExactDict.get(
                    _current_state, _state_name, _dependency_sentinel
                )
                is not _trusted_value
            ):
                raise _PrivateNumericKernelError(
                    "DEPENDENCY_SUBSTITUTION",
                    "trusted NumPy ufunc instance state changed",
                )

    _builtins = _MappingProxyType(_ExactDict(_trusted_builtins))
    _np = _MappingProxyType(_ExactDict(_trusted_numpy))
    _math = _MappingProxyType(_ExactDict(_trusted_math))
    _any = _builtins["any"]
    _len = _builtins["len"]
    _max = _builtins["max"]
    _memoryview = _builtins["memoryview"]
    _min = _builtins["min"]
    _property = _builtins["property"]
    _range = _builtins["range"]
    _sum = _builtins["sum"]
    _id = _builtins["id"]
    _Exception = _builtins["Exception"]
    _FloatingPointError = _builtins["FloatingPointError"]
    _MemoryError = _builtins["MemoryError"]
    _TypeError = _builtins["TypeError"]
    _ValueError = _builtins["ValueError"]
    _OverflowError = _builtins["OverflowError"]
    _numpy_version = _np["__version__"]

    _ndarray_fill = _np["ndarray"].fill
    _ndarray_nonzero = _np["ndarray"].nonzero
    _ndarray_take = _np["ndarray"].take

    def _np_all(value: Any, axis: Any = None) -> Any:
        return _logical_and_reduce(
            _np["asarray"](value), axis=axis, initial=True
        )

    def _np_any(value: Any, axis: Any = None) -> Any:
        return _logical_or_reduce(
            _np["asarray"](value), axis=axis, initial=False
        )

    def _np_array_equal(left: Any, right: Any) -> Any:
        if left.shape != right.shape:
            return False
        return _logical_and_reduce(
            _np["asarray"](left == right),
            axis=None,
            initial=True,
        )

    def _np_count_nonzero(value: Any) -> Any:
        return _add_reduce(
            _np["asarray"](value != 0).reshape(-1),
            axis=None,
            initial=0,
        )

    def _np_flatnonzero(value: Any) -> Any:
        flattened = _np["asarray"](value).reshape(-1)
        return _ndarray_nonzero(flattened)[0]

    def _np_full(shape: Any, fill_value: Any, dtype: Any) -> Any:
        result = _np["zeros"](shape, dtype=dtype)
        _ndarray_fill(result, fill_value)
        return result

    def _np_max(value: Any, axis: Any = None) -> Any:
        return _maximum_reduce(_np["asarray"](value), axis=axis)

    def _np_min(value: Any, axis: Any = None) -> Any:
        return _minimum_reduce(_np["asarray"](value), axis=axis)

    def _np_ones(shape: Any, dtype: Any) -> Any:
        result = _np["zeros"](shape, dtype=dtype)
        _ndarray_fill(result, 1)
        return result

    def _np_take(value: Any, indices: Any, axis: Any) -> Any:
        return _ndarray_take(value, indices, axis=axis)

    _DENSE_RESULT_TAG = b"act.v51b.private.dense-result.v1"
    _CONV_RESULT_TAG = b"act.v51b.private.conv-result.v1"
    if (
        _ExactType(_F64_RESULT_TAG) is not _ExactBytes
        or _ExactType(_BOOL_RESULT_TAG) is not _ExactBytes
        or _ExactType(_prepared_mode) is not _ExactBool
        or _ExactType(_numpy_version) is not _ExactStr
        or _numpy_version != "2.3.5"
        or _ExactType(_F64_ETA_FLOAT) is not _ExactFloat
        or _ExactType(_F64_HALF_ABOVE) is not _ExactFloat
        or _ExactType(_I64_MAX) is not _ExactInt
        or _ExactType(_MAX_SNAPSHOT_BYTES) is not _ExactInt
        or _ExactType(_MAX_CONV_AXIS_EXTENT) is not _ExactInt
        or _ExactType(_MAX_CONV_ELEMENTS) is not _ExactInt
        or _ExactType(_MAX_CONV_WORKSPACE_BYTES) is not _ExactInt
        or _I64_MAX != 9223372036854775807
        or _MAX_SNAPSHOT_BYTES != 1073741824
        or _MAX_CONV_AXIS_EXTENT != 1048576
        or _MAX_CONV_ELEMENTS != 134217728
        or _MAX_CONV_WORKSPACE_BYTES != 2147483648
        or _ExactType(_machine) is not _ExactStr
        or _machine != "x86_64"
    ):
        raise _PrivateNumericKernelError(
            "DEPENDENCY_SUBSTITUTION",
            "dtype tags did not encode to exact builtin bytes",
        )
    _F64_NORMAL_FREXP_EXPONENT = -1021
    _REQUIRED_EXTRA_MANTISSA_BITS = 8

    if (
        _ExactType(deadline) is not _ExactFloat
        or not _math["isfinite"](deadline)
    ):
        raise _PrivateNumericKernelError(
            "INVALID_DEADLINE",
            "deadline must be an exact finite float monotonic timestamp",
        )
    owner_pid = _getpid()
    end = deadline

    def _raise(code: str, message: str) -> NoReturn:
        raise _PrivateNumericKernelError(code, message)

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
        """Inspect a call-local snapshot without normalising control state."""

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
            raise _PrivateNumericKernelError(
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
                    # Returning to Python with a pending, unmasked x87
                    # exception can signal before the caller can handle the
                    # stable rejection.  Preserve all control bits, but clear
                    # only unmasked pending flags (plus the x87 summary/busy
                    # flags) in this already-invalid environment.
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
            raise _PrivateNumericKernelError(
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
        """Reject mutable Python callbacks around numeric operations."""

        try:
            trace_callback = _gettrace()
            profile_callback = _getprofile()
            monitoring_active = False
            for tool_id in _range(6):
                if _monitoring_get_tool(tool_id) is not None:
                    monitoring_active = True
        except _Exception as exc:
            raise _PrivateNumericKernelError(
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
            raise _PrivateNumericKernelError(
                "NUMERIC_ENVIRONMENT",
                "failed to install the canonical NumPy numeric policy",
            ) from exc

    def _leave_numeric_environment(token: Any) -> None:
        try:
            _extobj_reset(token)
        except _Exception as exc:
            raise _PrivateNumericKernelError(
                "NUMERIC_ENVIRONMENT",
                "failed to restore the caller NumPy numeric policy",
            ) from exc

    def _check_runtime_numeric_environment() -> None:
        """Check the actual caller thread's binary64 control state by bits."""

        _check_fenv_control()
        one = _ExactFloat(1.0)
        half = _ExactFloat(_F64_EPS) * _ExactFloat(0.5)
        eta_product = _F64_ETA_FLOAT * one
        if (
            _ExactFloat.hex(one + half)
            != "0x1.0000000000000p+0"
            or _ExactFloat.hex(one + _F64_HALF_ABOVE)
            != "0x1.0000000000001p+0"
            or _ExactFloat.hex(eta_product)
            != "0x0.0000000000001p-1022"
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "caller thread is not binary64 RN with gradual underflow",
            )

    def _check_platform() -> None:
        _check_fenv_control()
        if (
            _F64.itemsize != 8
            or _ExactType(_F64_NMANT) is not _ExactInt
            or _ExactType(_WIDE_NMANT) is not _ExactInt
            or _ExactType(_F64_EPS) is not _ExactFloat
            or _ExactType(_WIDE_EPS) is not _ExactFloat
            or _WIDE_NMANT
            < _F64_NMANT + _REQUIRED_EXTRA_MANTISSA_BITS
            or not _WIDE_EPS < _F64_EPS
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "longdouble must have at least eight more mantissa bits",
            )
        eta_l = _np["nextafter"](
            _np["longdouble"](0.0), _np["longdouble"](_math["inf"])
        )
        if (
            eta_l <= _np["longdouble"](0.0)
            or _np["longdouble"](eta_l * _np["longdouble"](1.0)) != eta_l
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "longdouble gradual-underflow probe failed",
            )
        half_ulp = _np["longdouble"](2.0) ** _np["longdouble"](
            -(_WIDE_NMANT + 1)
        )
        above = _np["nextafter"](half_ulp, _np["longdouble"](_math["inf"]))
        if (
            _np["longdouble"](1.0) + half_ulp != _np["longdouble"](1.0)
            or _np["longdouble"](1.0) + above == _np["longdouble"](1.0)
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "longdouble round-to-nearest-even probe failed",
            )
        eta_scalar = _ExactFloat(
            _np["float64"](_F64_ETA * _np["float64"](1.0))
        )
        if (
            _ExactFloat.hex(eta_scalar)
            != "0x0.0000000000001p-1022"
        ):
            _raise(
                "NUMERIC_PLATFORM",
                "binary64 gradual-underflow probe failed",
            )
        # Backend identity, fixed-four configuration, worker prewarming, and
        # matrix-lane bit probes belong to the process root's held BLAS
        # runtime lease.  Running a parallel DGEMM from this public,
        # non-authoritative factory can make libgomp terminate the process
        # instead of raising when address-space reserve is low.
        _check_runtime_numeric_environment()

    _check_instrumentation()
    _factory_numeric_token = _enter_numeric_environment()
    try:
        _check_platform()
        _check_instrumentation()
        if (
            _ExactFloat.hex(_F64_ETA_FLOAT)
            != "0x0.0000000000001p-1022"
            or _ExactFloat.hex(_F64_HALF_ABOVE)
            != "0x1.0000000000001p-53"
        ):
            _raise(
                "DEPENDENCY_SUBSTITUTION",
                "fixed binary64 probe constants changed representation",
            )
    except _FloatingPointError as exc:
        if _ExactType(exc) is not _FloatingPointError:
            raise
        raise _PrivateNumericKernelError(
            "NUMERIC_PLATFORM",
            "canonical numeric-platform probe raised a floating exception",
        ) from exc
    except _MemoryError as exc:
        if (
            _ExactType(exc) is not _MemoryError
            and _ExactType(exc) is not _ArrayMemoryError
        ):
            raise
        raise _PrivateNumericKernelError(
            "RESOURCE_LIMIT",
            "numeric-platform probe exceeded the fixed memory budget",
        ) from exc
    finally:
        _leave_numeric_environment(_factory_numeric_token)

    state = ["OPEN"]
    lifecycle_lock = _RLock()
    cores = {}
    locator_tokens = {}
    locator_refs = {}
    counters = {
        "dense_admissions": 0,
        "conv_admissions": 0,
        "dense_executions": 0,
        "conv_executions": 0,
    }

    def _drop_all(new_state: str) -> None:
        cores.clear()
        locator_tokens.clear()
        locator_refs.clear()
        state[0] = new_state

    def _expire() -> NoReturn:
        with lifecycle_lock:
            _drop_all("EXPIRED")
        raise _PrivateNumericKernelTimeout()

    def _check_live() -> None:
        if _getpid() != owner_pid:
            _raise(
                "FORKED_PROCESS",
                "a private-kernel capability cannot cross a fork",
            )
        with lifecycle_lock:
            if state[0] == "EXPIRED":
                raise _PrivateNumericKernelTimeout()
            if state[0] != "OPEN":
                _raise("CLOSED", "private numeric kernel is closed")
            # Reading the hardware control words uses no floating-point
            # arithmetic and must precede the monotonic deadline comparison:
            # an unmasked inexact exception can otherwise terminate Python.
            _check_fenv_control()
            if _monotonic() >= end:
                _expire()

    def _exact_array_metadata(
        value: Any,
        *,
        dtype: Any,
        ndim: int,
        name: str,
    ) -> tuple[Any, int]:
        # This exact-type branch must precede *every* ndarray attribute,
        # conversion protocol, getter, or content read.
        if _ExactType(value) is not _np["ndarray"]:
            _raise(
                "INVALID_ARRAY_TYPE",
                f"{name} must be an exact numpy.ndarray",
            )
        value_dtype = value.dtype
        value_shape = value.shape
        if (
            value_dtype != dtype
            or (dtype == _F64 and not value_dtype.isnative)
            or _ExactType(value_shape) is not _ExactTuple
            or _any(
                _ExactType(extent) is not _ExactInt
                for extent in value_shape
            )
            or value.ndim != ndim
            or not value.flags.c_contiguous
        ):
            _raise(
                "INVALID_ARRAY",
                f"{name} has invalid dtype, rank, or layout",
            )
        count = 1
        for extent in value_shape:
            count *= _ExactInt(extent)
        byte_count = count * value_dtype.itemsize
        if byte_count > _MAX_SNAPSHOT_BYTES:
            _raise(
                "RESOURCE_LIMIT",
                f"{name} exceeds the fixed snapshot byte budget",
            )
        return value_shape, byte_count

    def _snapshot_exact_array(
        value: Any,
        *,
        dtype: Any,
        ndim: int,
        name: str,
        finite: bool = True,
        expected_shape: Any = None,
    ) -> Any:
        value_shape, byte_count = _exact_array_metadata(
            value,
            dtype=dtype,
            ndim=ndim,
            name=name,
        )
        if (
            expected_shape is not None
            and value_shape != expected_shape
        ):
            _raise(
                "INVALID_ARRAY",
                f"{name} metadata changed before snapshot",
            )
        _check_live()
        # Exporting the exact ndarray's C buffer prevents concurrent resizing.
        # Its captured nbytes is checked before the sole proportional copy, so
        # a resize racing the earlier metadata pass cannot bypass the budget.
        try:
            exported = _memoryview(value)
        except (_TypeError, _ValueError, _OverflowError) as exc:
            raise _PrivateNumericKernelError(
                "INVALID_ARRAY",
                f"{name} could not export a stable buffer",
            ) from exc
        if (
            _ExactType(exported) is not _memoryview
            or not exported.c_contiguous
            or exported.nbytes != byte_count
        ):
            _raise(
                "INVALID_ARRAY",
                f"{name} changed before its stable buffer export",
            )
        # Everything after this point uses the root-owned bytes object,
        # including all finiteness checks.
        payload = _ExactBytes(exported)
        _check_live()
        post_shape, post_byte_count = _exact_array_metadata(
            value,
            dtype=dtype,
            ndim=ndim,
            name=name,
        )
        if _ExactType(payload) is not _ExactBytes:
            _raise(
                "INVALID_STORAGE",
                f"{name} snapshot is not exact builtin bytes",
            )
        if (
            _len(payload) != byte_count
            or post_shape != value_shape
            or post_byte_count != byte_count
        ):
            _raise("INVALID_ARRAY", f"{name} changed during snapshot")
        try:
            result = _np["frombuffer"](payload, dtype=dtype).reshape(value_shape)
        except (_TypeError, _ValueError, _OverflowError) as exc:
            raise _PrivateNumericKernelError(
                "INVALID_ARRAY",
                f"{name} could not be reconstructed",
            ) from exc
        if result.flags.writeable or result.flags.owndata:
            _raise(
                "INVALID_STORAGE",
                f"{name} snapshot is not immutable bytes-backed storage",
            )
        if finite and not _np_all(_np["isfinite"](result)):
            _raise("NONFINITE", f"{name} contains a non-finite value")
        return result

    def _immutable(value: Any, dtype: Any) -> Any:
        contiguous = _np["ascontiguousarray"](value, dtype=dtype)
        result = _np["frombuffer"](
            contiguous.tobytes(order="C"), dtype=dtype
        ).reshape(contiguous.shape)
        if result.flags.writeable or result.flags.owndata:
            _raise(
                "INVALID_STORAGE",
                "failed to create immutable bytes-backed output",
            )
        return result

    def _result_frame(value: Any, dtype: Any) -> tuple[Any, ...]:
        """Freeze one output as an exact bytes/shape/dtype pure value."""

        if dtype != _F64 and dtype != _BOOL:
            _raise(
                "INVALID_STORAGE",
                "result frame dtype is outside the fixed private ABI",
            )
        contiguous = _np["ascontiguousarray"](value, dtype=dtype)
        shape = contiguous.shape
        dtype_tag = (
            _F64_RESULT_TAG if dtype == _F64 else _BOOL_RESULT_TAG
        )
        count = 1
        for extent in shape:
            count *= extent
        byte_count = count * dtype.itemsize
        if byte_count > _MAX_SNAPSHOT_BYTES:
            _raise(
                "RESOURCE_LIMIT",
                "result frame exceeds the fixed snapshot byte budget",
            )
        payload = contiguous.tobytes(order="C")
        if (
            _ExactType(payload) is not _ExactBytes
            or _ExactType(shape) is not _ExactTuple
            or _any(_ExactType(extent) is not _ExactInt for extent in shape)
            or _any(extent <= 0 for extent in shape)
            or _ExactType(dtype_tag) is not _ExactBytes
            or _len(payload) != byte_count
            or (
                dtype == _BOOL
                and _any(byte not in (0, 1) for byte in payload)
            )
        ):
            _raise(
                "INVALID_STORAGE",
                "failed to create exact immutable result frame",
            )
        return (payload, shape, dtype_tag)

    def _same_array_bits(left: Any, right: Any) -> bool:
        if (
            _ExactType(left) is not _np["ndarray"]
            or _ExactType(right) is not _np["ndarray"]
            or left.dtype != right.dtype
            or left.shape != right.shape
        ):
            return False
        if left.dtype == _F64:
            return _ExactBool(
                _np_array_equal(
                    left.view(_np["uint64"]),
                    right.view(_np["uint64"]),
                )
            )
        return _ExactBool(_np_array_equal(left, right))

    def _exact_tuple(
        value: Any,
        *,
        length: int,
        name: str,
        positive: bool,
    ) -> tuple[int, ...]:
        if _ExactType(value) is not _ExactTuple or _len(value) != length:
            _raise(
                "INVALID_GEOMETRY",
                f"{name} must be an exact length-{length} tuple",
            )
        if _any(
            _ExactType(item) is not _ExactInt
            or (item <= 0 if positive else item < 0)
            for item in value
        ):
            qualifier = "positive" if positive else "nonnegative"
            _raise(
                "INVALID_GEOMETRY",
                f"{name} entries must be exact {qualifier} integers",
            )
        return value

    def _finite(value: Any, where: str) -> None:
        if not _np_all(_np["isfinite"](value)):
            _raise("NONFINITE", f"non-finite arithmetic at {where}")

    # Directed helpers copied from the frozen V5.1a numeric bodies.
    def _ld_up(value: Any) -> Any:
        array = _np["asarray"](value, dtype=_np["longdouble"])
        if not _np_all(_np["isfinite"](array)) or _np_any(array < 0):
            _raise(
                "NONFINITE",
                "invalid nonnegative longdouble expression",
            )
        result = _np["nextafter"](
            array,
            _np["longdouble"](_math["inf"]),
            dtype=_np["longdouble"],
        )
        if not _np_all(_np["isfinite"](result)):
            _raise("NONFINITE", "longdouble outward successor overflowed")
        return result

    def _ld_down_positive(value: Any) -> Any:
        array = _np["asarray"](value, dtype=_np["longdouble"])
        if not _np_all(_np["isfinite"](array)) or _np_any(array <= 0):
            _raise(
                "NUMERIC_GUARD",
                "invalid positive longdouble denominator",
            )
        result = _np["nextafter"](
            array,
            _np["longdouble"](0.0),
            dtype=_np["longdouble"],
        )
        if _np_any(result <= 0):
            _raise(
                "NUMERIC_GUARD",
                "longdouble denominator rounded to zero",
            )
        return result

    def _ld_add_up(left: Any, right: Any) -> Any:
        return _ld_up(
            _np["asarray"](left, dtype=_np["longdouble"])
            + _np["asarray"](right, dtype=_np["longdouble"])
        )

    def _ld_mul_up(left: Any, right: Any) -> Any:
        return _ld_up(
            _np["asarray"](left, dtype=_np["longdouble"])
            * _np["asarray"](right, dtype=_np["longdouble"])
        )

    def _ld_div_up(numerator: Any, denominator_lower: Any) -> Any:
        denominator = _np["asarray"](
            denominator_lower, dtype=_np["longdouble"]
        )
        if (
            _np_any(denominator <= 0)
            or not _np_all(_np["isfinite"](denominator))
        ):
            _raise(
                "NUMERIC_GUARD",
                "invalid longdouble division denominator",
            )
        return _ld_up(
            _np["asarray"](numerator, dtype=_np["longdouble"]) / denominator
        )

    def _ceil_f64(value: Any) -> Any:
        wide = _np["asarray"](value, dtype=_np["longdouble"])
        scalar = wide.ndim == 0
        if not _np_all(_np["isfinite"](wide)) or _np_any(wide < 0):
            _raise(
                "NONFINITE",
                "invalid longdouble value for binary64 ceiling",
            )
        nearest = _np["asarray"](wide, dtype=_np["float64"])
        if not _np_all(_np["isfinite"](nearest)):
            _raise("NONFINITE", "wide enclosure does not fit in binary64")
        below = _np["asarray"](nearest, dtype=_np["longdouble"]) < wide
        if _np_any(below):
            if scalar:
                nearest = _np["asarray"](
                    _np["nextafter"](
                        _np["float64"](nearest.item()),
                        _np["float64"](_math["inf"]),
                    ),
                    dtype=_np["float64"],
                )
            else:
                nearest = _np["ascontiguousarray"](nearest)
                nearest[below] = _np["nextafter"](
                    nearest[below], _np["float64"](_math["inf"])
                )
        if not _np_all(_np["isfinite"](nearest)):
            _raise("NONFINITE", "binary64 outward successor overflowed")
        if scalar:
            return _np["asarray"](nearest, dtype=_np["float64"]).reshape(())
        return _np["ascontiguousarray"](nearest, dtype=_np["float64"])

    def _wide_parameters(width: int) -> tuple[Any, Any]:
        if width <= 0:
            _raise("SHAPE_MISMATCH", "dot width must be positive")
        operations = 2 * _ExactInt(width) + 2
        mantissa_bits = _WIDE_MANTISSA_BITS
        unit_roundoff = _np["ldexp"](
            _np["longdouble"](1.0), -mantissa_bits
        )
        eta = _np["nextafter"](
            _np["longdouble"](0.0),
            _np["longdouble"](_math["inf"]),
            dtype=_np["longdouble"],
        )
        product = _ld_mul_up(
            _np["longdouble"](operations), unit_roundoff
        )
        if product.ndim != 0 or product >= _np["longdouble"](0.5):
            _raise(
                "NUMERIC_GUARD",
                "longdouble gamma is too large",
            )
        denominator = _ld_down_positive(
            _np["longdouble"](1.0) - product
        )
        gamma = _ld_div_up(product, denominator)
        tau = _ld_div_up(
            _ld_mul_up(_np["longdouble"](operations), eta),
            denominator,
        )
        return _np["longdouble"](gamma), _np["longdouble"](tau)

    def _dot_up_rows(left: Any, right: Any) -> Any:
        if (
            left.dtype != _F64
            or right.dtype != _F64
            or left.ndim != 2
            or right.ndim != 1
            or left.shape[1] != right.size
            or left.shape[0] <= 0
            or right.size <= 0
            or _np_any(left < 0.0)
            or _np_any(right < 0.0)
            or not _np_all(_np["isfinite"](left))
            or not _np_all(_np["isfinite"](right))
        ):
            _raise("INVALID_MASS", "invalid nonnegative DotUpL operands")
        exact_zero = ~_np_any(
            (left != 0.0)
            & (right.reshape(1, -1) != 0.0),
            axis=1,
        )
        nominal = _np["asarray"](
            _np["asarray"](left, dtype=_np["longdouble"])
            @ _np["asarray"](right, dtype=_np["longdouble"]),
            dtype=_np["longdouble"],
        )
        gamma, tau = _wide_parameters(right.size)
        numerator = _ld_add_up(nominal, tau)
        denominator = _ld_down_positive(
            _np["longdouble"](1.0) - gamma
        )
        upper = _ceil_f64(_ld_div_up(numerator, denominator))
        upper[exact_zero] = 0.0
        return _np["ascontiguousarray"](upper, dtype=_np["float64"])

    def _dot_up_matrix(left: Any, right: Any) -> tuple[Any, Any]:
        if (
            left.dtype != _F64
            or right.dtype != _F64
            or left.ndim != 2
            or right.ndim != 2
            or left.shape[1] != right.shape[0]
            or left.shape[0] <= 0
            or right.shape[1] <= 0
            or _np_any(left < 0.0)
            or _np_any(right < 0.0)
            or not _np_all(_np["isfinite"](left))
            or not _np_all(_np["isfinite"](right))
        ):
            _raise(
                "INVALID_MASS",
                "invalid nonnegative DotUpL matrix operands",
            )
        activity = _np["asarray"](
            (left != 0.0) @ (right != 0.0),
            dtype=_np["bool_"],
        )
        nominal = _np["asarray"](
            _np["asarray"](left, dtype=_np["longdouble"])
            @ _np["asarray"](right, dtype=_np["longdouble"]),
            dtype=_np["longdouble"],
        )
        gamma, tau = _wide_parameters(left.shape[1])
        numerator = _ld_add_up(nominal, tau)
        denominator = _ld_down_positive(
            _np["longdouble"](1.0) - gamma
        )
        result = _ceil_f64(
            _ld_div_up(numerator, denominator)
        )
        result[~activity] = 0.0
        return (
            _np["ascontiguousarray"](result, dtype=_np["float64"]),
            _np["ascontiguousarray"](activity, dtype=_np["bool_"]),
        )

    def _zero_sum(left: Any, right: Any) -> Any:
        if (
            left.shape != right.shape
            or _np_any(left < 0.0)
            or _np_any(right < 0.0)
            or not _np_all(_np["isfinite"](left))
            or not _np_all(_np["isfinite"](right))
        ):
            _raise(
                "NUMERIC_GUARD",
                "invalid zero-preserving sum operands",
            )
        left_active = left != 0.0
        right_active = right != 0.0
        both = left_active & right_active
        result = _np["zeros"](left.shape, dtype=_np["float64"])
        only_left = left_active & ~right_active
        only_right = right_active & ~left_active
        result[only_left] = left[only_left]
        result[only_right] = right[only_right]
        if _np_any(both):
            result[both] = _ceil_f64(
                _ld_add_up(left[both], right[both])
            )
        return result

    def _legacy_up(value: Any) -> Any:
        """Frozen V3 unconditional longdouble-to-f64 successor."""

        wide = _np["asarray"](value, dtype=_np["longdouble"])
        if not _np_all(_np["isfinite"](wide)):
            _raise(
                "NONFINITE",
                "legacy outward expression is non-finite",
            )
        result = _np["asarray"](wide, dtype=_np["float64"])
        if not _np_all(_np["isfinite"](result)):
            _raise("NONFINITE", "legacy outward conversion overflowed")
        result = _np["nextafter"](result, _math["inf"])
        if result.ndim == 0:
            return _ExactFloat(result)
        return _np["ascontiguousarray"](result)

    def _legacy_f64_parameters(operations: int) -> tuple[float, float]:
        product = _np["longdouble"](operations) * _np["longdouble"](_F64_U)
        if product >= _np["longdouble"](0.5):
            _raise("NUMERIC_GUARD", "operation count is too large")
        denominator = _np["longdouble"](1.0) - product
        gamma = _legacy_up(product / denominator)
        tau = _legacy_up(
            _np["longdouble"](operations)
            * _np["longdouble"](_F64_ETA)
            / denominator
        )
        return _ExactFloat(gamma), _ExactFloat(tau)

    def _conv_f64_parameters(operations: int) -> tuple[float, float]:
        if operations <= 0:
            _raise(
                "NUMERIC_GUARD",
                "binary64 operation count must be positive",
            )
        product = _ld_mul_up(
            _np["longdouble"](operations), _np["longdouble"](_F64_U)
        )
        if product >= _np["longdouble"](0.5):
            _raise(
                "NUMERIC_GUARD",
                "binary64 operation count is too large",
            )
        denominator = _ld_down_positive(
            _np["longdouble"](1.0) - product
        )
        gamma = _ceil_f64(_ld_div_up(product, denominator))
        tau = _ceil_f64(
            _ld_div_up(
                _ld_mul_up(
                    _np["longdouble"](operations),
                    _np["longdouble"](_F64_ETA),
                ),
                denominator,
            )
        )
        return _ExactFloat(gamma), _ExactFloat(tau)

    def _matrix_product_radius(left: Any, right: Any) -> Any:
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[1] != right.shape[0]
        ):
            _raise(
                "SHAPE_MISMATCH",
                "invalid binary64 matrix-product operands",
            )
        operations = 2 * _ExactInt(left.shape[1]) + 2
        product = _np["longdouble"](operations) * _np["longdouble"](_F64_U)
        if product >= _np["longdouble"](0.5):
            _raise(
                "NUMERIC_GUARD",
                "Higham gamma operation count is too large",
            )
        gamma = _ExactFloat(
            _legacy_up(
                product / (_np["longdouble"](1.0) - product)
            )
        )
        under = _ExactFloat(
            _legacy_up(
                _np["longdouble"](_max(1, operations))
                * _np["longdouble"](_F64_ETA)
            )
        )
        nominal = _np["asarray"](left @ right, dtype=_np["float64"])
        abs_nominal = _np["asarray"](
            _np["abs"](left) @ _np["abs"](right), dtype=_np["float64"]
        )
        _finite(nominal, "matrix product")
        _finite(abs_nominal, "absolute matrix product")
        sum_upper = _legacy_up(
            (
                _np["asarray"](abs_nominal, dtype=_np["longdouble"])
                + _np["longdouble"](under)
            )
            / (_np["longdouble"](1.0) - _np["longdouble"](gamma))
        )
        radius = _legacy_up(
            _np["longdouble"](gamma)
            * _np["asarray"](sum_upper, dtype=_np["longdouble"])
            + _np["longdouble"](under)
        )
        exact_zero = ~(
            _np_any(left != 0.0, axis=1).reshape(-1, 1)
            & _np_any(right != 0.0, axis=0).reshape(1, -1)
        )
        radius[exact_zero] = 0.0
        _finite(radius, "matrix-product error")
        return _np["ascontiguousarray"](radius)

    def _product_may_be_subnormal(
        first_min: int | None, second_min: int | None
    ) -> bool:
        if first_min is None or second_min is None:
            return False
        return (
            first_min + second_min
            <= _F64_NORMAL_FREXP_EXPONENT
        )

    def _exponent_extrema(
        value: Any,
    ) -> tuple[int | None, int | None]:
        absolute = _np["abs"](_np["asarray"](value, dtype=_np["float64"]))
        nonzero = absolute[absolute != 0.0]
        if nonzero.size == 0:
            return None, None
        _, exponents = _np["frexp"](nonzero)
        return _ExactInt(_np_min(exponents)), _ExactInt(_np_max(exponents))

    def _has_subnormal(value: Any) -> bool:
        absolute = _np["abs"](_np["asarray"](value, dtype=_np["float64"]))
        return _ExactBool(
            _np_any(
                (absolute > 0.0) & (absolute < _F64_TINY)
            )
        )

    # Numeric cores are exact tuples, not mutable heap-class instances.  Hot
    # execution therefore has no class descriptor dispatch for material
    # fields, even under reflective inspection of the hidden closure.
    _DENSE_CORE_TAG = b"act.v51b.private.dense-core.v1"
    _CONV_CORE_TAG = b"act.v51b.private.conv-core.v1"
    _OFFSET_CORE_TAG = b"act.v51b.private.conv-offset-core.v1"

    locator_capability = _ExactObject()

    def _make_locator_type() -> Any:
        __slots__ = ("__weakref__",)

        def __new__(cls, capability: Any = None) -> Any:
            if capability is not locator_capability:
                _raise(
                    "LOCATOR_CONSTRUCTION",
                    "locators are factory-minted",
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
            return "<private-numeric-locator>"

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
        reference: Any, locator_id: int, token: str
    ) -> None:
        # A callback running in a forked child must not touch an inherited
        # lock that another parent thread may have held at fork time.
        if _getpid() != owner_pid:
            return
        with lifecycle_lock:
            if locator_refs.get(locator_id) is reference:
                locator_refs.pop(locator_id, None)
                locator_tokens.pop(locator_id, None)
                cores.pop(token, None)

    def _mint_locator(core: Any) -> Any:
        _check_live()
        token = _id(core)
        with lifecycle_lock:
            _check_live()
            if token in cores:
                _raise(
                    "CORE_COLLISION",
                    "private core identity unexpectedly collided",
                )
            locator = _Locator(locator_capability)
            locator_id = _id(locator)
            if (
                locator_id in locator_tokens
                or locator_id in locator_refs
            ):
                _raise(
                    "LOCATOR_COLLISION",
                    "locator identity unexpectedly collided",
                )
            cores[token] = core
            locator_tokens[locator_id] = token
            reference = _weakref_ref(
                locator,
                lambda current, wanted_id=locator_id, wanted=token: (
                    _drop_locator(current, wanted_id, wanted)
                ),
            )
            locator_refs[locator_id] = reference
            return locator

    def _resolve(locator: Any, wanted: Any) -> Any:
        _check_live()
        if _ExactType(locator) is not _Locator:
            _raise(
                "LOCATOR_MISMATCH",
                "locator was copied or transplanted",
            )
        with lifecycle_lock:
            _check_live()
            locator_id = _id(locator)
            reference = locator_refs.get(locator_id)
            token = locator_tokens.get(locator_id)
            core = cores.get(token) if token is not None else None
            if (
                reference is None
                or reference() is not locator
                or _ExactType(core) is not _ExactTuple
                or _len(core) < 2
                or core[0] != wanted
            ):
                _raise(
                    "LOCATOR_MISMATCH",
                    "locator does not bind the requested private core",
                )
            return core

    def _dense_runtime(core: Any, coefficients: Any) -> Any:
        if (
            _ExactType(core) is not _ExactTuple
            or _len(core) != 11
            or core[0] != _DENSE_CORE_TAG
        ):
            _raise("CORE_MISMATCH", "invalid private Dense core tuple")
        (
            _,
            core_weight,
            core_max_abs,
            core_support,
            core_box_mass,
            core_weight_exponent_min,
            core_support_exponent_min,
            core_global_underflow,
            core_global_subnormal,
            core_disjoint_box_mass,
            core_tile_width,
        ) = core
        coefficient_shape, coefficient_bytes = _exact_array_metadata(
            coefficients,
            dtype=_F64,
            ndim=2,
            name="Dense coefficients",
        )
        if (
            coefficient_shape[0] <= 0
            or coefficient_shape[1] <= 0
            or coefficient_shape[1] != core_weight.shape[0]
        ):
            _raise(
                "SHAPE_MISMATCH",
                "Dense coefficient/core dimensions disagree",
            )
        rows = coefficient_shape[0]
        output_width = core_weight.shape[1]
        nominal_bytes = rows * output_width * 8
        result_bytes = nominal_bytes + rows * (4 * 8 + 2)
        tile_extent = _min(core_tile_width, output_width)
        base_workspace_bytes = (
            coefficient_bytes + nominal_bytes + result_bytes
        )
        extra_workspace_bytes = (
            64 * (rows * coefficient_shape[1])
            + 32 * (coefficient_shape[1] * tile_extent)
            + 96 * (rows * tile_extent)
            + 32
            * (coefficient_shape[1] + tile_extent + rows)
        )
        if (
            nominal_bytes > _MAX_SNAPSHOT_BYTES
            or result_bytes > _MAX_SNAPSHOT_BYTES
            or base_workspace_bytes + extra_workspace_bytes
            > _MAX_CONV_WORKSPACE_BYTES
        ):
            _raise(
                "RESOURCE_LIMIT",
                "Dense execution exceeds the fixed result/workspace budget",
            )
        coefficient = _snapshot_exact_array(
            coefficients,
            dtype=_F64,
            ndim=2,
            name="Dense coefficients",
            expected_shape=coefficient_shape,
        )
        _check_live()
        nominal = _np["asarray"](
            coefficient @ core_weight, dtype=_np["float64"]
        )
        if (
            nominal.shape
            != (coefficient.shape[0], core_weight.shape[1])
            or not _np_all(_np["isfinite"](nominal))
        ):
            _raise(
                "NONFINITE",
                "Dense nominal matrix product is non-finite",
            )
        nominal = _np["ascontiguousarray"](nominal)
        _check_live()

        absolute_coefficients = _np["ascontiguousarray"](
            _np["abs"](coefficient), dtype=_np["float64"]
        )
        support_mass = _dot_up_rows(
            absolute_coefficients, core_support
        )
        gamma, tau = _legacy_f64_parameters(
            2 * _ExactInt(coefficient.shape[1]) + 2
        )
        gamma_term = _ld_mul_up(
            _np["longdouble"](gamma),
            _np["asarray"](support_mass, dtype=_np["longdouble"]),
        )
        tau_term = _ld_mul_up(
            _np["longdouble"](tau),
            _np["longdouble"](core_box_mass),
        )
        wide_guard = _ceil_f64(_ld_add_up(gamma_term, tau_term))
        active = _np_any(
            (coefficient != 0.0)
            & (core_support.reshape(1, -1) != 0.0),
            axis=1,
        )
        wide_guard[~active] = 0.0
        if (
            _np_any(wide_guard < 0)
            or not _np_all(_np["isfinite"](wide_guard))
        ):
            _raise("NUMERIC_GUARD", "invalid Dense wide guard")

        absolute = _np["abs"](coefficient)
        coefficient_subnormal = _np_any(
            (absolute > 0.0) & (absolute < _F64_TINY), axis=1
        )
        _, coefficient_exponents = _np["frexp"](absolute)
        sentinel = _I32_MAX
        exponent_candidates = _np["zeros"](
            coefficient_exponents.shape,
            dtype=coefficient_exponents.dtype,
        )
        _ndarray_fill(exponent_candidates, sentinel)
        exponent_nonzero = absolute != 0.0
        exponent_candidates[exponent_nonzero] = (
            coefficient_exponents[exponent_nonzero]
        )
        row_exponent_min = _np_min(
            exponent_candidates,
            axis=1,
        )
        row_exponent_min = _np["ascontiguousarray"](
            row_exponent_min
        )
        row_exponent_min[row_exponent_min == sentinel] = 0
        support_subnormal = (
            (core_support > 0.0) & (core_support < _F64_TINY)
        )
        used_subnormal_support = _np_any(
            (coefficient != 0.0)
            & support_subnormal.reshape(1, -1),
            axis=1,
        )
        fallback = _np["zeros"](coefficient.shape[0], dtype=_np["bool_"])
        for row in _range(coefficient.shape[0]):
            if not active[row]:
                continue
            row_min = _ExactInt(row_exponent_min[row])
            fallback[row] = _ExactBool(
                core_global_subnormal
                or core_global_underflow
                or core_disjoint_box_mass
                or coefficient_subnormal[row]
                or used_subnormal_support[row]
                or _product_may_be_subnormal(
                    row_min, core_weight_exponent_min
                )
                or _product_may_be_subnormal(
                    row_min, core_support_exponent_min
                )
                or (0.0 < wide_guard[row] < _F64_TINY)
            )

        streamed = _np_full(
            coefficient.shape[0], _math["inf"], dtype=_np["float64"]
        )
        rows = _np_flatnonzero(fallback)
        if rows.size:
            selected = _np["ascontiguousarray"](
                coefficient[rows], dtype=_np["float64"]
            )
            total = _np["zeros"](rows.size, dtype=_np["float64"])
            for start in _range(
                0, core_weight.shape[1], core_tile_width
            ):
                _check_live()
                stop = _min(
                    start + core_tile_width, core_weight.shape[1]
                )
                weight_tile = _np["ascontiguousarray"](
                    core_weight[:, start:stop]
                )
                max_abs_tile = _np["ascontiguousarray"](
                    core_max_abs[start:stop]
                )
                radius = _matrix_product_radius(
                    selected, weight_tile
                )
                if not _np_any(radius):
                    continue
                penalty = _dot_up_rows(radius, max_abs_tile)
                total = _zero_sum(total, penalty)
            streamed[rows] = total
        streamed[~fallback] = wide_guard[~fallback]
        final = wide_guard.copy()
        if rows.size:
            final[fallback] = _np["minimum"](
                wide_guard[fallback], streamed[fallback]
            )
        final[~active] = 0.0
        if (
            _np_any(final < 0)
            or not _np_all(_np["isfinite"](final))
            or _np_any(final > wide_guard)
        ):
            _raise("NUMERIC_GUARD", "invalid Dense final guard")
        _check_live()
        return (
            _DENSE_RESULT_TAG,
            False,
            _result_frame(nominal, _F64),
            _result_frame(support_mass, _F64),
            _result_frame(wide_guard, _F64),
            _result_frame(streamed, _F64),
            _result_frame(final, _F64),
            _result_frame(active, _BOOL),
            _result_frame(fallback, _BOOL),
        )

    def _scaled_guard(
        mass: Any,
        *,
        gamma: float,
        tau: float,
        support_sum: float,
        active: Any,
    ) -> Any:
        result = _np["zeros"](mass.shape, dtype=_np["float64"])
        if _np_any(active):
            first = _ld_mul_up(
                _np["longdouble"](gamma),
                _np["asarray"](mass[active], dtype=_np["longdouble"]),
            )
            second = _ld_mul_up(
                _np["longdouble"](tau),
                _np["longdouble"](support_sum),
            )
            result[active] = _ceil_f64(_ld_add_up(first, second))
        return result

    def _conv_runtime(core: Any, coefficients: Any) -> Any:
        if (
            _ExactType(core) is not _ExactTuple
            or _len(core) != 6
            or core[0] != _CONV_CORE_TAG
        ):
            _raise("CORE_MISMATCH", "invalid private Conv core tuple")
        (
            _,
            core_weight,
            core_input_shape,
            core_output_shape,
            core_groups,
            core_offsets,
        ) = core
        coefficient_shape, coefficient_bytes = _exact_array_metadata(
            coefficients,
            dtype=_F64,
            ndim=2,
            name="Conv coefficients",
        )
        out_c, out_h, out_w = core_output_shape
        in_c, in_h, in_w = core_input_shape
        if (
            coefficient_shape[0] <= 0
            or coefficient_shape[1] != out_c * out_h * out_w
        ):
            _raise(
                "SHAPE_MISMATCH",
                "Conv coefficient/core dimensions disagree",
            )
        batch = coefficient_shape[0]
        input_elements = in_c * in_h * in_w
        nominal_bytes = batch * input_elements * 8
        result_bytes = nominal_bytes + batch * (3 * 8 + 3)
        out_per_group = out_c // core_groups
        in_per_group = in_c // core_groups
        max_offset_workspace_bytes = 0
        for offset in core_offsets:
            _check_live()
            if (
                _ExactType(offset) is not _ExactTuple
                or _len(offset) != 14
                or offset[0] != _OFFSET_CORE_TAG
                or _ExactType(offset[7]) is not _np["ndarray"]
                or _ExactType(offset[8]) is not _np["ndarray"]
                or _ExactType(offset[9]) is not _np["ndarray"]
            ):
                _raise(
                    "CORE_MISMATCH",
                    "invalid private Conv offset tuple",
                )
            valid_h_count = _ExactInt(offset[7].size)
            valid_w_count = _ExactInt(offset[8].size)
            valid_position_count = valid_h_count * valid_w_count
            if (
                valid_h_count <= 0
                or valid_w_count <= 0
                or offset[9].size != valid_position_count
            ):
                _raise(
                    "CORE_MISMATCH",
                    "private Conv offset geometry changed",
                )
            first_take_elements = (
                batch * out_per_group * valid_h_count * out_w
            )
            selected_elements = (
                batch * out_per_group * valid_position_count
            )
            term_elements = (
                batch * in_per_group * valid_position_count
            )
            weight_slice_elements = out_per_group * in_per_group
            support_elements = in_per_group * valid_position_count
            channel_elements = out_per_group * valid_position_count
            offset_workspace_bytes = (
                8 * first_take_elements
                + 64 * selected_elements
                + 64 * term_elements
                + 16 * weight_slice_elements
                + 24 * support_elements
                + 24 * channel_elements
                + 8
                * (
                    valid_h_count
                    + valid_w_count
                    + valid_position_count
                )
            )
            max_offset_workspace_bytes = _max(
                max_offset_workspace_bytes,
                offset_workspace_bytes,
            )
            _check_live()
        base_workspace_bytes = (
            coefficient_bytes
            + nominal_bytes
            + result_bytes
            + 64 * batch
        )
        if (
            nominal_bytes > _MAX_SNAPSHOT_BYTES
            or result_bytes > _MAX_SNAPSHOT_BYTES
            or base_workspace_bytes + max_offset_workspace_bytes
            > _MAX_CONV_WORKSPACE_BYTES
        ):
            _raise(
                "RESOURCE_LIMIT",
                "Conv execution exceeds the fixed result/workspace budget",
            )
        coefficient = _snapshot_exact_array(
            coefficients,
            dtype=_F64,
            ndim=2,
            name="Conv coefficients",
            expected_shape=coefficient_shape,
        )
        nonzero_count = _ExactInt(_np_count_nonzero(coefficient))
        if nonzero_count * 8 <= _ExactInt(coefficient.size):
            _raise(
                "SPARSE_UNCHANGED",
                "sparse Conv rows must use frozen V3 replay",
            )
        _check_live()
        shaped = coefficient.reshape(batch, out_c, out_h, out_w)
        nominal = _np["zeros"](
            (batch, in_c, in_h * in_w), dtype=_np["float64"]
        )
        channel_total = _np["zeros"](batch, dtype=_np["float64"])
        accumulation_total = _np["zeros"](batch, dtype=_np["float64"])
        channel_active_total = _np["zeros"](batch, dtype=_np["bool_"])
        accumulation_active_total = _np["zeros"](
            batch, dtype=_np["bool_"]
        )
        dot_gamma, dot_tau = _conv_f64_parameters(
            2 * out_per_group + 2
        )
        add_gamma, add_tau = _conv_f64_parameters(2)

        for offset in core_offsets:
            if (
                _ExactType(offset) is not _ExactTuple
                or _len(offset) != 14
                or offset[0] != _OFFSET_CORE_TAG
            ):
                _raise(
                    "CORE_MISMATCH",
                    "invalid private Conv offset tuple",
                )
            (
                _,
                offset_co_start,
                offset_co_end,
                offset_ci_start,
                offset_ci_end,
                offset_kh,
                offset_kw,
                offset_output_h,
                offset_output_w,
                offset_targets,
                offset_support,
                offset_channel_support,
                offset_support_activity,
                offset_support_sum,
            ) = offset
            _check_live()
            coeff_group = shaped[
                :, offset_co_start : offset_co_end, :, :
            ]
            nominal_group = nominal[
                :, offset_ci_start : offset_ci_end, :
            ]
            selected = _np_take(
                coeff_group, offset_output_h, axis=2
            )
            selected = _np_take(
                selected, offset_output_w, axis=3
            )
            selected_flat = _np["ascontiguousarray"](
                selected.transpose(0, 2, 3, 1).reshape(batch, -1)
            )
            left = _np["ascontiguousarray"](
                selected.transpose(0, 2, 3, 1).reshape(
                    -1, out_per_group
                )
            )
            weight_slice = _np["ascontiguousarray"](
                core_weight[
                    offset_co_start : offset_co_end,
                    :,
                    offset_kh,
                    offset_kw,
                ]
            )
            term = _np["asarray"](
                left @ weight_slice, dtype=_np["float64"]
            )
            _finite(term, "Conv channel GEMM")
            nh = offset_output_h.size
            nw = offset_output_w.size
            term = term.reshape(
                batch, nh, nw, in_per_group
            ).transpose(0, 3, 1, 2)
            term = _np["ascontiguousarray"](
                term.reshape(batch, in_per_group, -1)
            )
            old = _np["ascontiguousarray"](
                nominal_group[:, :, offset_targets]
            )
            merged = _np["asarray"](old + term, dtype=_np["float64"])
            _finite(merged, "Conv offset addition")
            nominal_group[:, :, offset_targets] = merged

            channel_active = _np_any(
                (selected_flat != 0.0)
                & offset_support_activity.reshape(1, -1),
                axis=1,
            )
            channel_mass = _dot_up_rows(
                _np["abs"](selected_flat), offset_channel_support
            )
            channel_guard = _scaled_guard(
                channel_mass,
                gamma=dot_gamma,
                tau=dot_tau,
                support_sum=offset_support_sum,
                active=channel_active,
            )
            channel_total = _zero_sum(
                channel_total, channel_guard
            )
            channel_active_total |= channel_active

            support_nonzero = offset_support.reshape(1, -1) != 0.0
            addition_active = _np_any(
                support_nonzero
                & (old.reshape(batch, -1) != 0.0)
                & (term.reshape(batch, -1) != 0.0),
                axis=1,
            )
            old_mass = _dot_up_rows(
                _np["abs"](old).reshape(batch, -1), offset_support
            )
            term_mass = _dot_up_rows(
                _np["abs"](term).reshape(batch, -1), offset_support
            )
            addition_mass = _zero_sum(old_mass, term_mass)
            accumulation_guard = _scaled_guard(
                addition_mass,
                gamma=add_gamma,
                tau=add_tau,
                support_sum=offset_support_sum,
                active=addition_active,
            )
            accumulation_total = _zero_sum(
                accumulation_total, accumulation_guard
            )
            accumulation_active_total |= addition_active
            _check_live()
            # Drop every per-offset allocation before the next offset starts.
            # Python otherwise keeps the previous iteration's locals alive
            # while evaluating the next first ``take``, invalidating the
            # max(single-offset) workspace proof above.
            coeff_group = None
            nominal_group = None
            selected = None
            selected_flat = None
            left = None
            weight_slice = None
            term = None
            old = None
            merged = None
            channel_active = None
            channel_mass = None
            channel_guard = None
            support_nonzero = None
            addition_active = None
            old_mass = None
            term_mass = None
            addition_mass = None
            accumulation_guard = None

        active = channel_active_total | accumulation_active_total
        scalar_guard = _zero_sum(
            channel_total, accumulation_total
        )
        scalar_guard[~active] = 0.0
        _check_live()
        return (
            _CONV_RESULT_TAG,
            False,
            _result_frame(nominal.reshape(batch, -1), _F64),
            _result_frame(scalar_guard, _F64),
            _result_frame(channel_total, _F64),
            _result_frame(accumulation_total, _F64),
            _result_frame(active, _BOOL),
            _result_frame(channel_active_total, _BOOL),
            _result_frame(accumulation_active_total, _BOOL),
        )

    port_capability = _ExactObject()

    port_reference = [None]

    def _check_port_identity(value: Any) -> None:
        if (
            _ExactType(value) is not _Port
            or port_reference[0] is None
            or port_reference[0]() is not value
        ):
            _raise(
                "PORT_MISMATCH",
                "private numeric port identity changed",
            )
        if _getpid() != owner_pid:
            _raise(
                "FORKED_PROCESS",
                "a private-kernel capability cannot cross a fork",
            )

    def _check_port(value: Any) -> None:
        _check_port_identity(value)
        _check_live()

    def _numeric_operation(operation: Any) -> Any:
        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Preserve identity/lifecycle error ordering before changing the
            # caller's context-local NumPy policy.
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
                raise _PrivateNumericKernelError(
                    "NUMERIC_GUARD",
                    "numeric execution raised a floating-point exception",
                ) from exc
            except _OverflowError as exc:
                if _ExactType(exc) is not _OverflowError:
                    raise
                raise _PrivateNumericKernelError(
                    "NUMERIC_GUARD",
                    "numeric execution exceeded an exact platform limit",
                ) from exc
            except _MemoryError as exc:
                if (
                    _ExactType(exc) is not _MemoryError
                    and _ExactType(exc) is not _ArrayMemoryError
                ):
                    raise
                raise _PrivateNumericKernelError(
                    "RESOURCE_LIMIT",
                    "numeric execution exceeded the fixed memory budget",
                ) from exc
            finally:
                _leave_numeric_environment(token)

        return guarded

    def _make_port_type() -> Any:
        __slots__ = ("__weakref__",)

        def __new__(cls, capability: Any = None) -> Any:
            if capability is not port_capability:
                _raise(
                    "PORT_CONSTRUCTION",
                    "private numeric ports are factory-minted",
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

        def _check_self(self) -> None:
            _check_port(self)

        def admit_dense(
            self,
            *,
            weight: Any,
            predecessor_max_abs: Any,
            tile_width: int = 256,
        ) -> Any:
            """Admit one exact Dense material snapshot."""

            _check_port(self)
            _check_runtime_numeric_environment()
            if _ExactType(tile_width) is not _ExactInt or tile_width <= 0:
                _raise(
                    "INVALID_TILE",
                    "tile_width must be an exact positive integer",
                )
            weight_shape, weight_bytes = _exact_array_metadata(
                weight,
                dtype=_F64,
                ndim=2,
                name="Dense weight",
            )
            max_abs_shape, max_abs_bytes = _exact_array_metadata(
                predecessor_max_abs,
                dtype=_F64,
                ndim=1,
                name="Dense predecessor_max_abs",
            )
            if (
                weight_shape[0] <= 0
                or weight_shape[1] <= 0
                or weight_shape[1] != max_abs_shape[0]
            ):
                _raise(
                    "SHAPE_MISMATCH",
                    "Dense weight and max-abs dimensions disagree",
                )
            if (
                (weight_bytes + max_abs_bytes) * 4
                > _MAX_SNAPSHOT_BYTES
            ):
                _raise(
                    "RESOURCE_LIMIT",
                    "Dense inputs exceed the aggregate admission budget",
                )
            owned_weight = _snapshot_exact_array(
                weight,
                dtype=_F64,
                ndim=2,
                name="Dense weight",
                expected_shape=weight_shape,
            )
            owned_max_abs = _snapshot_exact_array(
                predecessor_max_abs,
                dtype=_F64,
                ndim=1,
                name="Dense predecessor_max_abs",
                expected_shape=max_abs_shape,
            )
            if _np_any(owned_max_abs < 0.0):
                _raise(
                    "SHAPE_MISMATCH",
                    "Dense predecessor_max_abs is negative",
                )
            _check_port(self)
            try:
                public = _prepare_dense_public(
                    owned_weight,
                    owned_max_abs,
                    deadline=end,
                )
            except _DenseAdmissionError as exc:
                if exc.code == "DEADLINE_EXPIRED":
                    _expire()
                raise _PrivateNumericKernelError(
                    f"ADMISSION_{exc.code}", _ExactStr(exc)
                ) from exc
            _check_port(self)
            if _ExactType(public) is not _DenseSupport:
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Dense admission returned a substituted support type",
                )
            owned_support = _snapshot_exact_array(
                public.support_upper,
                dtype=_F64,
                ndim=1,
                name="admitted Dense support",
            )

            # Bind every numerical support field back to the one raw snapshot
            # instead of trusting even an exact-type public result by itself.
            expected_support = _dot_up_rows(
                _np["ascontiguousarray"](
                    _np["abs"](owned_weight), dtype=_np["float64"]
                ),
                owned_max_abs,
            )
            expected_box_mass = _ExactFloat(
                _dot_up_rows(
                    _np_ones(
                        (1, owned_max_abs.size),
                        dtype=_np["float64"],
                    ),
                    owned_max_abs,
                )[0]
            )
            weight_min, weight_max = _exponent_extrema(
                owned_weight
            )
            max_abs_min, max_abs_max = _exponent_extrema(
                owned_max_abs
            )
            support_min, support_max = _exponent_extrema(
                expected_support
            )
            expected_global_subnormal = _has_subnormal(
                owned_weight
            ) or _has_subnormal(owned_max_abs)
            expected_global_underflow = _product_may_be_subnormal(
                weight_min, max_abs_min
            )
            expected_disjoint = _ExactBool(
                _np_any(
                    (owned_max_abs != 0.0)
                    & ~_np_any(owned_weight != 0.0, axis=0)
                )
            )
            scalar_metadata = (
                public.box_mass_upper,
                public.weight_exponent_min,
                public.weight_exponent_max,
                public.support_exponent_min,
                public.support_exponent_max,
                public.max_abs_exponent_min,
                public.max_abs_exponent_max,
                public.global_underflow_risk,
                public.global_subnormal_operand,
                public.disjoint_box_mass,
            )
            if (
                public.proof_authority is not False
                or _ExactType(public.weight_shape) is not _ExactTuple
                or _any(
                    _ExactType(value) is not _ExactInt
                    for value in public.weight_shape
                )
                or public.weight_shape != owned_weight.shape
                or _ExactType(scalar_metadata[0]) is not _ExactFloat
                or not _math["isfinite"](scalar_metadata[0])
                or scalar_metadata[0] < 0.0
                or _any(
                    value is not None and _ExactType(value) is not _ExactInt
                    for value in scalar_metadata[1:7]
                )
                or _any(
                    _ExactType(value) is not _ExactBool
                    for value in scalar_metadata[7:]
                )
                or owned_support.shape != (owned_weight.shape[0],)
                or _np_any(owned_support < 0.0)
                or not _same_array_bits(
                    owned_support, expected_support
                )
                or public.box_mass_upper != expected_box_mass
                or (
                    public.weight_exponent_min,
                    public.weight_exponent_max,
                )
                != (weight_min, weight_max)
                or (
                    public.support_exponent_min,
                    public.support_exponent_max,
                )
                != (support_min, support_max)
                or (
                    public.max_abs_exponent_min,
                    public.max_abs_exponent_max,
                )
                != (max_abs_min, max_abs_max)
                or public.global_underflow_risk
                is not expected_global_underflow
                or public.global_subnormal_operand
                is not expected_global_subnormal
                or public.disjoint_box_mass is not expected_disjoint
            ):
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Dense admission metadata was substituted",
                )
            core_weight = _immutable(owned_weight, _F64)
            core_max_abs = _immutable(owned_max_abs, _F64)
            core_support = _immutable(expected_support, _F64)
            core_box_mass = expected_box_mass
            core_weight_exponent_min = weight_min
            core_support_exponent_min = support_min
            core_global_underflow = expected_global_underflow
            core_global_subnormal = expected_global_subnormal
            core_disjoint_box_mass = expected_disjoint
            core_tile_width = tile_width
            core = (
                _DENSE_CORE_TAG,
                core_weight,
                core_max_abs,
                core_support,
                core_box_mass,
                core_weight_exponent_min,
                core_support_exponent_min,
                core_global_underflow,
                core_global_subnormal,
                core_disjoint_box_mass,
                core_tile_width,
            )
            _check_runtime_numeric_environment()
            locator = _mint_locator(core)
            with lifecycle_lock:
                _check_live()
                counters["dense_admissions"] += 1
                return locator

        def admit_conv(
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
        ) -> Any:
            """Admit one exact dense-Conv material snapshot."""

            _check_port(self)
            _check_runtime_numeric_environment()
            if _ExactType(layer_id) is not _ExactInt or layer_id < 0:
                _raise(
                    "INVALID_GEOMETRY",
                    "layer_id must be an exact nonnegative integer",
                )
            if _ExactType(groups) is not _ExactInt or groups <= 0:
                _raise(
                    "INVALID_GEOMETRY",
                    "groups must be an exact positive integer",
                )
            weight_shape, weight_bytes = _exact_array_metadata(
                weight,
                dtype=_F64,
                ndim=4,
                name="Conv weight",
            )
            lb_shape, lb_bytes = _exact_array_metadata(
                predecessor_lb,
                dtype=_F64,
                ndim=1,
                name="Conv predecessor lower",
            )
            ub_shape, ub_bytes = _exact_array_metadata(
                predecessor_ub,
                dtype=_F64,
                ndim=1,
                name="Conv predecessor upper",
            )
            input_tuple = _exact_tuple(
                input_shape,
                length=3,
                name="input_shape",
                positive=True,
            )
            output_tuple = _exact_tuple(
                output_shape,
                length=3,
                name="output_shape",
                positive=True,
            )
            stride_tuple = _exact_tuple(
                stride,
                length=2,
                name="stride",
                positive=True,
            )
            padding_tuple = _exact_tuple(
                padding,
                length=2,
                name="padding",
                positive=False,
            )
            dilation_tuple = _exact_tuple(
                dilation,
                length=2,
                name="dilation",
                positive=True,
            )
            all_geometry = (
                input_tuple
                + output_tuple
                + stride_tuple
                + padding_tuple
                + dilation_tuple
            )
            if _any(value > _I64_MAX for value in all_geometry):
                _raise(
                    "INVALID_GEOMETRY",
                    "Conv geometry exceeds the signed int64 index domain",
                )
            if (
                _math["prod"](input_tuple) > _I64_MAX
                or _math["prod"](output_tuple) > _I64_MAX
                or input_tuple[1] * input_tuple[2] - 1 > _I64_MAX
            ):
                _raise(
                    "INVALID_GEOMETRY",
                    "Conv flattened geometry exceeds signed int64",
                )
            for (
                output_extent,
                step,
                pad,
                rate,
                kernel_extent,
            ) in (
                (
                    output_tuple[1],
                    stride_tuple[0],
                    padding_tuple[0],
                    dilation_tuple[0],
                    weight_shape[2],
                ),
                (
                    output_tuple[2],
                    stride_tuple[1],
                    padding_tuple[1],
                    dilation_tuple[1],
                    weight_shape[3],
                ),
            ):
                stepped = (output_extent - 1) * step
                dilated = (kernel_extent - 1) * rate
                final_index = stepped - pad + dilated
                if (
                    stepped > _I64_MAX
                    or dilated > _I64_MAX
                    or final_index > _I64_MAX
                    or final_index < -_I64_MAX - 1
                ):
                    _raise(
                        "INVALID_GEOMETRY",
                        "Conv index arithmetic exceeds signed int64",
                    )
            if (
                _any(extent <= 0 for extent in weight_shape)
                or weight_shape[0] != output_tuple[0]
                or input_tuple[0]
                != weight_shape[1] * groups
                or output_tuple[0] % groups
                or lb_shape != ub_shape
                or lb_shape[0] != _math["prod"](input_tuple)
            ):
                _raise(
                    "INVALID_GEOMETRY",
                    "Conv weight, bounds, groups, and shapes disagree",
                )
            input_elements = _math["prod"](input_tuple)
            output_elements = _math["prod"](output_tuple)
            if (
                input_tuple[1] > _MAX_CONV_AXIS_EXTENT
                or input_tuple[2] > _MAX_CONV_AXIS_EXTENT
                or output_tuple[1] > _MAX_CONV_AXIS_EXTENT
                or output_tuple[2] > _MAX_CONV_AXIS_EXTENT
                or input_elements > _MAX_CONV_ELEMENTS
                or output_elements > _MAX_CONV_ELEMENTS
            ):
                _raise(
                    "RESOURCE_LIMIT",
                    "Conv geometry exceeds the fixed element budget",
                )
            worst_positions = output_tuple[1] * output_tuple[2]
            worst_offsets = (
                groups
                * weight_shape[2]
                * weight_shape[3]
            )
            in_per_group = input_tuple[0] // groups
            out_per_group = output_tuple[0] // groups
            worst_workspace_bytes = (
                worst_offsets
                * worst_positions
                * (in_per_group + out_per_group + 6)
                * 8
            )
            if worst_workspace_bytes > _MAX_CONV_WORKSPACE_BYTES:
                _raise(
                    "RESOURCE_LIMIT",
                    "Conv offset material exceeds the fixed workspace budget",
                )
            base_h = (
                (output_tuple[1] - 1) * stride_tuple[0]
                - 2 * padding_tuple[0]
                + dilation_tuple[0] * (weight_shape[2] - 1)
                + 1
            )
            base_w = (
                (output_tuple[2] - 1) * stride_tuple[1]
                - 2 * padding_tuple[1]
                + dilation_tuple[1] * (weight_shape[3] - 1)
                + 1
            )
            output_padding = (
                input_tuple[1] - base_h,
                input_tuple[2] - base_w,
            )
            if (
                output_padding[0] < 0
                or output_padding[1] < 0
                or output_padding[0] >= stride_tuple[0]
                or output_padding[1] >= stride_tuple[1]
            ):
                _raise(
                    "INVALID_GEOMETRY",
                    "Conv declared shapes violate output-padding contract",
                )
            if (
                (weight_bytes + lb_bytes + ub_bytes) * 4
                > _MAX_SNAPSHOT_BYTES
            ):
                _raise(
                    "RESOURCE_LIMIT",
                    "Conv inputs exceed the aggregate snapshot budget",
                )
            owned_weight = _snapshot_exact_array(
                weight,
                dtype=_F64,
                ndim=4,
                name="Conv weight",
                expected_shape=weight_shape,
            )
            owned_lb = _snapshot_exact_array(
                predecessor_lb,
                dtype=_F64,
                ndim=1,
                name="Conv predecessor lower",
                expected_shape=lb_shape,
            )
            owned_ub = _snapshot_exact_array(
                predecessor_ub,
                dtype=_F64,
                ndim=1,
                name="Conv predecessor upper",
                expected_shape=ub_shape,
            )
            if _np_any(owned_lb > owned_ub):
                _raise(
                    "INVALID_GEOMETRY",
                    "Conv predecessor bounds are reversed",
                )
            _check_port(self)
            if _prepared_mode:
                public = _prepare_conv_public(
                    (
                        b"act.v51b.private.prepared-conv-dispatch.v1",
                        layer_id,
                        input_tuple,
                        output_tuple,
                        stride_tuple,
                        padding_tuple,
                        dilation_tuple,
                        groups,
                    ),
                    (
                        owned_weight.shape,
                        owned_lb.shape,
                        owned_ub.shape,
                    ),
                    deadline=end,
                )
            else:
                params = _MappingProxyType(
                    {
                        "weight": owned_weight,
                        "input_shape": input_tuple,
                        "output_shape": output_tuple,
                        "stride": stride_tuple,
                        "padding": padding_tuple,
                        "dilation": dilation_tuple,
                        "groups": groups,
                    }
                )
                layer = _FrozenLayer(
                    id=layer_id,
                    kind="CONV2D",
                    preds=(layer_id - 1,) if layer_id else (),
                    width=_math["prod"](output_tuple),
                    in_vars=(),
                    out_vars=(),
                    params=params,
                )
                box = _Box(lb=owned_lb, ub=owned_ub)
                public_deadline = _Deadline(end=end)
                try:
                    public = _prepare_conv_public(
                        layer, box, deadline=public_deadline
                    )
                except _ReplayTimeout:
                    _expire()
                except _ReplayError as exc:
                    raise _PrivateNumericKernelError(
                        "ADMISSION_REJECTED", _ExactStr(exc)
                    ) from exc
            _check_port(self)
            if _ExactType(public) is not _ConvPlan:
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Conv admission returned a substituted plan type",
                )
            if _ExactType(public.offsets) is not _ExactTuple:
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Conv admission offset collection was substituted",
                )
            geometry_fields = (
                public.input_shape,
                public.output_shape,
                public.stride,
                public.padding,
                public.dilation,
            )
            if (
                public.proof_authority is not False
                or _ExactType(public.layer_id) is not _ExactInt
                or public.layer_id != layer_id
                or _ExactType(public.groups) is not _ExactInt
                or public.groups != groups
                or _any(
                    _ExactType(value) is not _ExactTuple
                    or _any(_ExactType(item) is not _ExactInt for item in value)
                    for value in geometry_fields
                )
                or geometry_fields
                != (
                    input_tuple,
                    output_tuple,
                    stride_tuple,
                    padding_tuple,
                    dilation_tuple,
                )
            ):
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Conv admission geometry was not bound to raw input",
                )
            admitted_weight = _snapshot_exact_array(
                public.weight,
                dtype=_F64,
                ndim=4,
                name="admitted Conv weight",
            )
            if not _same_array_bits(admitted_weight, owned_weight):
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Conv admission changed the captured weight",
                )
            admitted_support = _snapshot_exact_array(
                public.support,
                dtype=_F64,
                ndim=1,
                name="admitted Conv support",
            )
            expected_support = _np["ascontiguousarray"](
                _np["maximum"](
                    _np["abs"](owned_lb), _np["abs"](owned_ub)
                ),
                dtype=_np["float64"],
            )
            if (
                admitted_support.shape != expected_support.shape
                or not _same_array_bits(
                    admitted_support, expected_support
                )
            ):
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Conv admission support was not bound to raw bounds",
                )
            core_weight = _immutable(owned_weight, _F64)
            core_input_shape = input_tuple
            core_output_shape = output_tuple
            core_groups = groups
            private_offsets = []
            out_c, out_h, out_w = output_tuple
            in_c, in_h, in_w = input_tuple
            out_per_group = out_c // groups
            in_per_group = in_c // groups
            support_view = expected_support.reshape(in_c, -1)
            source_index = 0
            for group in _range(groups):
                co_start = group * out_per_group
                co_end = co_start + out_per_group
                ci_start = group * in_per_group
                ci_end = ci_start + in_per_group
                for kh in _range(owned_weight.shape[2]):
                    _check_port(self)
                    input_h_indices = (
                        _np["arange"](out_h, dtype=_np["int64"])
                        * stride_tuple[0]
                        - padding_tuple[0]
                        + kh * dilation_tuple[0]
                    )
                    valid_h = (
                        (input_h_indices >= 0)
                        & (input_h_indices < in_h)
                    )
                    if not _np_any(valid_h):
                        input_h_indices = None
                        valid_h = None
                        _check_port(self)
                        continue
                    output_h = _np_flatnonzero(valid_h)
                    input_h_indices = input_h_indices[valid_h]
                    for kw in _range(owned_weight.shape[3]):
                        _check_port(self)
                        input_w_indices = (
                            _np["arange"](out_w, dtype=_np["int64"])
                            * stride_tuple[1]
                            - padding_tuple[1]
                            + kw * dilation_tuple[1]
                        )
                        valid_w = (
                            (input_w_indices >= 0)
                            & (input_w_indices < in_w)
                        )
                        if not _np_any(valid_w):
                            input_w_indices = None
                            valid_w = None
                            _check_port(self)
                            continue
                        output_w = _np_flatnonzero(valid_w)
                        input_w_valid = input_w_indices[valid_w]
                        targets = (
                            input_h_indices[:, None] * in_w
                            + input_w_valid[None, :]
                        ).reshape(-1)
                        selected_support = _np["ascontiguousarray"](
                            support_view[ci_start:ci_end, :][
                                :, targets
                            ],
                            dtype=_np["float64"],
                        )
                        weight_abs = _np["ascontiguousarray"](
                            _np["abs"](
                                owned_weight[
                                    co_start:co_end,
                                    :,
                                    kh,
                                    kw,
                                ]
                            ),
                            dtype=_np["float64"],
                        )
                        channel_upper, support_activity = (
                            _dot_up_matrix(
                                weight_abs, selected_support
                            )
                        )
                        expected_output_h = _immutable(
                            output_h, _I64
                        )
                        expected_output_w = _immutable(
                            output_w, _I64
                        )
                        expected_targets = _immutable(
                            targets, _I64
                        )
                        expected_offset_support = _immutable(
                            selected_support.reshape(-1), _F64
                        )
                        expected_channel_support = _immutable(
                            channel_upper.T.reshape(-1), _F64
                        )
                        expected_activity = _immutable(
                            support_activity.T.reshape(-1), _BOOL
                        )
                        expected_support_sum = _ExactFloat(
                            _dot_up_rows(
                                expected_offset_support.reshape(
                                    1, -1
                                ),
                                _np_ones(
                                    expected_offset_support.size,
                                    dtype=_np["float64"],
                                ),
                            )[0]
                        )
                        if source_index >= _len(public.offsets):
                            _raise(
                                "INVALID_ADMISSION_RESULT",
                                "Conv admission omitted an offset",
                            )
                        source = public.offsets[source_index]
                        source_index += 1
                        if _ExactType(source) is not _ConvOffset:
                            _raise(
                                "INVALID_ADMISSION_RESULT",
                                "Conv admission substituted offset type",
                            )
                        integer_metadata = (
                            source.group,
                            source.co_start,
                            source.co_end,
                            source.ci_start,
                            source.ci_end,
                            source.kh,
                            source.kw,
                        )
                        expected_metadata = (
                            group,
                            co_start,
                            co_end,
                            ci_start,
                            ci_end,
                            kh,
                            kw,
                        )
                        if (
                            _any(
                                _ExactType(value) is not _ExactInt
                                for value in integer_metadata
                            )
                            or integer_metadata
                            != expected_metadata
                            or _ExactType(source.support_sum_upper)
                            is not _ExactFloat
                            or not _math["isfinite"](
                                source.support_sum_upper
                            )
                            or source.support_sum_upper
                            != expected_support_sum
                        ):
                            _raise(
                                "INVALID_ADMISSION_RESULT",
                                "Conv offset metadata was not raw-bound",
                            )
                        admitted_output_h = (
                            _snapshot_exact_array(
                                source.output_h_indices,
                                dtype=_I64,
                                ndim=1,
                                name="admitted Conv output-h indices",
                                finite=False,
                            )
                        )
                        admitted_output_w = (
                            _snapshot_exact_array(
                                source.output_w_indices,
                                dtype=_I64,
                                ndim=1,
                                name="admitted Conv output-w indices",
                                finite=False,
                            )
                        )
                        admitted_targets = _snapshot_exact_array(
                            source.targets,
                            dtype=_I64,
                            ndim=1,
                            name="admitted Conv targets",
                            finite=False,
                        )
                        admitted_offset_support = (
                            _snapshot_exact_array(
                                source.support_flat,
                                dtype=_F64,
                                ndim=1,
                                name="admitted Conv offset support",
                            )
                        )
                        admitted_channel_support = (
                            _snapshot_exact_array(
                                source.channel_support_flat,
                                dtype=_F64,
                                ndim=1,
                                name="admitted Conv channel support",
                            )
                        )
                        admitted_activity = _snapshot_exact_array(
                            source.support_activity_flat,
                            dtype=_BOOL,
                            ndim=1,
                            name="admitted Conv support activity",
                            finite=False,
                        )
                        pairs = (
                            (
                                admitted_output_h,
                                expected_output_h,
                            ),
                            (
                                admitted_output_w,
                                expected_output_w,
                            ),
                            (
                                admitted_targets,
                                expected_targets,
                            ),
                            (
                                admitted_offset_support,
                                expected_offset_support,
                            ),
                            (
                                admitted_channel_support,
                                expected_channel_support,
                            ),
                            (
                                admitted_activity,
                                expected_activity,
                            ),
                        )
                        if _any(
                            not _same_array_bits(left, right)
                            for left, right in pairs
                        ):
                            _raise(
                                "INVALID_ADMISSION_RESULT",
                                "Conv offset arrays were not raw-bound",
                            )
                        offset_co_start = co_start
                        offset_co_end = co_end
                        offset_ci_start = ci_start
                        offset_ci_end = ci_end
                        offset_kh = kh
                        offset_kw = kw
                        offset_output_h = expected_output_h
                        offset_output_w = expected_output_w
                        offset_targets = expected_targets
                        offset_support = expected_offset_support
                        offset_channel_support = (
                            expected_channel_support
                        )
                        offset_support_activity = expected_activity
                        offset_support_sum = expected_support_sum
                        offset = (
                            _OFFSET_CORE_TAG,
                            offset_co_start,
                            offset_co_end,
                            offset_ci_start,
                            offset_ci_end,
                            offset_kh,
                            offset_kw,
                            offset_output_h,
                            offset_output_w,
                            offset_targets,
                            offset_support,
                            offset_channel_support,
                            offset_support_activity,
                            offset_support_sum,
                        )
                        private_offsets.append(offset)
                        input_w_indices = None
                        valid_w = None
                        output_w = None
                        input_w_valid = None
                        targets = None
                        selected_support = None
                        weight_abs = None
                        channel_upper = None
                        support_activity = None
                        expected_output_h = None
                        expected_output_w = None
                        expected_targets = None
                        expected_offset_support = None
                        expected_channel_support = None
                        expected_activity = None
                        source = None
                        integer_metadata = None
                        expected_metadata = None
                        admitted_output_h = None
                        admitted_output_w = None
                        admitted_targets = None
                        admitted_offset_support = None
                        admitted_channel_support = None
                        admitted_activity = None
                        pairs = None
                        offset_output_h = None
                        offset_output_w = None
                        offset_targets = None
                        offset_support = None
                        offset_channel_support = None
                        offset_support_activity = None
                        offset = None
                        _check_port(self)
                    input_h_indices = None
                    valid_h = None
                    output_h = None
                    _check_port(self)
            if source_index != _len(public.offsets):
                _raise(
                    "INVALID_ADMISSION_RESULT",
                    "Conv admission added an unexpected offset",
                )
            _check_port(self)
            core_offsets = _ExactTuple(private_offsets)
            core = (
                _CONV_CORE_TAG,
                core_weight,
                core_input_shape,
                core_output_shape,
                core_groups,
                core_offsets,
            )
            _check_runtime_numeric_environment()
            locator = _mint_locator(core)
            with lifecycle_lock:
                _check_live()
                counters["conv_admissions"] += 1
                return locator

        def execute_dense(
            self, locator: Any, coefficients: Any
        ) -> Any:
            """Execute Dense and return the fixed non-authoritative ABI.

            The exact tuple order is ``kind, False, nominal, support_mass,
            wide_guard, streamed_v3_guard, final_guard, active_mask,
            fallback_mask``.  Every field after the first two is an exact
            ``(bytes, shape_tuple, dtype_tag_bytes)`` tuple.  It is a pure
            value, never a capability or valid input to this port.
            """

            _check_port(self)
            _check_runtime_numeric_environment()
            core = _resolve(locator, _DENSE_CORE_TAG)
            result = _dense_runtime(core, coefficients)
            _check_runtime_numeric_environment()
            with lifecycle_lock:
                _check_live()
                counters["dense_executions"] += 1
                return result

        def execute_conv(
            self, locator: Any, coefficients: Any
        ) -> Any:
            """Execute dense-Conv and return the fixed value-only ABI.

            The exact tuple order is ``kind, False, coefficient,
            scalar_guard, channel_dot_guard, accumulation_guard, active_mask,
            channel_dot_active_mask, accumulation_active_mask``.  Array
            fields use the same exact bytes/shape/dtype frame as Dense.  A
            future session must consume it only inside the operation that
            invoked this hidden port, never through a caller round-trip.
            """

            _check_port(self)
            _check_runtime_numeric_environment()
            core = _resolve(locator, _CONV_CORE_TAG)
            result = _conv_runtime(core, coefficients)
            _check_runtime_numeric_environment()
            with lifecycle_lock:
                _check_live()
                counters["conv_executions"] += 1
                return result

        def stats(self) -> Any:
            """Return counts only; no locator or material is exposed."""

            _check_port(self)
            with lifecycle_lock:
                _check_live()
                dense_materials = _sum(
                    _ExactType(value) is _ExactTuple
                    and _len(value) == 11
                    and value[0] == _DENSE_CORE_TAG
                    for value in cores.values()
                )
                conv_materials = _sum(
                    _ExactType(value) is _ExactTuple
                    and _len(value) == 6
                    and value[0] == _CONV_CORE_TAG
                    for value in cores.values()
                )
                return _MappingProxyType(
                    {
                        "material_count": _len(cores),
                        "locator_count": _len(locator_tokens),
                        "dense_materials": dense_materials,
                        "conv_materials": conv_materials,
                        **counters,
                    }
                )

        def close(self) -> None:
            _check_port_identity(self)
            with lifecycle_lock:
                _check_port_identity(self)
                if state[0] == "OPEN":
                    _drop_all("CLOSED")

        guarded_admit_dense = _numeric_operation(admit_dense)
        guarded_admit_conv = _numeric_operation(admit_conv)
        guarded_execute_dense = _numeric_operation(execute_dense)
        guarded_execute_conv = _numeric_operation(execute_conv)

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
                "_check_self": _check_self,
                "admit_dense": guarded_admit_dense,
                "admit_conv": guarded_admit_conv,
                "execute_dense": guarded_execute_dense,
                "execute_conv": guarded_execute_conv,
                "stats": stats,
                "close": close,
            },
        )

    _Port = _make_port_type()
    port = _Port(port_capability)
    port_reference[0] = _weakref_ref(port)
    _check_live()
    _check_instrumentation()
    return port


def _seal_private_numeric_factory(
    implementation: Any,
    module_gates: Any,
    trusted_builtins: Any,
    trusted_numpy: Any,
    trusted_math: Any,
    direct_dependencies: Any,
) -> Any:
    # Retain only immutable implementation components.  A long-lived Python
    # function would expose mutable ``__code__`` and ``__globals__`` through
    # the public factory closure.  Each construction instead receives a fresh
    # function and fresh empty builtin/global dictionaries.
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

    def create_private_numeric_kernel(*, deadline: float) -> Any:
        """Return one dependency-sealed private numeric-kernel port."""

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

    return create_private_numeric_kernel


_TRUSTED_BUILTINS = (
    ("Exception", _ExceptionModule),
    ("FloatingPointError", _FloatingPointErrorModule),
    ("MemoryError", _MemoryErrorModule),
    ("OverflowError", _OverflowErrorModule),
    ("TypeError", _TypeErrorModule),
    ("ValueError", _ValueErrorModule),
    ("any", _any_module),
    ("bool", _ExactBoolModule),
    ("bytes", _ExactBytesModule),
    ("dict", _ExactDictModule),
    ("float", _ExactFloatModule),
    ("int", _ExactIntModule),
    ("id", _id_module),
    ("len", _len_module),
    ("max", _max_module),
    ("memoryview", _memoryview_module),
    ("min", _min_module),
    ("object", _ExactObjectModule),
    ("property", _property_module),
    ("range", _range_module),
    ("str", _ExactStrModule),
    ("sum", _sum_module),
    ("tuple", _ExactTupleModule),
    ("type", _ExactTypeModule),
)

_TRUSTED_NUMPY = (
    ("__version__", _np_module.__version__),
    ("abs", _np_module.abs),
    ("add", _np_module.add),
    ("all", _np_module.all),
    ("any", _np_module.any),
    ("arange", _np_module.arange),
    ("array_equal", _np_module.array_equal),
    ("asarray", _np_module.asarray),
    ("ascontiguousarray", _np_module.ascontiguousarray),
    ("bool_", _np_module.bool_),
    ("count_nonzero", _np_module.count_nonzero),
    ("dtype", _np_module.dtype),
    ("finfo", _np_module.finfo),
    ("flatnonzero", _np_module.flatnonzero),
    ("float64", _np_module.float64),
    ("frexp", _np_module.frexp),
    ("frombuffer", _np_module.frombuffer),
    ("full", _np_module.full),
    ("iinfo", _np_module.iinfo),
    ("int32", _np_module.int32),
    ("int64", _np_module.int64),
    ("isfinite", _np_module.isfinite),
    ("ldexp", _np_module.ldexp),
    ("logical_and", _np_module.logical_and),
    ("logical_or", _np_module.logical_or),
    ("longdouble", _np_module.longdouble),
    ("max", _np_module.max),
    ("maximum", _np_module.maximum),
    ("min", _np_module.min),
    ("minimum", _np_module.minimum),
    ("ndarray", _np_module.ndarray),
    ("nextafter", _np_module.nextafter),
    ("ones", _np_module.ones),
    ("take", _np_module.take),
    ("uint64", _np_module.uint64),
    ("where", _np_module.where),
    ("zeros", _np_module.zeros),
)

_TRUSTED_MATH = (
    ("inf", _math_module.inf),
    ("isfinite", _math_module.isfinite),
    ("prod", _math_module.prod),
)

_AUTHORITY_GATES = (
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
    (
        _weakref_module,
        (
            ("ref", _weakref_module.ref),
        ),
    ),
    (
        _np_exceptions_module,
        (
            ("_ArrayMemoryError", _ArrayMemoryErrorModule),
        ),
    ),
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
    (
        _dense_v51,
        (
            ("DenseV51Support", _dense_v51.DenseV51Support),
            (
                "QueryDualScalarGuardV51Error",
                _dense_v51.QueryDualScalarGuardV51Error,
            ),
            (
                "prepare_dense_support_v51",
                _dense_v51.prepare_dense_support_v51,
            ),
        ),
    ),
    (
        _conv_v51,
        (
            ("DenseConvV51Plan", _conv_v51.DenseConvV51Plan),
            ("_OffsetSupport", _conv_v51._OffsetSupport),
            (
                "prepare_dense_conv_v51_plan",
                _conv_v51.prepare_dense_conv_v51_plan,
            ),
        ),
    ),
    (
        _frozen,
        (
            ("QueryDualReplayError", _frozen.QueryDualReplayError),
            ("QueryDualReplayTimeout", _frozen.QueryDualReplayTimeout),
            ("_Box", _frozen._Box),
            ("_Deadline", _frozen._Deadline),
            ("_FrozenLayer", _frozen._FrozenLayer),
        ),
    ),
)

_DIRECT_DEPENDENCIES = (
    PrivateNumericKernelError,
    PrivateNumericKernelTimeout,
    _ArrayMemoryErrorModule,
    _time_module.monotonic,
    _os_module.getpid,
    _thread_module.RLock,
    _weakref_module.ref,
    _MappingProxyTypeModule,
    _dense_v51.prepare_dense_support_v51,
    _conv_v51.prepare_dense_conv_v51_plan,
    _frozen._FrozenLayer,
    _frozen._Box,
    _frozen._Deadline,
    _frozen.QueryDualReplayTimeout,
    _dense_v51.QueryDualScalarGuardV51Error,
    _frozen.QueryDualReplayError,
    _dense_v51.DenseV51Support,
    _conv_v51.DenseConvV51Plan,
    _conv_v51._OffsetSupport,
    SCHEMA,
    _F64DtypeModule,
    _BoolDtypeModule,
    _I64DtypeModule,
    _F64NmantModule,
    _WideNmantModule,
    _F64EpsModule,
    _WideEpsModule,
    _I32MaxModule,
    _F64ResultTagModule,
    _BoolResultTagModule,
    _F64UnitRoundoffModule,
    _F64EtaModule,
    _F64TinyModule,
    _WideMantissaBitsModule,
    _GatePrimitivesModule,
    _UfuncTypeModule,
    _UfuncReduceDescriptorModule,
    _UfuncInstanceStatesModule,
    _AddReduceModule,
    _LogicalAndReduceModule,
    _LogicalOrReduceModule,
    _MaximumReduceModule,
    _MinimumReduceModule,
    False,
)

create_private_numeric_kernel = _seal_private_numeric_factory(
    _create_private_numeric_kernel_impl,
    _AUTHORITY_GATES,
    _TRUSTED_BUILTINS,
    _TRUSTED_NUMPY,
    _TRUSTED_MATH,
    _DIRECT_DEPENDENCIES,
)

del _create_private_numeric_kernel_impl
del _seal_private_numeric_factory


__all__ = [
    "NUMERIC_PROTOCOL",
    "PrivateNumericKernelError",
    "PrivateNumericKernelTimeout",
    "SCHEMA",
    "create_private_numeric_kernel",
]
