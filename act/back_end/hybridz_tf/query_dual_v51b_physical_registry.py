# ===- query_dual_v51b_physical_registry.py - Physical registry ----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Isolated V5.1b frame-local physical material registry.

This module is deliberately not integrated with the replay session.  It is a
non-authoritative production scaffold for the V5.1b physical-registry
boundary:

* one closure owns one raw frozen root, one bounds frame, all physical
  numerical material, all semantic aliases, and its private HMAC key;
* the first occurrence of an affine layer derives its physical key directly
  from the raw frozen layer and predecessor box and performs one exhaustive
  V5.1 validator;
* later overlapping-cone occurrences mint distinct stage aliases while
  reusing the frame-local physical core;
* execution lookup is an O(1) alias lookup and does not perform numerical
  execution in this phase; and
* commit independently re-walks the raw root, re-derives the complete affine
  set and physical keys, validates the authenticated event chain, and
  exhaustively validates every physical core exactly once.

The Dense V5.1a object combines numerical material and diagnostics.  This
module therefore snapshots the reusable numerical fields into
``DenseNumericCore`` and creates a separate diagnostic digest for every stage
alias.  The original V5.1a object remains private and is used only by the
admission and commit validators.

Every public value has ``proof_authority=False``.  This module cannot issue a
solver verdict.  Its factory-private module views reject persistent class
namespace substitution at operation entry and publication, but pure Python
cannot close an in-operation ``change -> dispatch -> restore`` cycle against
``type.__setattr__``.  Formal integration therefore requires a hidden runtime
seal and numerical callable closures with no retained module or module-view
references.
"""

from __future__ import annotations

import builtins as _builtins_module
import dis
import gc
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time
import weakref
from collections.abc import Mapping as _RuntimeMapping
from dataclasses import dataclass, field
from types import (
    CodeType,
    FunctionType,
    MappingProxyType,
    MemberDescriptorType,
    ModuleType,
)
from typing import Any, Dict, Mapping, NoReturn, Optional, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as dense_v51
from act.back_end.hybridz_tf import query_dual_v51_authority as authority


SCHEMA = "act.query_dual_v51b_physical_registry.v1"
PHYSICAL_KEY_SCHEMA = "act.query_dual_v51b_physical_key.v1"
PHYSICAL_HANDLE_SCHEMA = "act.query_dual_v51b_physical_handle.v1"
STAGE_ALIAS_SCHEMA = "act.query_dual_v51b_stage_alias.v1"
STAGE_ADMISSION_SCHEMA = "act.query_dual_v51b_stage_admission.v1"
COMMIT_SCHEMA = "act.query_dual_v51b_physical_commit.v1"
EVENT_SCHEMA = "act.query_dual_v51b_physical_event.v1"
DENSE_NUMERIC_SCHEMA = "act.query_dual_v51b_dense_numeric_core.v1"
DENSE_STAGE_DIAGNOSTIC_SCHEMA = (
    "act.query_dual_v51b_dense_stage_diagnostic.v1"
)
CONV_STAGE_DIAGNOSTIC_SCHEMA = (
    "act.query_dual_v51b_conv_stage_diagnostic.v1"
)
NUMERIC_PROTOCOL = "frame_local_physical_core_stage_alias_v51b"
_STAGE_USE_SCHEMA = "act.query_dual_v51_stage_use.v1"
_STAGE_TARGET = "TARGET"
_STAGE_PROPERTY = "PROPERTY"

BRANCH_DENSE = authority.BRANCH_DENSE
BRANCH_CONV_DENSE = authority.BRANCH_CONV_DENSE
BOX_OUTPUT = "output_box_v1"
BOX_RELU_POST = "relu_postactivation_from_preactivation_box_v1"
_AFFINE_KINDS = frozenset(("DENSE", "CONV2D"))
_ZERO_SHA256 = "0" * 64
_F64 = np.dtype(np.float64)
_PRIVATE_BUILTINS_TEMPLATE = MappingProxyType(
    dict(vars(_builtins_module))
)
_GC_GET_REFERENTS = gc.get_referents
_HASHLIB_SHA256 = hashlib.sha256
_HMAC_NEW = hmac._hashopenssl.hmac_new
_HMAC_COMPARE_DIGEST = hmac.compare_digest
_JSON_DUMPS = json.dumps
_MATH_ISFINITE = math.isfinite
_OS_GETPID = os.getpid
_SECRETS_TOKEN_BYTES = secrets.token_bytes
_SECRETS_TOKEN_HEX = secrets.token_hex
_THREADING_LOCK = threading.Lock
_TIME_MONOTONIC = time.monotonic
_WEAKREF_FINALIZE = weakref.finalize
_WEAKREF_REF = weakref.ref
_LRU_CACHE_WRAPPER_TYPE = type(dense_v51.check_v51_platform)
_FACTORY_MODULE_OWNERS = MappingProxyType(
    {
        "builtins": _builtins_module,
        "query_dual_replay": frozen,
        "query_dual_scalar_guard_v51": dense_v51,
        "query_dual_replay_v51_conv": conv_v51,
        "query_dual_v51_authority": authority,
        "query_dual_v51b_physical_registry": sys.modules[__name__],
    }
)
_PRIVATE_MODULE_ALIASES_BY_OWNER_ID: Mapping[int, Any] = (
    MappingProxyType({})
)
_PRIVATE_MODULE_ALIAS_MANIFEST: Mapping[str, Any] = MappingProxyType({})
_FACTORY_PUBLIC_GLOBALS: Dict[str, Any] = globals()
_FACTORY_PUBLIC_BINDING_GUARD: Tuple[Any, ...] = ()
_FACTORY_MODULE_ATTRIBUTE_GUARD: Tuple[Any, ...] = ()
_FACTORY_PRIVATE_MODULE_VIEW_TYPE: Any = None
_FACTORY_PRIVATE_MODULE_VIEW_TYPE_SNAPSHOT: Any = None


class V51BPhysicalRegistryError(RuntimeError):
    """Fail-closed error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


class V51BPhysicalRegistryTimeout(V51BPhysicalRegistryError):
    """The one absolute frame deadline expired."""

    def __init__(self, message: str = "V5.1b frame deadline expired"):
        super().__init__("DEADLINE_EXPIRED", message)


def _fail(code: str, message: str) -> NoReturn:
    raise V51BPhysicalRegistryError(code, message)


def _builtins_binding_gate() -> None:
    """Reject public builtin rebinding before any dependency dispatch."""

    source = vars(_FACTORY_MODULE_OWNERS["builtins"])
    for name, expected in _PRIVATE_BUILTINS_TEMPLATE.items():
        if name not in source or source[name] is not expected:
            _fail(
                "DEPENDENCY_SUBSTITUTION",
                f"builtin {name} changed from its import-time binding",
            )


def _factory_public_dependency_gate() -> None:
    """Reject guarded public binding changes without invoking them."""

    for name, existed, expected in _FACTORY_PUBLIC_BINDING_GUARD:
        current_exists = name in _FACTORY_PUBLIC_GLOBALS
        if (
            current_exists is not existed
            or (
                existed
                and _FACTORY_PUBLIC_GLOBALS[name] is not expected
            )
        ):
            _fail(
                "DEPENDENCY_SUBSTITUTION",
                f"physical registry binding {name} changed",
            )
    for source, owner_name, name, existed, expected in (
        _FACTORY_MODULE_ATTRIBUTE_GUARD
    ):
        current_exists = name in source
        if (
            current_exists is not existed
            or (existed and source[name] is not expected)
        ):
            _fail(
                "DEPENDENCY_SUBSTITUTION",
                f"runtime dependency {owner_name}.{name} changed",
            )
    private_view_type = _FACTORY_PRIVATE_MODULE_VIEW_TYPE
    if (
        private_view_type is not None
        and not _private_view_type_matches(
            private_view_type,
            _FACTORY_PRIVATE_MODULE_VIEW_TYPE_SNAPSHOT,
        )
    ):
        _fail(
            "DEPENDENCY_SUBSTITUTION",
            "factory-private module view type changed",
        )


class _CapturedModuleNamespace:
    """Template for a private, immutable module-attribute snapshot."""

    __slots__ = ("_module_name", "_values")

    def __init__(
        self,
        module_name: str,
        values: Dict[str, Any],
        *,
        copy_values: bool = True,
        _setattr: Any = object.__setattr__,
        _mapping_proxy: Any = MappingProxyType,
        _dict: Any = dict,
    ):
        _setattr(self, "_module_name", module_name)
        _setattr(
            self,
            "_values",
            _mapping_proxy(
                _dict(values) if copy_values else values
            ),
        )

    @property
    def __dict__(
        self, _getattribute: Any = object.__getattribute__
    ) -> Mapping[str, Any]:
        return _getattribute(self, "_values")

    def __getattr__(
        self,
        name: str,
        _getattribute: Any = object.__getattribute__,
        _attribute_error: Any = AttributeError,
        _key_error: Any = KeyError,
    ) -> Any:
        try:
            return _getattribute(self, "_values")[name]
        except _key_error as exc:
            module_name = _getattribute(self, "_module_name")
            raise _attribute_error(
                f"{module_name} has no captured attribute {name}"
            ) from exc

    def __setattr__(
        self, name: str, value: Any, _private_fail: Any = _fail
    ) -> NoReturn:
        del name, value
        _private_fail(
            "DEPENDENCY_SUBSTITUTION",
            "captured module namespace is immutable",
        )


def _canonical(value: Any) -> Any:
    if isinstance(value, _RuntimeMapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


def _json_bytes(value: Any) -> bytes:
    return _JSON_DUMPS(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: Any) -> str:
    return _HASHLIB_SHA256(_json_bytes(value)).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, _RuntimeMapping):
        return MappingProxyType(
            {
                str(key): _deep_freeze(item)
                for key, item in sorted(
                    value.items(), key=lambda pair: str(pair[0])
                )
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _is_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, name: str) -> str:
    if not _is_sha256(value):
        _fail("INVALID_BINDING", f"{name} must be a lowercase SHA-256")
    return str(value)


def _source_sha256() -> str:
    try:
        return _HASHLIB_SHA256(Path(__file__).read_bytes()).hexdigest()
    except Exception as exc:
        raise V51BPhysicalRegistryError(
            "SOURCE_UNAVAILABLE",
            f"cannot hash physical-registry implementation: {exc}",
        ) from exc


def _file_sha256(path: Any) -> str:
    try:
        return _HASHLIB_SHA256(Path(path).read_bytes()).hexdigest()
    except Exception as exc:
        raise V51BPhysicalRegistryError(
            "SOURCE_UNAVAILABLE",
            f"cannot hash dependency implementation: {exc}",
        ) from exc


def _callable_name(value: Any) -> str:
    module_name = getattr(value, "__module__", None)
    qualified_name = getattr(value, "__qualname__", None)
    if type(module_name) is not str or type(qualified_name) is not str:
        return (
            f"{type(value).__module__}."
            f"{type(value).__qualname__}"
        )
    return f"{module_name}.{qualified_name}"


def _bytes_backed(value: Any) -> bool:
    if type(value) is not np.ndarray or value.flags.writeable:
        return False
    current: Any = value
    seen = set()
    while type(current) is np.ndarray:
        if id(current) in seen:
            return False
        seen.add(id(current))
        current = current.base
    return isinstance(current, bytes)


def _require_raw_f64(
    value: Any, *, name: str, ndim: Optional[int] = None
) -> np.ndarray:
    if type(value) is not np.ndarray:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact numpy array")
    array = np.asarray(value)
    if (
        array.dtype != _F64
        or not array.dtype.isnative
        or (ndim is not None and array.ndim != ndim)
        or not array.flags.c_contiguous
        or not _bytes_backed(array)
        or not np.all(np.isfinite(array))
    ):
        _fail(
            "RAW_CONTEXT_MISMATCH",
            f"{name} must be finite, C-contiguous, bytes-backed binary64",
        )
    return array


def _immutable_f64(value: Any, *, name: str) -> np.ndarray:
    return frozen._immutable_f64_array(value, name=name)


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    if (
        array.dtype != _F64
        or not array.dtype.isnative
        or not array.flags.c_contiguous
        or not np.all(np.isfinite(array))
    ):
        _fail("INVALID_NUMERIC", "array digest requires finite native binary64")
    digest = _HASHLIB_SHA256()
    digest.update(
        _JSON_DUMPS(
            {"dtype": "<f8", "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(
        array.astype(np.dtype("<f8"), copy=False).tobytes(order="C")
    )
    return digest.hexdigest()


def _validated_deadline(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        _fail("INVALID_DEADLINE", "deadline must be a monotonic timestamp")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise V51BPhysicalRegistryError(
            "INVALID_DEADLINE",
            "deadline must be a monotonic timestamp",
        ) from exc
    if not _MATH_ISFINITE(result):
        _fail("INVALID_DEADLINE", "deadline must be a monotonic timestamp")
    return result


class _NoCopy:
    __slots__ = ()

    def __copy__(self) -> NoReturn:
        _fail("COPY_FORBIDDEN", "process-local handles cannot be copied")

    def __deepcopy__(self, memo: Any) -> NoReturn:
        del memo
        _fail("COPY_FORBIDDEN", "process-local handles cannot be copied")

    def __reduce__(self) -> NoReturn:
        _fail("COPY_FORBIDDEN", "process-local handles cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        _fail("COPY_FORBIDDEN", "process-local handles cannot be serialized")


def _module_owned_function(
    value: Any, module_globals: Dict[str, Any]
) -> Optional[FunctionType]:
    if type(value) is FunctionType and value.__globals__ is module_globals:
        return value
    if type(value) is _LRU_CACHE_WRAPPER_TYPE:
        wrapped = object.__getattribute__(value, "__wrapped__")
        if (
            type(wrapped) is FunctionType
            and wrapped.__globals__ is module_globals
        ):
            return wrapped
    return None


def _clone_function(
    value: FunctionType, isolated_globals: Dict[str, Any]
) -> FunctionType:
    if value.__closure__ is not None:
        _fail(
            "DEPENDENCY_SUBSTITUTION",
            f"dependency function {value.__qualname__} has a live closure",
        )
    clone = FunctionType(
        value.__code__,
        isolated_globals,
        value.__name__,
        value.__defaults__,
        None,
    )
    clone.__kwdefaults__ = (
        None
        if value.__kwdefaults__ is None
        else dict(value.__kwdefaults__)
    )
    clone.__annotations__ = dict(value.__annotations__)
    clone.__dict__.update(value.__dict__)
    clone.__module__ = value.__module__
    clone.__qualname__ = value.__qualname__
    clone.__doc__ = value.__doc__
    return clone


def _recursive_code_objects(value: CodeType) -> Tuple[CodeType, ...]:
    """Return one code object and every nested code constant."""

    result = [value]
    for constant in value.co_consts:
        if type(constant) is CodeType:
            result.extend(_recursive_code_objects(constant))
    return tuple(result)


def _private_function_source(value: Any) -> Optional[FunctionType]:
    """Resolve an exact Python function, including fixed C dispatchers."""

    if type(value) is FunctionType:
        return value
    try:
        wrapped = object.__getattribute__(value, "__wrapped__")
    except (AttributeError, TypeError):
        return None
    return wrapped if type(wrapped) is FunctionType else None


def _function_module_alias_paths(
    functions: Tuple[FunctionType, ...],
    source_globals: Dict[str, Any],
) -> Mapping[str, Tuple[Tuple[str, ...], ...]]:
    """Recursively enumerate module globals loaded by helper bytecode."""

    result: Dict[str, set[Tuple[str, ...]]] = {}
    for function in functions:
        for code in _recursive_code_objects(function.__code__):
            instructions = tuple(dis.get_instructions(code))
            for index, instruction in enumerate(instructions):
                if instruction.opname != "LOAD_GLOBAL":
                    continue
                name = instruction.argval
                if (
                    type(name) is not str
                    or type(source_globals.get(name)) is not ModuleType
                ):
                    continue
                path = []
                following = index + 1
                while (
                    following < len(instructions)
                    and instructions[following].opname
                    in ("LOAD_ATTR", "LOAD_METHOD")
                ):
                    path.append(str(instructions[following].argval))
                    following += 1
                result.setdefault(name, set()).add(tuple(path))
    return MappingProxyType(
        {
            name: tuple(sorted(paths))
            for name, paths in sorted(result.items())
        }
    )


def _reachable_private_functions(
    roots: Tuple[FunctionType, ...],
    source_globals: Dict[str, Any],
) -> Tuple[Mapping[int, FunctionType], Mapping[str, FunctionType]]:
    """Find same-owner Python helpers reached through LOAD_GLOBAL."""

    functions: Dict[int, FunctionType] = {}
    bindings: Dict[str, FunctionType] = {}
    pending = list(roots)
    for root in roots:
        functions[id(root)] = root
    while pending:
        function = pending.pop()
        for code in _recursive_code_objects(function.__code__):
            for instruction in dis.get_instructions(code):
                if instruction.opname != "LOAD_GLOBAL":
                    continue
                name = instruction.argval
                if type(name) is not str:
                    continue
                candidate = _private_function_source(
                    source_globals.get(name)
                )
                if (
                    candidate is None
                    or candidate.__globals__ is not source_globals
                ):
                    continue
                bindings[name] = candidate
                if id(candidate) not in functions:
                    functions[id(candidate)] = candidate
                    pending.append(candidate)
    for name, value in source_globals.items():
        candidate = _private_function_source(value)
        if (
            candidate is not None
            and id(candidate) in functions
            and candidate.__globals__ is source_globals
        ):
            bindings[name] = candidate
    return MappingProxyType(functions), MappingProxyType(bindings)


def _build_private_module_alias_registry(
    owners: Tuple[ModuleType, ...],
    *,
    view_type: type,
) -> Tuple[
    Mapping[int, Any],
    Tuple[Tuple[Dict[str, Any], str, str, bool, Any], ...],
    Mapping[str, Any],
]:
    """Build fixed module views and a bytecode-derived coverage manifest."""

    owner_alias_paths: Dict[
        int, Tuple[ModuleType, Dict[str, set[Tuple[str, ...]]]]
    ] = {}
    required_paths: Dict[
        int, Tuple[ModuleType, set[Tuple[str, ...]]]
    ] = {}
    groups: Dict[int, Dict[str, Any]] = {}
    root_attributes: Dict[Tuple[int, str], Tuple[int, FunctionType]] = {}

    def require_module(module: ModuleType) -> set[Tuple[str, ...]]:
        entry = required_paths.get(id(module))
        if entry is None or entry[0] is not module:
            entry = (module, set())
            required_paths[id(module)] = entry
        return entry[1]

    for owner in owners:
        source = vars(owner)
        functions = []
        for value in source.values():
            owned = _module_owned_function(value, source)
            if owned is not None:
                functions.append(owned)
        discovered = _function_module_alias_paths(
            tuple(functions), source
        )
        alias_paths: Dict[str, set[Tuple[str, ...]]] = {}
        for name, value in source.items():
            if type(value) is not ModuleType:
                continue
            paths = set(discovered.get(name, ()))
            alias_paths[name] = paths
            require_module(value).update(paths)
        owner_alias_paths[id(owner)] = (owner, alias_paths)

    processed_attributes = set()
    scanned_group_functions = set()
    while True:
        changed = False
        for module_id, (module, paths) in tuple(required_paths.items()):
            source = vars(module)
            for path in tuple(paths):
                if not path:
                    continue
                name = path[0]
                attribute_key = (module_id, name)
                if name not in source:
                    processed_attributes.add(attribute_key)
                    continue
                value = source[name]
                if type(value) is ModuleType:
                    nested = require_module(value)
                    remainder = path[1:]
                    if remainder not in nested:
                        nested.add(remainder)
                        changed = True
                if attribute_key in processed_attributes:
                    continue
                processed_attributes.add(attribute_key)
                function = _private_function_source(value)
                if function is None:
                    continue
                if function.__closure__ is not None:
                    _fail(
                        "DEPENDENCY_SUBSTITUTION",
                        (
                            f"captured module callable "
                            f"{module.__name__}.{name} has a live closure"
                        ),
                    )
                source_globals = function.__globals__
                group = groups.setdefault(
                    id(source_globals),
                    {
                        "source": source_globals,
                        "roots": {},
                        "functions": {},
                        "bindings": {},
                    },
                )
                group["roots"][id(function)] = function
                root_attributes[attribute_key] = (
                    id(source_globals),
                    function,
                )
                reachable, bindings = _reachable_private_functions(
                    tuple(group["roots"].values()), source_globals
                )
                before = len(group["functions"])
                group["functions"].update(reachable)
                group["bindings"].update(bindings)
                if len(group["functions"]) != before:
                    changed = True

        for group_id, group in tuple(groups.items()):
            source_globals = group["source"]
            for name, value in source_globals.items():
                if type(value) is ModuleType:
                    require_module(value)
            unscanned = tuple(
                function
                for function_id, function in group["functions"].items()
                if (group_id, function_id)
                not in scanned_group_functions
            )
            if not unscanned:
                continue
            for function in unscanned:
                scanned_group_functions.add((group_id, id(function)))
            discovered = _function_module_alias_paths(
                unscanned, source_globals
            )
            for name, paths in discovered.items():
                module = source_globals[name]
                destination = require_module(module)
                before = len(destination)
                destination.update(paths)
                if len(destination) != before:
                    changed = True
        if not changed:
            pending_attributes = any(
                path
                and (module_id, path[0]) not in processed_attributes
                for module_id, (_, paths) in required_paths.items()
                for path in paths
            )
            if not pending_attributes:
                break

    backings: Dict[int, Dict[str, Any]] = {}
    views: Dict[int, Any] = {}
    for module_id, (module, _) in required_paths.items():
        backing: Dict[str, Any] = {
            "__name__": module.__name__,
        }
        if "__file__" in vars(module):
            backing["__file__"] = vars(module)["__file__"]
        backings[module_id] = backing
        views[module_id] = view_type(
            module.__name__, backing, copy_values=False
        )

    for module_id, (module, paths) in required_paths.items():
        source = vars(module)
        for path in paths:
            if not path:
                continue
            name = path[0]
            if name not in source:
                continue
            value = source[name]
            if type(value) is ModuleType:
                backings[module_id][name] = views[id(value)]
            elif (module_id, name) not in root_attributes:
                backings[module_id][name] = value

    clones_by_group: Dict[int, Mapping[int, FunctionType]] = {}
    for group_id, group in groups.items():
        source_globals = group["source"]
        isolated = dict(source_globals)
        isolated["__builtins__"] = dict(_PRIVATE_BUILTINS_TEMPLATE)
        for name, value in source_globals.items():
            if type(value) is ModuleType:
                isolated[name] = views[id(value)]
        clones = {
            function_id: _clone_function(function, isolated)
            for function_id, function in group["functions"].items()
        }
        for name, function in group["bindings"].items():
            isolated[name] = clones[id(function)]
        clones_by_group[group_id] = MappingProxyType(clones)

    for (module_id, name), (
        group_id,
        function,
    ) in root_attributes.items():
        backings[module_id][name] = clones_by_group[group_id][id(function)]

    registry: Dict[int, Any] = {}
    guard_entries: Dict[Tuple[int, str], Tuple[Any, ...]] = {}
    manifest: Dict[str, Any] = {}
    for owner_id, (owner, alias_paths) in owner_alias_paths.items():
        source = vars(owner)
        entries = []
        manifest_aliases = {}
        for name, paths in sorted(alias_paths.items()):
            module = source[name]
            entries.append((name, module, views[id(module)]))
            guard_entries[(id(source), name)] = (
                source,
                owner.__name__,
                name,
                True,
                module,
            )
            manifest_aliases[name] = {
                "module": module.__name__,
                "attribute_paths": [
                    ".".join(path) for path in sorted(paths) if path
                ],
            }
        registry[owner_id] = (owner, tuple(entries))
        manifest[owner.__name__] = manifest_aliases

    for module, paths in required_paths.values():
        source = vars(module)
        for path in paths:
            if not path:
                continue
            name = path[0]
            existed = name in source
            guard_entries[(id(source), name)] = (
                source,
                module.__name__,
                name,
                existed,
                source.get(name),
            )
    return (
        MappingProxyType(registry),
        tuple(
            guard_entries[key]
            for key in sorted(
                guard_entries,
                key=lambda item: (
                    guard_entries[item][1],
                    guard_entries[item][2],
                ),
            )
        ),
        _deep_freeze(manifest),
    )


def _isolated_module_globals(
    module: Any,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Clone every module-owned function into one private globals dict."""

    source = vars(module)
    if type(source) is not dict:
        _fail(
            "DEPENDENCY_SUBSTITUTION",
            "dependency module has no exact globals dictionary",
        )
    isolated = dict(source)
    isolated["__builtins__"] = dict(_PRIVATE_BUILTINS_TEMPLATE)
    for name, value in _PRIVATE_BUILTINS_TEMPLATE.items():
        isolated[name] = value
    sealed_alias_names = frozenset()
    owner_entry = _PRIVATE_MODULE_ALIASES_BY_OWNER_ID.get(id(module))
    if owner_entry is not None and owner_entry[0] is module:
        sealed_alias_names = frozenset(
            name for name, _, _ in owner_entry[1]
        )
        for name, _, private_view in owner_entry[1]:
            isolated[name] = private_view
    for name, value in source.items():
        if (
            name in _PRIVATE_BUILTINS_TEMPLATE
            or name in sealed_alias_names
        ):
            continue
        owned = _module_owned_function(value, source)
        if owned is not None:
            isolated[name] = _clone_function(owned, isolated)
    if overrides is not None:
        for name, value in overrides.items():
            isolated[name] = value
    for name, _, _ in (() if owner_entry is None else owner_entry[1]):
        if type(isolated.get(name)) is ModuleType:
            _fail(
                "DEPENDENCY_SUBSTITUTION",
                f"isolated module alias {module.__name__}.{name} is live",
            )
    return isolated


def _code_constant(value: Any) -> Any:
    value_type = type(value)
    if value is None or value is Ellipsis or value_type in (bool, int, str):
        return {
            "type": (
                "ellipsis"
                if value is Ellipsis
                else value_type.__qualname__
            ),
            "value": None if value is Ellipsis else value,
        }
    if value_type is float:
        return {"type": "float", "hex": value.hex()}
    if value_type is complex:
        return {
            "type": "complex",
            "real_hex": value.real.hex(),
            "imag_hex": value.imag.hex(),
        }
    if value_type is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if value_type is tuple:
        return {
            "type": "tuple",
            "items": [_code_constant(item) for item in value],
        }
    if value_type is frozenset:
        items = [_code_constant(item) for item in value]
        return {
            "type": "frozenset",
            "items": sorted(items, key=_json_bytes),
        }
    if value_type is CodeType:
        return {"type": "code", "value": _code_implementation_body(value)}
    _fail(
        "SOURCE_UNAVAILABLE",
        f"unsupported code constant {value_type.__qualname__}",
    )


def _code_implementation_body(value: CodeType) -> Mapping[str, Any]:
    return {
        "argcount": value.co_argcount,
        "posonlyargcount": value.co_posonlyargcount,
        "kwonlyargcount": value.co_kwonlyargcount,
        "nlocals": value.co_nlocals,
        "stacksize": value.co_stacksize,
        "flags": value.co_flags,
        "code_hex": value.co_code.hex(),
        "constants": [
            _code_constant(item) for item in value.co_consts
        ],
        "names": list(value.co_names),
        "varnames": list(value.co_varnames),
        "freevars": list(value.co_freevars),
        "cellvars": list(value.co_cellvars),
        "name": value.co_name,
        "qualname": value.co_qualname,
        "filename": value.co_filename,
        "firstlineno": value.co_firstlineno,
        "linetable_hex": value.co_linetable.hex(),
        "exceptiontable_hex": value.co_exceptiontable.hex(),
    }


def _function_implementation_sha256(value: FunctionType) -> str:
    return _json_sha256(
        {
            "code": _code_implementation_body(value.__code__),
            "defaults": value.__defaults__,
            "kwdefaults": value.__kwdefaults__,
        }
    )


def _module_function_manifest(
    module_globals: Dict[str, Any],
) -> Mapping[str, str]:
    result: Dict[str, str] = {}
    for name, value in sorted(module_globals.items()):
        owned = _module_owned_function(value, module_globals)
        if owned is not None:
            result[name] = _function_implementation_sha256(owned)
    return MappingProxyType(result)


def _named_module_function_manifest(
    module_globals: Dict[str, Any],
    names: Tuple[str, ...],
) -> Mapping[str, str]:
    result: Dict[str, str] = {}
    for name in names:
        value = module_globals.get(name)
        owned = _module_owned_function(value, module_globals)
        if owned is None:
            _fail(
                "DEPENDENCY_SUBSTITUTION",
                f"required module function {name} is not exact",
            )
        result[name] = _function_implementation_sha256(owned)
    return MappingProxyType(result)


def _module_binding_snapshot(
    module_globals: Dict[str, Any],
) -> Tuple[Tuple[str, int, Any, Optional[int], Any], ...]:
    """Strongly anchored module bindings and owned-function code objects."""

    values = []
    for name, value in sorted(module_globals.items()):
        owned = _module_owned_function(value, module_globals)
        code = None if owned is None else owned.__code__
        values.append(
            (
                name,
                id(value),
                value,
                None if code is None else id(code),
                code,
            )
        )
    return tuple(values)


def _named_module_binding_snapshot(
    module_globals: Dict[str, Any],
    names: Tuple[str, ...],
) -> Tuple[Tuple[str, int, Any, int, CodeType], ...]:
    """Strongly anchored required bindings and their exact code objects."""

    values = []
    for name in names:
        value = module_globals.get(name)
        owned = _module_owned_function(value, module_globals)
        if owned is None:
            _fail(
                "DEPENDENCY_SUBSTITUTION",
                f"required module function {name} is not exact",
            )
        code = owned.__code__
        values.append((name, id(value), value, id(code), code))
    return tuple(values)


def _private_view_method_snapshot(value: type) -> Any:
    """Strong exact method/code/default anchors for a private view type."""

    namespace = type.__getattribute__(value, "__dict__")
    result = []
    for name in (
        "__getattribute__",
        "__getattr__",
        "__dict__",
        "__setattr__",
        "__reduce_ex__",
    ):
        item = namespace.get(name)
        function = item.fget if type(item) is property else item
        if type(function) is FunctionType:
            code = function.__code__
            defaults = function.__defaults__
            kwdefaults = function.__kwdefaults__
            detail = (
                function,
                code,
                defaults,
                (
                    None
                    if defaults is None
                    else tuple(entry for entry in defaults)
                ),
                kwdefaults,
                (
                    None
                    if kwdefaults is None
                    else tuple(
                        (key, entry)
                        for key, entry in kwdefaults.items()
                    )
                ),
            )
        else:
            detail = (function, None, None, None, None, None)
        result.append((name, item, *detail))
    return tuple(result)


def _private_view_type_snapshot(value: type) -> Any:
    if type(value) is not type:
        _fail(
            "DEPENDENCY_SUBSTITUTION",
            "factory-private view must use the exact builtin metaclass",
        )
    namespace = type.__getattribute__(value, "__dict__")
    return (
        value,
        tuple(namespace.keys()),
        tuple(namespace.values()),
        _private_view_method_snapshot(value),
        tuple(map(id, namespace.values())),
    )


def _private_view_type_matches(value: type, snapshot: Any) -> bool:
    if type(value) is not type or value is not snapshot[0]:
        return False
    namespace = type.__getattribute__(value, "__dict__")
    if (
        tuple(namespace.keys()) != snapshot[1]
        or tuple(map(id, namespace.values())) != snapshot[4]
    ):
        return False
    for (
        name,
        expected_item,
        function,
        expected_code,
        expected_defaults,
        expected_default_items,
        expected_kwdefaults,
        expected_kwdefault_items,
    ) in snapshot[3]:
        if (
            namespace.get(name) is not expected_item
            or type(function) is not FunctionType
        ):
            return False
        current_defaults = function.__defaults__
        current_kwdefaults = function.__kwdefaults__
        if (
            function.__code__ is not expected_code
            or current_defaults is not expected_defaults
            or current_kwdefaults is not expected_kwdefaults
        ):
            return False
        if expected_default_items is not None and (
            type(current_defaults) is not tuple
            or current_defaults != expected_default_items
        ):
            return False
        if expected_kwdefault_items is not None:
            if type(current_kwdefaults) is not dict:
                return False
            if (
                tuple(current_kwdefaults.items())
                != expected_kwdefault_items
            ):
                return False
    return True


def _class_binding_snapshot(value: type) -> Any:
    """Strong exact class-namespace fingerprint without overrides."""

    if type(value) is not type:
        _fail(
            "DEPENDENCY_SUBSTITUTION",
            "public boundary class must use the exact builtin metaclass",
        )

    classes = []
    for cls in type.__getattribute__(value, "__mro__"):
        if cls is object:
            break
        namespace = type.__getattribute__(cls, "__dict__")
        classes.append(
            (
                cls,
                len(namespace),
                tuple(namespace.values()),
                tuple(map(id, namespace.values())),
            )
        )
    return tuple(classes)


def _class_binding_matches(value: type, snapshot: Any) -> bool:
    if type(value) is not type:
        return False
    current_mro = type.__getattribute__(value, "__mro__")
    current_classes = tuple(
        cls for cls in current_mro if cls is not object
    )
    if len(current_classes) != len(snapshot):
        return False
    for cls, (
        expected_class,
        expected_size,
        expected_values,
        expected_value_ids,
    ) in zip(
        current_classes, snapshot
    ):
        if cls is not expected_class:
            return False
        namespace = type.__getattribute__(cls, "__dict__")
        if (
            len(namespace) != expected_size
            or tuple(map(id, namespace.values()))
            != expected_value_ids
        ):
            return False
    return True


def _exact_proxy_backing(value: Any, *, name: str) -> Dict[Any, Any]:
    """Return an exact-dict mappingproxy backing without invoking it."""

    if type(value) is not MappingProxyType:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact mappingproxy")
    referents = _GC_GET_REFERENTS(value)
    if len(referents) != 1 or type(referents[0]) is not dict:
        _fail(
            "RAW_EXACT_TYPE",
            f"{name} must have one exact-dict backing",
        )
    return referents[0]


def _exact_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact int")
    return value


def _exact_string(value: Any, *, name: str) -> str:
    if type(value) is not str:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact str")
    return value


def _exact_optional_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    return _exact_int(value, name=name)


def _exact_int_tuple(
    value: Any, *, name: str, length: Optional[int] = None
) -> Tuple[int, ...]:
    if type(value) is not tuple:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact tuple")
    if length is not None and len(value) != length:
        _fail("RAW_CONTEXT_MISMATCH", f"{name} has the wrong length")
    return tuple(
        _exact_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _validate_exact_tree(
    value: Any, *, name: str, allow_bool: bool = False
) -> None:
    value_type = type(value)
    if value_type is np.ndarray:
        _require_raw_f64(value, name=name)
        return
    if value_type is MappingProxyType:
        backing = _exact_proxy_backing(value, name=name)
        for key, item in backing.items():
            _exact_string(key, name=f"{name} key")
            _validate_exact_tree(
                item, name=f"{name}[{key}]", allow_bool=allow_bool
            )
        return
    if value_type is tuple:
        for index, item in enumerate(value):
            _validate_exact_tree(
                item,
                name=f"{name}[{index}]",
                allow_bool=allow_bool,
            )
        return
    if value is None or value_type in (str, int):
        return
    if value_type is bool and allow_bool:
        return
    if value_type is float:
        if not _MATH_ISFINITE(value):
            _fail("RAW_CONTEXT_MISMATCH", f"{name} is non-finite")
        return
    _fail(
        "RAW_EXACT_TYPE",
        f"{name} contains unsupported type {value_type.__qualname__}",
    )


def _validate_exact_layer(
    layer: Any, *, mapping_key: int, name: str
) -> None:
    if type(layer) is not frozen._FrozenLayer:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact _FrozenLayer")
    layer_id = _exact_int(layer.id, name=f"{name}.id")
    if layer_id != mapping_key:
        _fail("RAW_CONTEXT_MISMATCH", f"{name} id/key mismatch")
    kind = _exact_string(layer.kind, name=f"{name}.kind")
    if kind not in (
        "INPUT",
        "INPUT_SPEC",
        "CONV2D",
        "RELU",
        "ADD",
        "FLATTEN",
        "DENSE",
        "ASSERT",
    ):
        _fail("RAW_CONTEXT_MISMATCH", f"{name} has unsupported kind")
    _exact_int_tuple(layer.preds, name=f"{name}.preds")
    width = _exact_int(layer.width, name=f"{name}.width")
    if width < 0:
        _fail("RAW_CONTEXT_MISMATCH", f"{name}.width is negative")
    if type(layer.in_vars) is not tuple:
        _fail("RAW_EXACT_TYPE", f"{name}.in_vars must be an exact tuple")
    if type(layer.out_vars) is not tuple:
        _fail("RAW_EXACT_TYPE", f"{name}.out_vars must be an exact tuple")
    _validate_exact_tree(
        layer.in_vars, name=f"{name}.in_vars", allow_bool=True
    )
    _validate_exact_tree(
        layer.out_vars, name=f"{name}.out_vars", allow_bool=True
    )
    params = _exact_proxy_backing(
        layer.params, name=f"{name}.params"
    )
    for key, value in params.items():
        _exact_string(key, name=f"{name}.params key")
        _validate_exact_tree(value, name=f"{name}.params[{key}]")

    if kind == "DENSE":
        weight = params.get("weight")
        bias = params.get("bias")
        _require_raw_f64(
            weight, name=f"{name}.params[weight]", ndim=2
        )
        _require_raw_f64(bias, name=f"{name}.params[bias]", ndim=1)
    elif kind == "CONV2D":
        weight = params.get("weight")
        bias = params.get("bias_channels")
        _require_raw_f64(
            weight, name=f"{name}.params[weight]", ndim=4
        )
        _require_raw_f64(
            bias, name=f"{name}.params[bias_channels]", ndim=1
        )
        _exact_int_tuple(
            params.get("input_shape"),
            name=f"{name}.params[input_shape]",
            length=3,
        )
        _exact_int_tuple(
            params.get("output_shape"),
            name=f"{name}.params[output_shape]",
            length=3,
        )
        for geometry_name in ("stride", "padding", "dilation"):
            _exact_int_tuple(
                params.get(geometry_name),
                name=f"{name}.params[{geometry_name}]",
                length=2,
            )
        _exact_int(
            params.get("groups"), name=f"{name}.params[groups]"
        )
    elif kind == "FLATTEN":
        _exact_int(
            params.get("start_dim"), name=f"{name}.params[start_dim]"
        )
        _exact_int(
            params.get("end_dim"), name=f"{name}.params[end_dim]"
        )
    elif kind == "ADD":
        _require_raw_f64(
            params.get("bias"), name=f"{name}.params[bias]", ndim=1
        )


def _validate_exact_box(box: Any, *, name: str) -> None:
    if type(box) is not frozen._Box:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact _Box")
    _require_raw_f64(box.lb, name=f"{name}.lb")
    _require_raw_f64(box.ub, name=f"{name}.ub")


def _validate_exact_context(context: Any, *, name: str) -> None:
    if type(context) is not frozen._SealedCone:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact _SealedCone")
    _exact_optional_int(context.start_lid, name=f"{name}.start_lid")
    layers = _exact_proxy_backing(
        context.layers, name=f"{name}.layers"
    )
    for key, value in layers.items():
        layer_id = _exact_int(key, name=f"{name}.layers key")
        if type(value) is not frozen._FrozenLayer:
            _fail(
                "RAW_EXACT_TYPE",
                f"{name}.layers[{layer_id}] must be exact _FrozenLayer",
            )
    _exact_int_tuple(
        context.reverse_order, name=f"{name}.reverse_order"
    )
    _exact_int(context.output_id, name=f"{name}.output_id")
    _exact_int(context.output_width, name=f"{name}.output_width")
    _exact_string(context.start_mode, name=f"{name}.start_mode")
    _exact_int(context.input_spec_id, name=f"{name}.input_spec_id")
    for field_name in ("replay_net_sha256", "manifest_sha256"):
        value = _exact_string(
            getattr(context, field_name), name=f"{name}.{field_name}"
        )
        if not _is_sha256(value):
            _fail(
                "RAW_CONTEXT_MISMATCH",
                f"{name}.{field_name} is not SHA-256",
            )


def _validate_exact_stage_use(
    value: Any, *, name: str, stage_use_type: type
) -> None:
    if type(value) is not stage_use_type:
        _fail("RAW_EXACT_TYPE", f"{name} must be an exact StageUse")
    use_index = _exact_int(value.use_index, name=f"{name}.use_index")
    stage_kind = _exact_string(
        value.stage_kind, name=f"{name}.stage_kind"
    )
    if use_index < 0:
        _fail("INVALID_STAGE_USE", f"{name}.use_index must be nonnegative")
    if stage_kind == _STAGE_TARGET:
        stage_index = _exact_optional_int(
            value.stage_index, name=f"{name}.stage_index"
        )
        target_relu_lid = _exact_optional_int(
            value.target_relu_lid, name=f"{name}.target_relu_lid"
        )
        cone_start_lid = _exact_optional_int(
            value.cone_start_lid, name=f"{name}.cone_start_lid"
        )
        if (
            stage_index is None
            or target_relu_lid is None
            or cone_start_lid is None
            or stage_index < 0
            or target_relu_lid < 0
            or cone_start_lid < 0
        ):
            _fail(
                "INVALID_STAGE_USE",
                f"{name} TARGET fields must be nonnegative exact ints",
            )
    elif stage_kind == _STAGE_PROPERTY:
        stage_index = _exact_optional_int(
            value.stage_index, name=f"{name}.stage_index"
        )
        target_relu_lid = _exact_optional_int(
            value.target_relu_lid, name=f"{name}.target_relu_lid"
        )
        cone_start_lid = _exact_optional_int(
            value.cone_start_lid, name=f"{name}.cone_start_lid"
        )
        if (
            stage_index is not None
            or target_relu_lid is not None
            or cone_start_lid is not None
        ):
            _fail(
                "INVALID_STAGE_USE",
                f"{name} PROPERTY fields must all be null",
            )
    else:
        _fail(
            "INVALID_STAGE_USE",
            f"{name}.stage_kind must be TARGET or PROPERTY",
        )
    stage_sha = _exact_string(
        value.stage_use_sha256, name=f"{name}.stage_use_sha256"
    )
    if not _is_sha256(stage_sha):
        _fail(
            "RAW_CONTEXT_MISMATCH",
            f"{name}.stage_use_sha256 is not SHA-256",
        )
    expected_sha = _json_sha256(
        {
            "schema": _STAGE_USE_SCHEMA,
            "use_index": use_index,
            "stage_kind": stage_kind,
            "stage_index": stage_index,
            "target_relu_lid": target_relu_lid,
            "cone_start_lid": cone_start_lid,
        }
    )
    if not _HMAC_COMPARE_DIGEST(stage_sha, expected_sha):
        _fail(
            "INVALID_STAGE_USE",
            f"{name}.stage_use_sha256 does not match its exact fields",
        )


def _validate_exact_external_raw(
    *,
    full_layers: Any,
    contexts: Any,
    stage_uses: Any,
    frame_bounds: Any,
    stage_use_type: type,
) -> None:
    layer_backing = _exact_proxy_backing(
        full_layers, name="full_layers"
    )
    context_backing = _exact_proxy_backing(contexts, name="contexts")
    bounds_backing = _exact_proxy_backing(
        frame_bounds, name="frame_bounds"
    )
    if type(stage_uses) is not tuple or not stage_uses:
        _fail(
            "RAW_EXACT_TYPE",
            "stage_uses must be a nonempty exact tuple",
        )

    # First reject every subclass before any dataclass field is touched.
    for key, layer in layer_backing.items():
        _exact_int(key, name="full_layers key")
        if type(layer) is not frozen._FrozenLayer:
            _fail(
                "RAW_EXACT_TYPE",
                "full_layers values must be exact _FrozenLayer",
            )
    for key, box in bounds_backing.items():
        _exact_int(key, name="frame_bounds key")
        if type(box) is not frozen._Box:
            _fail(
                "RAW_EXACT_TYPE",
                "frame_bounds values must be exact _Box",
            )
    for key, context in context_backing.items():
        if key is not None:
            _exact_int(key, name="contexts key")
        if type(context) is not frozen._SealedCone:
            _fail(
                "RAW_EXACT_TYPE",
                "contexts values must be exact _SealedCone",
            )
    for stage_use in stage_uses:
        if type(stage_use) is not stage_use_type:
            _fail(
                "RAW_EXACT_TYPE",
                "stage_uses values must be exact StageUse",
            )

    for key, layer in layer_backing.items():
        _validate_exact_layer(
            layer, mapping_key=key, name=f"full_layers[{key}]"
        )
    for key, box in bounds_backing.items():
        _validate_exact_box(box, name=f"frame_bounds[{key}]")
    for key, context in context_backing.items():
        _validate_exact_context(context, name=f"contexts[{key!r}]")
    for index, stage_use in enumerate(stage_uses):
        _validate_exact_stage_use(
            stage_use,
            name=f"stage_uses[{index}]",
            stage_use_type=stage_use_type,
        )


@dataclass(frozen=True)
class DenseNumericCore:
    """Stage-independent Dense numerical material.

    No V5.1a binding or diagnostic object is retained in this value.
    """

    support_upper: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    box_mass_upper: float
    weight_shape: Tuple[int, int]
    weight_sha256: str
    max_abs_sha256: str
    support_sha256: str
    weight_exponent_min: Optional[int]
    weight_exponent_max: Optional[int]
    support_exponent_min: Optional[int]
    support_exponent_max: Optional[int]
    max_abs_exponent_min: Optional[int]
    max_abs_exponent_max: Optional[int]
    global_underflow_risk: bool
    global_subnormal_operand: bool
    disjoint_box_mass: bool
    content_sha256: str
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("V5.1b numerical material has no authority")
        values = np.asarray(self.support_upper)
        if (
            values.dtype != _F64
            or values.ndim != 1
            or not values.dtype.isnative
            or not values.flags.c_contiguous
            or not _bytes_backed(values)
            or np.any(values < 0.0)
            or not np.all(np.isfinite(values))
            or not all(
                _is_sha256(value)
                for value in (
                    self.weight_sha256,
                    self.max_abs_sha256,
                    self.support_sha256,
                    self.content_sha256,
                )
            )
        ):
            raise ValueError("malformed V5.1b Dense numerical core")


@dataclass(frozen=True, slots=True)
class PhysicalCoreHandle(_NoCopy):
    """Opaque, process-local handle; it never exposes numerical material."""

    operator_branch: str
    layer_id: int
    predecessor_id: int
    physical_key_sha256: str
    core_content_sha256: str
    proof_authority: bool = False
    _token: str = field(default="", repr=False, compare=False)
    _seal: str = field(default="", repr=False, compare=False)
    _capability: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.proof_authority
            or self.operator_branch not in (BRANCH_DENSE, BRANCH_CONV_DENSE)
            or isinstance(self.layer_id, bool)
            or not isinstance(self.layer_id, int)
            or isinstance(self.predecessor_id, bool)
            or not isinstance(self.predecessor_id, int)
            or not all(
                _is_sha256(value)
                for value in (
                    self.physical_key_sha256,
                    self.core_content_sha256,
                    self._seal,
                )
            )
            or not isinstance(self._token, str)
            or not self._token
        ):
            raise ValueError("malformed V5.1b physical handle")


@dataclass(frozen=True, slots=True)
class StageAliasHandle(_NoCopy):
    """One stage occurrence of a frame-local physical core."""

    use_index: int
    stage_use_sha256: str
    layer_id: int
    predecessor_id: int
    physical_core: PhysicalCoreHandle
    stage_diagnostic_schema: str
    stage_diagnostic_sha256: str
    alias_content_sha256: str
    proof_authority: bool = False
    _token: str = field(default="", repr=False, compare=False)
    _seal: str = field(default="", repr=False, compare=False)
    _capability: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.proof_authority
            or isinstance(self.use_index, bool)
            or not isinstance(self.use_index, int)
            or self.use_index < 0
            or not isinstance(self.physical_core, PhysicalCoreHandle)
            or self.stage_diagnostic_schema
            != (
                DENSE_STAGE_DIAGNOSTIC_SCHEMA
                if self.physical_core.operator_branch == BRANCH_DENSE
                else CONV_STAGE_DIAGNOSTIC_SCHEMA
            )
            or not all(
                _is_sha256(value)
                for value in (
                    self.stage_use_sha256,
                    self.stage_diagnostic_sha256,
                    self.alias_content_sha256,
                    self._seal,
                )
            )
            or not isinstance(self._token, str)
            or not self._token
        ):
            raise ValueError("malformed V5.1b stage alias")


@dataclass(frozen=True, slots=True)
class StageAdmission(_NoCopy):
    """Ordered aliases admitted for one independently reconstructed cone."""

    use_index: int
    stage_use_sha256: str
    aliases: Tuple[StageAliasHandle, ...]
    content_sha256: str
    proof_authority: bool = False
    _token: str = field(default="", repr=False, compare=False)
    _seal: str = field(default="", repr=False, compare=False)
    _capability: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.proof_authority
            or isinstance(self.use_index, bool)
            or not isinstance(self.use_index, int)
            or self.use_index < 0
            or not isinstance(self.aliases, tuple)
            or not self.aliases
            or any(
                not isinstance(value, StageAliasHandle)
                for value in self.aliases
            )
            or not all(
                _is_sha256(value)
                for value in (
                    self.stage_use_sha256,
                    self.content_sha256,
                    self._seal,
                )
            )
            or not isinstance(self._token, str)
            or not self._token
        ):
            raise ValueError("malformed V5.1b stage admission")


@dataclass(frozen=True, slots=True)
class RegistryStats(_NoCopy):
    physical_builds: int
    dense_physical_builds: int
    conv_physical_builds: int
    stage_aliases: int
    cross_stage_physical_hits: int
    execution_alias_lookups: int
    admission_full_validations: int
    private_execution_full_validations: int
    commit_full_validations: int
    admitted_stages: int
    event_count: int
    event_chain_head_sha256: str
    state: str
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("V5.1b registry statistics have no authority")


@dataclass(frozen=True, slots=True)
class PhysicalRegistryCertificate(_NoCopy):
    """Terminal non-authoritative diagnostic; never accepted back as proof."""

    physical_builds: int
    dense_physical_builds: int
    conv_physical_builds: int
    stage_aliases: int
    cross_stage_physical_hits: int
    execution_alias_lookups: int
    admission_full_validations: int
    private_execution_full_validations: int
    commit_full_validations: int
    receipt: Mapping[str, Any]
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.proof_authority
            or type(self.receipt) is not MappingProxyType
        ):
            raise ValueError("malformed V5.1b registry certificate")


@dataclass(frozen=True)
class _SourceSpec:
    locator: Tuple[int, int]
    layer: frozen._FrozenLayer
    predecessor_layer: frozen._FrozenLayer
    raw_box: frozen._Box
    effective_box: frozen._Box
    max_abs: np.ndarray
    operator_branch: str
    physical_key_body: Mapping[str, Any]
    physical_key_sha256: str
    layer_params_identity: Any = field(repr=False, compare=False)
    raw_lb_identity: Any = field(repr=False, compare=False)
    raw_ub_identity: Any = field(repr=False, compare=False)


@dataclass
class _CoreRecord:
    spec: _SourceSpec
    material: Any
    numeric_core: Optional[DenseNumericCore]
    core_content_sha256: str
    handle: PhysicalCoreHandle
    handle_body: Mapping[str, Any]
    handle_seal: str
    handle_fast_fields: Tuple[Any, ...]


@dataclass
class _AliasRecord:
    use_index: int
    layer_id: int
    predecessor_id: int
    physical_key_sha256: str
    alias_body: Mapping[str, Any]
    handle: StageAliasHandle
    handle_body: Mapping[str, Any]
    handle_seal: str
    handle_fast_fields: Tuple[Any, ...]


@dataclass
class _AdmissionRecord:
    use_index: int
    alias_by_layer: Mapping[int, _AliasRecord]
    handle: StageAdmission
    handle_body: Mapping[str, Any]
    handle_seal: str
    fast_fields: Tuple[Any, ...]


@dataclass(frozen=True)
class _Event:
    sequence: int
    operation: str
    payload: Mapping[str, Any]
    previous_mac: str
    mac: str


def _root_manifest(
    full_layers: Mapping[int, frozen._FrozenLayer],
    *,
    layer_manifest: Any = None,
) -> Tuple[str, Mapping[int, str]]:
    if layer_manifest is None:
        layer_manifest = frozen._layer_manifest
    backing = _exact_proxy_backing(full_layers, name="full_layers")
    if not backing:
        _fail(
            "INVALID_ROOT",
            "full_layers must be a nonempty immutable mapping proxy",
        )
    manifests = []
    per_layer: Dict[int, str] = {}
    for raw_lid in backing:
        _exact_int(raw_lid, name="full_layers key")
        if type(backing[raw_lid]) is not frozen._FrozenLayer:
            _fail(
                "RAW_EXACT_TYPE",
                "full_layers values must be exact _FrozenLayer",
            )
    for raw_lid in sorted(backing):
        layer = backing[raw_lid]
        if (
            layer.id != raw_lid
        ):
            _fail("INVALID_ROOT", f"malformed frozen layer {raw_lid}")
        params = _exact_proxy_backing(
            layer.params, name=f"layer[{raw_lid}].params"
        )
        for name, value in params.items():
            if type(value) is np.ndarray:
                _require_raw_f64(
                    value, name=f"layer[{raw_lid}].params[{name}]"
                )
        manifest = layer_manifest(layer)
        digest = _json_sha256(manifest)
        per_layer[raw_lid] = digest
        manifests.append(manifest)
    return _json_sha256(manifests), MappingProxyType(per_layer)


def _frame_bounds_manifest(
    full_layers: Mapping[int, frozen._FrozenLayer],
    frame_bounds: Mapping[int, frozen._Box],
) -> str:
    layer_backing = _exact_proxy_backing(
        full_layers, name="full_layers"
    )
    bounds_backing = _exact_proxy_backing(
        frame_bounds, name="frame_bounds"
    )
    if not bounds_backing:
        _fail(
            "INVALID_FRAME",
            "frame_bounds must be a nonempty immutable mapping proxy",
        )
    manifest = []
    for raw_lid, box in bounds_backing.items():
        _exact_int(raw_lid, name="frame_bounds key")
        if type(box) is not frozen._Box:
            _fail(
                "RAW_EXACT_TYPE",
                "frame_bounds values must be exact _Box",
            )
    for raw_lid in sorted(bounds_backing):
        if raw_lid not in layer_backing:
            _fail("INVALID_FRAME", "frame bounds reference an unknown layer")
        box = bounds_backing[raw_lid]
        lb = _require_raw_f64(box.lb, name=f"bounds[{raw_lid}].lb")
        ub = _require_raw_f64(box.ub, name=f"bounds[{raw_lid}].ub")
        if (
            lb.shape != ub.shape
            or lb.size != layer_backing[raw_lid].width
            or np.any(lb > ub)
        ):
            _fail("INVALID_FRAME", f"malformed bounds for layer {raw_lid}")
        manifest.append(
            {
                "layer_id": raw_lid,
                "semantics": (
                    "preactivation"
                    if layer_backing[raw_lid].kind == "RELU"
                    else "output"
                ),
                "lb_sha256": _array_sha256(lb),
                "ub_sha256": _array_sha256(ub),
            }
        )
    return _json_sha256(manifest)


def _identity_tree(value: Any) -> Any:
    """Strong identity snapshot for already immutable raw objects."""

    if type(value) is np.ndarray:
        return (
            "ndarray",
            id(value),
            value,
            str(value.dtype),
            tuple(value.shape),
            tuple(value.strides),
            bool(value.flags.c_contiguous),
            bool(value.flags.writeable),
        )
    if isinstance(value, _RuntimeMapping):
        return (
            "mapping",
            id(value),
            value,
            tuple(
                (
                    type(key).__qualname__,
                    repr(key),
                    _identity_tree(item),
                )
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: (
                        type(pair[0]).__qualname__,
                        repr(pair[0]),
                    ),
                )
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_identity_tree(item) for item in value))
    if isinstance(value, list):
        return (
            "list",
            id(value),
            value,
            tuple(_identity_tree(item) for item in value),
        )
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return (type(value).__qualname__, value)
    return (
        "opaque",
        type(value).__module__,
        type(value).__qualname__,
        id(value),
        value,
    )


def _raw_identity_snapshot(
    *,
    full_layers: Mapping[int, frozen._FrozenLayer],
    contexts: Mapping[Optional[int], frozen._SealedCone],
    stage_uses: Tuple[authority.StageUse, ...],
    frame_bounds: Mapping[int, frozen._Box],
) -> Any:
    layer_snapshot = tuple(
        (
            lid,
            id(layer),
            layer,
            layer.id,
            layer.kind,
            tuple(layer.preds),
            layer.width,
            _identity_tree(layer.in_vars),
            _identity_tree(layer.out_vars),
            _identity_tree(layer.params),
        )
        for lid, layer in sorted(full_layers.items())
    )
    bounds_snapshot = tuple(
        (
            lid,
            id(box),
            box,
            _identity_tree(box.lb),
            _identity_tree(box.ub),
        )
        for lid, box in sorted(frame_bounds.items())
    )
    context_snapshot = tuple(
        (
            start,
            id(context),
            context,
            context.start_lid,
            tuple(context.reverse_order),
            context.output_id,
            context.output_width,
            context.start_mode,
            context.input_spec_id,
            context.replay_net_sha256,
            context.manifest_sha256,
            _identity_tree(context.layers),
        )
        for start, context in sorted(
            contexts.items(),
            key=lambda pair: (
                pair[0] is None,
                -1 if pair[0] is None else int(pair[0]),
            ),
        )
    )
    stage_snapshot = tuple(
        (
            id(value),
            value,
            value.use_index,
            value.stage_kind,
            value.stage_index,
            value.target_relu_lid,
            value.cone_start_lid,
            value.stage_use_sha256,
        )
        for value in stage_uses
    )
    return (
        id(full_layers),
        full_layers,
        layer_snapshot,
        id(contexts),
        contexts,
        context_snapshot,
        id(stage_uses),
        stage_uses,
        stage_snapshot,
        id(frame_bounds),
        frame_bounds,
        bounds_snapshot,
    )


def _owned_value(value: Any) -> Any:
    if type(value) is np.ndarray:
        # Raw numerical arrays were already required to be bytes-backed.  A
        # private container may therefore share their immutable byte storage
        # without duplicating model-sized tensors.
        return value
    if type(value) is MappingProxyType:
        backing = _exact_proxy_backing(value, name="owned value")
        return MappingProxyType(
            {
                key: _owned_value(item)
                for key, item in backing.items()
            }
        )
    if type(value) is tuple:
        return tuple(_owned_value(item) for item in value)
    return value


def _owned_raw_snapshot(
    *,
    full_layers: Mapping[int, frozen._FrozenLayer],
    contexts: Mapping[Optional[int], frozen._SealedCone],
    stage_uses: Tuple[authority.StageUse, ...],
    frame_bounds: Mapping[int, frozen._Box],
    assert_id: int,
    stage_use_type: type,
) -> Tuple[
    Mapping[int, frozen._FrozenLayer],
    Mapping[Optional[int], frozen._SealedCone],
    Tuple[authority.StageUse, ...],
    Mapping[int, frozen._Box],
]:
    layer_backing = _exact_proxy_backing(
        full_layers, name="full_layers"
    )
    context_backing = _exact_proxy_backing(contexts, name="contexts")
    bounds_backing = _exact_proxy_backing(
        frame_bounds, name="frame_bounds"
    )
    owned_layers = MappingProxyType(
        {
            lid: frozen._FrozenLayer(
                id=layer.id,
                kind=layer.kind,
                preds=tuple(layer.preds),
                width=layer.width,
                in_vars=tuple(_owned_value(layer.in_vars)),
                out_vars=tuple(_owned_value(layer.out_vars)),
                params=MappingProxyType(
                    {
                        key: _owned_value(item)
                        for key, item in _exact_proxy_backing(
                            layer.params,
                            name=f"full_layers[{lid}].params",
                        ).items()
                    }
                ),
            )
            for lid, layer in layer_backing.items()
        }
    )
    manifests = {
        lid: frozen._layer_manifest(layer)
        for lid, layer in owned_layers.items()
    }
    owned_contexts = MappingProxyType(
        {
            start: frozen._sealed_cone(
                owned_layers,
                manifests,
                assert_id=assert_id,
                start_lid=start,
            )
            for start in context_backing
        }
    )
    owned_uses_list = []
    for index, value in enumerate(stage_uses):
        owned = object.__new__(stage_use_type)
        for field_name in (
            "use_index",
            "stage_kind",
            "stage_index",
            "target_relu_lid",
            "cone_start_lid",
            "stage_use_sha256",
        ):
            object.__setattr__(owned, field_name, getattr(value, field_name))
        _validate_exact_stage_use(
            owned,
            name=f"owned_stage_uses[{index}]",
            stage_use_type=stage_use_type,
        )
        owned_uses_list.append(owned)
    owned_uses = tuple(owned_uses_list)
    owned_bounds = MappingProxyType(
        {
            lid: frozen._Box(lb=box.lb, ub=box.ub)
            for lid, box in bounds_backing.items()
        }
    )
    return owned_layers, owned_contexts, owned_uses, owned_bounds


def _assert_id(full_layers: Mapping[int, frozen._FrozenLayer]) -> int:
    values = [
        lid for lid, layer in full_layers.items() if layer.kind == "ASSERT"
    ]
    if len(values) != 1:
        _fail("INVALID_ROOT", "raw root must have exactly one ASSERT")
    assertion = full_layers[values[0]]
    if len(assertion.preds) != 1:
        _fail("INVALID_ROOT", "terminal ASSERT must have one predecessor")
    return values[0]


def _independent_reverse_order(
    full_layers: Mapping[int, frozen._FrozenLayer],
    *,
    assert_id: int,
    start_lid: Optional[int],
) -> Tuple[int, ...]:
    if start_lid is None:
        assertion = full_layers.get(assert_id)
        if (
            assertion is None
            or assertion.kind != "ASSERT"
            or len(assertion.preds) != 1
        ):
            _fail("RAW_CONTEXT_MISMATCH", "raw root ASSERT changed")
        output_id = int(assertion.preds[0])
    else:
        if (
            isinstance(start_lid, bool)
            or not isinstance(start_lid, int)
            or start_lid not in full_layers
            or full_layers[start_lid].kind == "ASSERT"
        ):
            _fail("RAW_CONTEXT_MISMATCH", "invalid cone start")
        output_id = start_lid

    state: Dict[int, int] = {}
    topo = []

    def visit(lid: int) -> None:
        mark = state.get(lid, 0)
        if mark == 1:
            _fail("RAW_CONTEXT_MISMATCH", "cycle in raw root")
        if mark == 2:
            return
        layer = full_layers.get(lid)
        if type(layer) is not frozen._FrozenLayer:
            _fail("RAW_CONTEXT_MISMATCH", f"unknown raw layer {lid}")
        if layer.kind == "ASSERT":
            _fail("RAW_CONTEXT_MISMATCH", "ASSERT entered a replay cone")
        if len(set(layer.preds)) != len(layer.preds):
            _fail("RAW_CONTEXT_MISMATCH", f"layer {lid} repeats a predecessor")
        state[lid] = 1
        for predecessor in layer.preds:
            visit(int(predecessor))
        state[lid] = 2
        topo.append(lid)

    visit(output_id)
    input_specs = [
        lid for lid in topo if full_layers[lid].kind == "INPUT_SPEC"
    ]
    if len(input_specs) != 1:
        _fail(
            "RAW_CONTEXT_MISMATCH",
            "each raw replay cone needs exactly one INPUT_SPEC",
        )
    return tuple(reversed(topo))


def _stage_start(stage_use: authority.StageUse) -> Optional[int]:
    return (
        int(stage_use.cone_start_lid)
        if stage_use.stage_kind == _STAGE_TARGET
        else None
    )


def _validate_context(
    *,
    full_layers: Mapping[int, frozen._FrozenLayer],
    contexts: Mapping[Optional[int], frozen._SealedCone],
    stage_use: authority.StageUse,
    assert_id: int,
) -> Tuple[int, ...]:
    start = _stage_start(stage_use)
    if stage_use.stage_kind == _STAGE_TARGET:
        target = full_layers.get(stage_use.target_relu_lid)
        if (
            type(target) is not frozen._FrozenLayer
            or target.kind != "RELU"
            or tuple(target.preds) != (start,)
        ):
            _fail(
                "RAW_CONTEXT_MISMATCH",
                "target StageUse is not bound to its raw ReLU predecessor",
            )
    context = contexts.get(start)
    if type(context) is not frozen._SealedCone:
        _fail("RAW_CONTEXT_MISMATCH", f"missing sealed cone {start!r}")
    expected = _independent_reverse_order(
        full_layers, assert_id=assert_id, start_lid=start
    )
    if (
        context.start_lid != start
        or tuple(context.reverse_order) != expected
        or set(context.layers) != set(expected)
        or any(
            context.layers[lid] is not full_layers[lid]
            for lid in expected
        )
    ):
        _fail("RAW_CONTEXT_MISMATCH", f"sealed cone {start!r} changed")
    return expected


def _stage_use_body(value: authority.StageUse) -> Mapping[str, Any]:
    return {
        "use_index": value.use_index,
        "stage_kind": value.stage_kind,
        "stage_index": value.stage_index,
        "target_relu_lid": value.target_relu_lid,
        "cone_start_lid": value.cone_start_lid,
        "stage_use_sha256": value.stage_use_sha256,
    }


def _physical_handle_body(value: PhysicalCoreHandle) -> Mapping[str, Any]:
    return {
        "schema": PHYSICAL_HANDLE_SCHEMA,
        "operator_branch": value.operator_branch,
        "layer_id": value.layer_id,
        "predecessor_id": value.predecessor_id,
        "physical_key_sha256": value.physical_key_sha256,
        "core_content_sha256": value.core_content_sha256,
        "proof_authority": value.proof_authority,
        "token": value._token,
    }


def _alias_handle_body(value: StageAliasHandle) -> Mapping[str, Any]:
    return {
        "schema": STAGE_ALIAS_SCHEMA,
        "use_index": value.use_index,
        "stage_use_sha256": value.stage_use_sha256,
        "layer_id": value.layer_id,
        "predecessor_id": value.predecessor_id,
        "physical_key_sha256": value.physical_core.physical_key_sha256,
        "core_content_sha256": value.physical_core.core_content_sha256,
        "stage_diagnostic_schema": value.stage_diagnostic_schema,
        "stage_diagnostic_sha256": value.stage_diagnostic_sha256,
        "alias_content_sha256": value.alias_content_sha256,
        "proof_authority": value.proof_authority,
        "token": value._token,
    }


def _admission_handle_body(value: StageAdmission) -> Mapping[str, Any]:
    return {
        "schema": STAGE_ADMISSION_SCHEMA,
        "use_index": value.use_index,
        "stage_use_sha256": value.stage_use_sha256,
        "alias_content_sha256": [
            alias.alias_content_sha256 for alias in value.aliases
        ],
        "content_sha256": value.content_sha256,
        "proof_authority": value.proof_authority,
        "token": value._token,
    }


def _sealed_hmac(secret: bytes, domain: str, body: Any) -> str:
    payload = domain.encode("ascii") + b"\0" + _json_bytes(body)
    return _HMAC_NEW(
        secret, payload, digestmod=_HASHLIB_SHA256
    ).hexdigest()


def _dense_numeric_body(
    support: dense_v51.DenseV51Support,
) -> Mapping[str, Any]:
    return {
        "schema": DENSE_NUMERIC_SCHEMA,
        "support_upper_sha256": _array_sha256(support.support_upper),
        "box_mass_upper_hex": float(support.box_mass_upper).hex(),
        "weight_shape": list(support.weight_shape),
        "weight_sha256": support.weight_sha256,
        "max_abs_sha256": support.max_abs_sha256,
        "support_sha256": support.support_sha256,
        "weight_exponent_min": support.weight_exponent_min,
        "weight_exponent_max": support.weight_exponent_max,
        "support_exponent_min": support.support_exponent_min,
        "support_exponent_max": support.support_exponent_max,
        "max_abs_exponent_min": support.max_abs_exponent_min,
        "max_abs_exponent_max": support.max_abs_exponent_max,
        "global_underflow_risk": support.global_underflow_risk,
        "global_subnormal_operand": support.global_subnormal_operand,
        "disjoint_box_mass": support.disjoint_box_mass,
        "proof_authority": False,
    }


def _dense_numeric_core(
    support: dense_v51.DenseV51Support,
    *,
    immutable_f64: Any = None,
) -> DenseNumericCore:
    if immutable_f64 is None:
        immutable_f64 = _immutable_f64
    values = immutable_f64(
        support.support_upper, name="V5.1b Dense numerical support"
    )
    body = _dense_numeric_body(support)
    return DenseNumericCore(
        support_upper=values,
        box_mass_upper=float(support.box_mass_upper),
        weight_shape=tuple(support.weight_shape),
        weight_sha256=support.weight_sha256,
        max_abs_sha256=support.max_abs_sha256,
        support_sha256=support.support_sha256,
        weight_exponent_min=support.weight_exponent_min,
        weight_exponent_max=support.weight_exponent_max,
        support_exponent_min=support.support_exponent_min,
        support_exponent_max=support.support_exponent_max,
        max_abs_exponent_min=support.max_abs_exponent_min,
        max_abs_exponent_max=support.max_abs_exponent_max,
        global_underflow_risk=support.global_underflow_risk,
        global_subnormal_operand=support.global_subnormal_operand,
        disjoint_box_mass=support.disjoint_box_mass,
        content_sha256=_json_sha256(body),
    )


def _validate_dense_numeric_core(
    value: DenseNumericCore,
    support: dense_v51.DenseV51Support,
    *,
    immutable_f64: Any = None,
) -> None:
    if immutable_f64 is None:
        immutable_f64 = _immutable_f64
    rebuilt = _dense_numeric_core(
        support, immutable_f64=immutable_f64
    )
    scalar_fields = (
        "box_mass_upper",
        "weight_shape",
        "weight_sha256",
        "max_abs_sha256",
        "support_sha256",
        "weight_exponent_min",
        "weight_exponent_max",
        "support_exponent_min",
        "support_exponent_max",
        "max_abs_exponent_min",
        "max_abs_exponent_max",
        "global_underflow_risk",
        "global_subnormal_operand",
        "disjoint_box_mass",
        "content_sha256",
        "proof_authority",
    )
    if type(value) is not DenseNumericCore:
        _fail("MATERIAL_SUBSTITUTION", "Dense numerical core changed")
    value_support = np.asarray(value.support_upper)
    rebuilt_support = np.asarray(rebuilt.support_upper)
    if (
        any(
            getattr(value, name) != getattr(rebuilt, name)
            for name in scalar_fields
        )
        or value_support.dtype != rebuilt_support.dtype
        or value_support.shape != rebuilt_support.shape
        or value_support.tobytes(order="C")
        != rebuilt_support.tobytes(order="C")
    ):
        _fail("MATERIAL_SUBSTITUTION", "Dense numerical core changed")


def _validate_dense_support_against_fresh(
    actual: Any,
    fresh: Any,
    *,
    support_type: type,
    diagnostics_type: type,
) -> None:
    """Compare every Dense material field to an independent fresh rebuild."""

    if type(actual) is not support_type or type(fresh) is not support_type:
        _fail("MATERIAL_SUBSTITUTION", "Dense support type changed")
    actual_values = np.asarray(actual.support_upper)
    fresh_values = np.asarray(fresh.support_upper)
    if (
        actual_values.dtype != fresh_values.dtype
        or actual_values.shape != fresh_values.shape
        or actual_values.strides != fresh_values.strides
        or actual_values.tobytes(order="C")
        != fresh_values.tobytes(order="C")
    ):
        _fail(
            "MATERIAL_SUBSTITUTION",
            "Dense support vector differs from its fresh rebuild",
        )
    scalar_fields = (
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
        "proof_authority",
    )
    if (
        actual.box_mass_upper.hex() != fresh.box_mass_upper.hex()
        or any(
            getattr(actual, name) != getattr(fresh, name)
            for name in scalar_fields
        )
    ):
        _fail(
            "MATERIAL_SUBSTITUTION",
            "Dense support metadata differs from its fresh rebuild",
        )
    actual_diagnostics = actual.diagnostics
    fresh_diagnostics = fresh.diagnostics
    if (
        type(actual_diagnostics) is not diagnostics_type
        or type(fresh_diagnostics) is not diagnostics_type
        or actual_diagnostics.items != fresh_diagnostics.items
        or not _HMAC_COMPARE_DIGEST(
            actual_diagnostics.sha256,
            fresh_diagnostics.sha256,
        )
    ):
        _fail(
            "MATERIAL_SUBSTITUTION",
            "Dense diagnostics differ from their fresh rebuild",
        )


def _derive_source_spec(
    *,
    full_layers: Mapping[int, frozen._FrozenLayer],
    frame_bounds: Mapping[int, frozen._Box],
    layer_id: int,
    root_content_sha256: str,
    raw_root_manifest_sha256: str,
    frame_content_sha256: str,
    bounds_manifest_sha256: str,
    numeric_contract_sha256: str,
    numeric_platform_sha256: str,
    implementation_sha256: str,
    dependency_implementation_sha256: str,
    module_source_sha256: str,
    conv_geometry: Any,
    frozen_layer_type: Any,
    frozen_box_type: Any,
    immutable_f64: Any,
) -> _SourceSpec:
    layer = full_layers.get(layer_id)
    if type(layer) is not frozen_layer_type:
        _fail("RAW_CONTEXT_MISMATCH", f"missing frozen layer {layer_id}")
    if layer.kind not in _AFFINE_KINDS or len(layer.preds) != 1:
        _fail("RAW_CONTEXT_MISMATCH", f"layer {layer_id} is not unary affine")
    predecessor_id = int(layer.preds[0])
    predecessor = full_layers.get(predecessor_id)
    raw_box = frame_bounds.get(predecessor_id)
    if (
        type(predecessor) is not frozen_layer_type
        or type(raw_box) is not frozen_box_type
    ):
        _fail(
            "RAW_CONTEXT_MISMATCH",
            f"layer {layer_id} has no raw predecessor anchor",
        )
    raw_lb = _require_raw_f64(
        raw_box.lb, name=f"bounds[{predecessor_id}].lb"
    ).reshape(-1)
    raw_ub = _require_raw_f64(
        raw_box.ub, name=f"bounds[{predecessor_id}].ub"
    ).reshape(-1)
    if (
        raw_lb.shape != raw_ub.shape
        or raw_lb.size != predecessor.width
        or np.any(raw_lb > raw_ub)
    ):
        _fail(
            "RAW_CONTEXT_MISMATCH",
            f"predecessor box {predecessor_id} changed",
        )
    if predecessor.kind == "RELU":
        effective_lb = immutable_f64(
            np.maximum(raw_lb, 0.0), name="V5.1b ReLU-post lower"
        )
        effective_ub = immutable_f64(
            np.maximum(raw_ub, 0.0), name="V5.1b ReLU-post upper"
        )
        semantics = BOX_RELU_POST
    else:
        effective_lb = immutable_f64(
            raw_lb, name="V5.1b predecessor lower"
        )
        effective_ub = immutable_f64(
            raw_ub, name="V5.1b predecessor upper"
        )
        semantics = BOX_OUTPUT
    max_abs = immutable_f64(
        np.maximum(np.abs(effective_lb), np.abs(effective_ub)),
        name="V5.1b predecessor max-abs",
    )
    weight = _require_raw_f64(
        layer.params.get("weight"),
        name=f"layer[{layer_id}].weight",
        ndim=2 if layer.kind == "DENSE" else 4,
    )
    if layer.kind == "DENSE":
        if (
            weight.shape[0] != layer.width
            or weight.shape[1] != max_abs.size
        ):
            _fail("RAW_CONTEXT_MISMATCH", "Dense geometry changed")
        operator_branch = BRANCH_DENSE
        geometry = {
            "weight_shape": list(weight.shape),
            "input_width": int(weight.shape[1]),
            "output_width": int(weight.shape[0]),
        }
    else:
        geometry_value = conv_geometry(layer)
        if int(np.prod(geometry_value["input_shape"])) != max_abs.size:
            _fail("RAW_CONTEXT_MISMATCH", "Conv input geometry changed")
        operator_branch = BRANCH_CONV_DENSE
        geometry = {
            "weight_shape": list(weight.shape),
            "input_shape": list(geometry_value["input_shape"]),
            "output_shape": list(geometry_value["output_shape"]),
            "stride": list(geometry_value["stride"]),
            "padding": list(geometry_value["padding"]),
            "dilation": list(geometry_value["dilation"]),
            "groups": int(geometry_value["groups"]),
        }
    body = {
        "schema": PHYSICAL_KEY_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "root_content_sha256": root_content_sha256,
        "raw_root_manifest_sha256": raw_root_manifest_sha256,
        "frame_content_sha256": frame_content_sha256,
        "bounds_manifest_sha256": bounds_manifest_sha256,
        "numeric_contract_sha256": numeric_contract_sha256,
        "numeric_platform_sha256": numeric_platform_sha256,
        "implementation_sha256": implementation_sha256,
        "dependency_implementation_sha256": (
            dependency_implementation_sha256
        ),
        "module_source_sha256": module_source_sha256,
        "operator_branch": operator_branch,
        "layer_id": layer_id,
        "predecessor_id": predecessor_id,
        "predecessor_kind": predecessor.kind,
        "box_semantics": semantics,
        "raw_lb_sha256": _array_sha256(raw_lb),
        "raw_ub_sha256": _array_sha256(raw_ub),
        "effective_lb_sha256": _array_sha256(effective_lb),
        "effective_ub_sha256": _array_sha256(effective_ub),
        "max_abs_sha256": _array_sha256(max_abs),
        "weight_sha256": _array_sha256(weight),
        "geometry": geometry,
        "proof_authority": False,
    }
    frozen_body = _deep_freeze(body)
    return _SourceSpec(
        locator=(layer_id, predecessor_id),
        layer=layer,
        predecessor_layer=predecessor,
        raw_box=raw_box,
        effective_box=frozen_box_type(lb=effective_lb, ub=effective_ub),
        max_abs=max_abs,
        operator_branch=operator_branch,
        physical_key_body=frozen_body,
        physical_key_sha256=_json_sha256(body),
        layer_params_identity=layer.params,
        raw_lb_identity=raw_box.lb,
        raw_ub_identity=raw_box.ub,
    )


def _open_v51b_frame_physical_registry_impl(
    *,
    full_layers: Mapping[int, frozen._FrozenLayer],
    contexts: Mapping[Optional[int], frozen._SealedCone],
    stage_uses: Tuple[authority.StageUse, ...],
    frame_bounds: Mapping[int, frozen._Box],
    root_content_sha256: str,
    frame_content_sha256: str,
    numeric_contract_sha256: str,
    implementation_sha256: str,
    deadline: Optional[float],
) -> Any:
    """Open one isolated frame-local physical registry.

    The returned object's class is factory-local.  Its only state is held by
    the closures below; it has no state dictionary, material attribute, seal
    key, or registry table.
    """

    _builtins_binding_gate()
    _factory_public_dependency_gate()
    root_content = _require_sha256(
        root_content_sha256, name="root_content_sha256"
    )
    frame_content = _require_sha256(
        frame_content_sha256, name="frame_content_sha256"
    )
    numeric_contract = _require_sha256(
        numeric_contract_sha256, name="numeric_contract_sha256"
    )
    implementation = _require_sha256(
        implementation_sha256, name="implementation_sha256"
    )
    checked_deadline = _validated_deadline(deadline)
    if (
        checked_deadline is not None
        and _TIME_MONOTONIC() >= checked_deadline
    ):
        raise V51BPhysicalRegistryTimeout()
    stage_use_type = authority.StageUse
    _validate_exact_external_raw(
        full_layers=full_layers,
        contexts=contexts,
        stage_uses=stage_uses,
        frame_bounds=frame_bounds,
        stage_use_type=stage_use_type,
    )
    raw_full_layers = full_layers
    raw_contexts = contexts
    raw_stage_uses = stage_uses
    raw_frame_bounds = frame_bounds
    initial_raw_identity = _raw_identity_snapshot(
        full_layers=raw_full_layers,
        contexts=raw_contexts,
        stage_uses=raw_stage_uses,
        frame_bounds=raw_frame_bounds,
    )

    for index, stage_use in enumerate(stage_uses):
        if (
            stage_use.use_index != index
        ):
            _fail(
                "INVALID_STAGE_USE",
                "stage uses must be sealed and consecutively ordered",
            )
    if len({value.stage_use_sha256 for value in stage_uses}) != len(
        stage_uses
    ):
        _fail("INVALID_STAGE_USE", "frame repeats a sealed stage use")
    target_indices = tuple(
        value.stage_index
        for value in stage_uses
        if value.stage_kind == _STAGE_TARGET
    )
    if len(set(target_indices)) != len(target_indices):
        _fail("INVALID_STAGE_USE", "frame repeats a target stage index")
    if (
        sum(
            value.stage_kind == _STAGE_PROPERTY
            for value in stage_uses
        )
        > 1
    ):
        _fail("INVALID_STAGE_USE", "frame has more than one property use")

    raw_assert_id = _assert_id(full_layers)
    captured_schedules: Dict[int, Tuple[int, ...]] = {}
    for stage_use in stage_uses:
        reverse_order = _validate_context(
            full_layers=full_layers,
            contexts=contexts,
            stage_use=stage_use,
            assert_id=raw_assert_id,
        )
        affine = tuple(
            lid
            for lid in reverse_order
            if full_layers[lid].kind in _AFFINE_KINDS
        )
        if not affine:
            _fail("INVALID_STAGE_USE", "stage cone has no affine layer")
        captured_schedules[stage_use.use_index] = affine

    (
        owned_full_layers,
        owned_contexts,
        owned_stage_uses,
        owned_frame_bounds,
    ) = _owned_raw_snapshot(
        full_layers=full_layers,
        contexts=contexts,
        stage_uses=stage_uses,
        frame_bounds=frame_bounds,
        assert_id=raw_assert_id,
        stage_use_type=stage_use_type,
    )
    _validate_exact_external_raw(
        full_layers=raw_full_layers,
        contexts=raw_contexts,
        stage_uses=raw_stage_uses,
        frame_bounds=raw_frame_bounds,
        stage_use_type=stage_use_type,
    )
    if (
        _raw_identity_snapshot(
            full_layers=raw_full_layers,
            contexts=raw_contexts,
            stage_uses=raw_stage_uses,
            frame_bounds=raw_frame_bounds,
        )
        != initial_raw_identity
    ):
        _fail(
            "RAW_CONTEXT_MISMATCH",
            "raw root/frame changed while the owned snapshot was built",
        )
    full_layers = owned_full_layers
    contexts = owned_contexts
    stage_uses = owned_stage_uses
    frame_bounds = owned_frame_bounds

    # Build factory-private dependency namespaces.  Cloned functions share
    # private globals dictionaries, so a later assignment in an imported
    # module (including a transitive helper) cannot redirect their calls.
    dependency_owners = _FACTORY_MODULE_OWNERS
    frozen_owner = dependency_owners["query_dual_replay"]
    dense_owner = dependency_owners[
        "query_dual_scalar_guard_v51"
    ]
    conv_owner = dependency_owners["query_dual_replay_v51_conv"]
    authority_owner = dependency_owners["query_dual_v51_authority"]
    physical_owner = dependency_owners[
        "query_dual_v51b_physical_registry"
    ]
    dependency_module_globals = {
        "query_dual_replay": vars(frozen_owner),
        "query_dual_scalar_guard_v51": vars(dense_owner),
        "query_dual_replay_v51_conv": vars(conv_owner),
        "query_dual_v51_authority": vars(authority_owner),
    }
    module_binding_snapshot = _module_binding_snapshot
    named_module_binding_snapshot = _named_module_binding_snapshot
    dependency_binding_anchors = {
        name: module_binding_snapshot(values)
        for name, values in dependency_module_globals.items()
    }
    dependency_function_manifests = {
        name: _module_function_manifest(values)
        for name, values in dependency_module_globals.items()
    }
    dependency_function_anchors = {}
    for module_name, module_globals in dependency_module_globals.items():
        anchors: Dict[str, FunctionType] = {}
        for name, value in module_globals.items():
            owned = _module_owned_function(value, module_globals)
            if owned is not None:
                anchors[name] = owned
        dependency_function_anchors[module_name] = MappingProxyType(
            anchors
        )

    frozen_globals = _isolated_module_globals(frozen_owner)
    private_view_type = _FACTORY_PRIVATE_MODULE_VIEW_TYPE
    private_view_type_snapshot = (
        _FACTORY_PRIVATE_MODULE_VIEW_TYPE_SNAPSHOT
    )
    frozen_view = private_view_type(
        frozen_owner.__name__,
        frozen_globals,
        copy_values=False,
    )
    dense_globals = _isolated_module_globals(
        dense_owner, overrides={"_v3": frozen_view}
    )
    conv_globals = _isolated_module_globals(
        conv_owner, overrides={"frozen": frozen_view}
    )
    authority_globals = _isolated_module_globals(authority_owner)
    dense_view = private_view_type(
        dense_owner.__name__,
        dense_globals,
        copy_values=False,
    )
    conv_view = private_view_type(
        conv_owner.__name__,
        conv_globals,
        copy_values=False,
    )
    authority_view = private_view_type(
        authority_owner.__name__,
        authority_globals,
        copy_values=False,
    )
    physical_module_globals = vars(physical_owner)
    physical_function_names = (
        "_array_sha256",
        "_builtins_binding_gate",
        "_bytes_backed",
        "_canonical",
        "_class_binding_matches",
        "_class_binding_snapshot",
        "_private_view_method_snapshot",
        "_private_view_type_matches",
        "_private_view_type_snapshot",
        "_deep_freeze",
        "_dense_numeric_body",
        "_dense_numeric_core",
        "_derive_source_spec",
        "_fail",
        "_json_bytes",
        "_json_sha256",
        "_module_binding_snapshot",
        "_module_owned_function",
        "_named_module_binding_snapshot",
        "_require_raw_f64",
        "_source_sha256",
        "_validate_dense_numeric_core",
        "_validate_dense_support_against_fresh",
    )
    physical_binding_anchor = named_module_binding_snapshot(
        physical_module_globals, physical_function_names
    )
    physical_function_manifest = _named_module_function_manifest(
        physical_module_globals, physical_function_names
    )
    dependency_function_manifests[
        "query_dual_v51b_physical_registry"
    ] = physical_function_manifest
    physical_globals = _isolated_module_globals(
        physical_owner,
        overrides={
            "authority": authority_view,
            "conv_v51": conv_view,
            "dense_v51": dense_view,
            "frozen": frozen_view,
            "_GC_GET_REFERENTS": _GC_GET_REFERENTS,
            "_HASHLIB_SHA256": _HASHLIB_SHA256,
            "_HMAC_NEW": _HMAC_NEW,
            "_HMAC_COMPARE_DIGEST": _HMAC_COMPARE_DIGEST,
            "_JSON_DUMPS": _JSON_DUMPS,
            "_MATH_ISFINITE": _MATH_ISFINITE,
            "_OS_GETPID": _OS_GETPID,
            "_RuntimeMapping": _RuntimeMapping,
            "_SECRETS_TOKEN_BYTES": _SECRETS_TOKEN_BYTES,
            "_SECRETS_TOKEN_HEX": _SECRETS_TOKEN_HEX,
            "_THREADING_LOCK": _THREADING_LOCK,
            "_TIME_MONOTONIC": _TIME_MONOTONIC,
            "_WEAKREF_FINALIZE": _WEAKREF_FINALIZE,
            "_WEAKREF_REF": _WEAKREF_REF,
        },
    )
    physical_globals["_JSON_DUMPS"] = physical_globals[
        "json"
    ].dumps
    private_token_hex = physical_globals["secrets"].token_hex
    physical_globals["_SECRETS_TOKEN_HEX"] = private_token_hex
    physical_globals["_SECRETS_TOKEN_BYTES"] = (
        private_token_hex.__globals__["token_bytes"]
    )
    module_binding_snapshot = physical_globals[
        "_module_binding_snapshot"
    ]
    named_module_binding_snapshot = physical_globals[
        "_named_module_binding_snapshot"
    ]
    class_binding_snapshot = physical_globals[
        "_class_binding_snapshot"
    ]
    class_binding_matches = physical_globals[
        "_class_binding_matches"
    ]
    private_view_type_matches = physical_globals[
        "_private_view_type_matches"
    ]
    dependency_fail = physical_globals["_fail"]
    boundary_type_names = (
        "PhysicalCoreHandle",
        "StageAliasHandle",
        "StageAdmission",
        "RegistryStats",
        "PhysicalRegistryCertificate",
    )
    boundary_types = {
        name: physical_module_globals[name]
        for name in boundary_type_names
    }
    boundary_type_anchors = {
        name: class_binding_snapshot(value)
        for name, value in boundary_types.items()
    }
    port_abi: Dict[str, Any] = {
        "type": None,
        "snapshot": None,
    }
    isolated_globals_by_module = {
        "query_dual_replay": frozen_globals,
        "query_dual_scalar_guard_v51": dense_globals,
        "query_dual_replay_v51_conv": conv_globals,
        "query_dual_v51_authority": authority_globals,
    }
    for module_name, anchors in dependency_function_anchors.items():
        isolated_globals = isolated_globals_by_module[module_name]
        for name, original in anchors.items():
            isolated = isolated_globals.get(name)
            if (
                type(isolated) is not FunctionType
                or isolated.__code__ is not original.__code__
                or isolated.__defaults__ != original.__defaults__
                or isolated.__kwdefaults__ != original.__kwdefaults__
            ):
                _fail(
                    "DEPENDENCY_SUBSTITUTION",
                    f"{module_name}.{name} changed while isolated",
                )
    for name, _, _, code_id, _ in physical_binding_anchor:
        original = _module_owned_function(
            physical_module_globals.get(name), physical_module_globals
        )
        isolated = physical_globals.get(name)
        if (
            original is None
            or type(isolated) is not FunctionType
            or id(isolated.__code__) != code_id
            or isolated.__defaults__ != original.__defaults__
            or isolated.__kwdefaults__ != original.__kwdefaults__
        ):
            _fail(
                "DEPENDENCY_SUBSTITUTION",
                f"physical registry helper {name} changed while isolated",
            )

    def private_view_abi_gate() -> None:
        if not private_view_type_matches(
            private_view_type, private_view_type_snapshot
        ):
            dependency_fail(
                "DEPENDENCY_SUBSTITUTION",
                "factory-private module view type changed",
            )

    def dependency_binding_gate() -> None:
        private_view_abi_gate()
        _builtins_binding_gate()
        _factory_public_dependency_gate()
        for module_name, module_globals in (
            dependency_module_globals.items()
        ):
            if (
                module_binding_snapshot(module_globals)
                != dependency_binding_anchors[module_name]
            ):
                dependency_fail(
                    "DEPENDENCY_SUBSTITUTION",
                    f"{module_name} globals changed after registry open",
                )
        if (
            named_module_binding_snapshot(
                physical_module_globals, physical_function_names
            )
            != physical_binding_anchor
        ):
            dependency_fail(
                "DEPENDENCY_SUBSTITUTION",
                "physical registry key helpers changed after registry open",
            )
        public_abi_gate()

    def public_abi_gate() -> None:
        for name in boundary_type_names:
            expected_type = boundary_types[name]
            if (
                physical_module_globals.get(name) is not expected_type
                or not class_binding_matches(
                    expected_type, boundary_type_anchors[name]
                )
            ):
                dependency_fail(
                    "PUBLIC_ABI_SUBSTITUTION",
                    f"public boundary class {name} changed after open",
                )
        port_type = port_abi["type"]
        if (
            port_type is not None
            and (
                not class_binding_matches(
                    port_type, port_abi["snapshot"]
                )
            )
        ):
            dependency_fail(
                "PUBLIC_ABI_SUBSTITUTION",
                "registry port class changed after publication",
            )

    dependency_binding_gate()
    dense_platform_check = dense_globals["check_v51_platform"]
    dense_prepare = dense_globals["prepare_dense_support_v51"]
    dense_validate = dense_globals["_validate_support"]
    dense_array_sha256 = dense_globals["_array_sha256"]
    dense_support_type = dense_globals["DenseV51Support"]
    dense_diagnostics_type = dense_globals["V51Diagnostics"]
    conv_platform_check = conv_globals["_wide_platform"]
    conv_geometry = conv_globals["_geometry"]
    conv_prepare = conv_globals["prepare_dense_conv_v51_plan"]
    conv_validate = conv_globals["_validate_plan"]
    conv_plan_type = conv_globals["DenseConvV51Plan"]
    frozen_layer_manifest = frozen_globals["_layer_manifest"]
    frozen_immutable_f64 = frozen_globals["_immutable_f64_array"]
    frozen_layer_type = frozen_globals["_FrozenLayer"]
    frozen_box_type = frozen_globals["_Box"]
    derive_source_spec = physical_globals["_derive_source_spec"]
    dense_numeric_core_builder = physical_globals[
        "_dense_numeric_core"
    ]
    validate_dense_numeric_core = physical_globals[
        "_validate_dense_numeric_core"
    ]
    validate_dense_support_fresh = physical_globals[
        "_validate_dense_support_against_fresh"
    ]
    physical_core_handle_type = boundary_types["PhysicalCoreHandle"]
    stage_alias_handle_type = boundary_types["StageAliasHandle"]
    stage_admission_type = boundary_types["StageAdmission"]
    registry_stats_type = boundary_types["RegistryStats"]
    physical_registry_certificate_type = boundary_types[
        "PhysicalRegistryCertificate"
    ]
    boundary_field_names = {
        physical_core_handle_type: (
            "operator_branch",
            "layer_id",
            "predecessor_id",
            "physical_key_sha256",
            "core_content_sha256",
            "proof_authority",
            "_token",
            "_seal",
            "_capability",
        ),
        stage_alias_handle_type: (
            "use_index",
            "stage_use_sha256",
            "layer_id",
            "predecessor_id",
            "physical_core",
            "stage_diagnostic_schema",
            "stage_diagnostic_sha256",
            "alias_content_sha256",
            "proof_authority",
            "_token",
            "_seal",
            "_capability",
        ),
        stage_admission_type: (
            "use_index",
            "stage_use_sha256",
            "aliases",
            "content_sha256",
            "proof_authority",
            "_token",
            "_seal",
            "_capability",
        ),
        registry_stats_type: (
            "physical_builds",
            "dense_physical_builds",
            "conv_physical_builds",
            "stage_aliases",
            "cross_stage_physical_hits",
            "execution_alias_lookups",
            "admission_full_validations",
            "private_execution_full_validations",
            "commit_full_validations",
            "admitted_stages",
            "event_count",
            "event_chain_head_sha256",
            "state",
            "proof_authority",
        ),
        physical_registry_certificate_type: (
            "physical_builds",
            "dense_physical_builds",
            "conv_physical_builds",
            "stage_aliases",
            "cross_stage_physical_hits",
            "execution_alias_lookups",
            "admission_full_validations",
            "private_execution_full_validations",
            "commit_full_validations",
            "receipt",
            "proof_authority",
        ),
    }
    boundary_slot_readers: Dict[type, Tuple[Any, ...]] = {}
    for boundary_type, field_names in boundary_field_names.items():
        namespace = type.__getattribute__(boundary_type, "__dict__")
        readers = tuple(namespace.get(name) for name in field_names)
        if any(type(value) is not MemberDescriptorType for value in readers):
            _fail(
                "PUBLIC_ABI_SUBSTITUTION",
                "public boundary fields must use exact builtin slots",
            )
        boundary_slot_readers[boundary_type] = readers

    def boundary_values(value: Any, expected_type: type) -> Tuple[Any, ...]:
        if type(value) is not expected_type:
            dependency_fail(
                "HANDLE_SEAL_MISMATCH",
                "public boundary value has a substituted exact type",
            )
        return tuple(
            MemberDescriptorType.__get__(reader, value, expected_type)
            for reader in boundary_slot_readers[expected_type]
        )

    def boundary_mapping(value: Any, expected_type: type) -> Dict[str, Any]:
        return dict(
            zip(
                boundary_field_names[expected_type],
                boundary_values(value, expected_type),
            )
        )

    def make_boundary(expected_type: type, **values: Any) -> Any:
        field_names = boundary_field_names[expected_type]
        if set(values) != set(field_names):
            dependency_fail(
                "INTERNAL_FAILURE",
                "boundary constructor fields do not match exact slots",
            )
        result = object.__new__(expected_type)
        for name, reader in zip(
            field_names, boundary_slot_readers[expected_type]
        ):
            MemberDescriptorType.__set__(reader, result, values[name])
        return result

    def physical_boundary_body(value: Any) -> Mapping[str, Any]:
        fields = boundary_mapping(value, physical_core_handle_type)
        return {
            "schema": PHYSICAL_HANDLE_SCHEMA,
            "operator_branch": fields["operator_branch"],
            "layer_id": fields["layer_id"],
            "predecessor_id": fields["predecessor_id"],
            "physical_key_sha256": fields["physical_key_sha256"],
            "core_content_sha256": fields["core_content_sha256"],
            "proof_authority": fields["proof_authority"],
            "token": fields["_token"],
        }

    def alias_boundary_body(value: Any) -> Mapping[str, Any]:
        fields = boundary_mapping(value, stage_alias_handle_type)
        core_fields = boundary_mapping(
            fields["physical_core"], physical_core_handle_type
        )
        return {
            "schema": STAGE_ALIAS_SCHEMA,
            "use_index": fields["use_index"],
            "stage_use_sha256": fields["stage_use_sha256"],
            "layer_id": fields["layer_id"],
            "predecessor_id": fields["predecessor_id"],
            "physical_key_sha256": core_fields[
                "physical_key_sha256"
            ],
            "core_content_sha256": core_fields[
                "core_content_sha256"
            ],
            "stage_diagnostic_schema": fields[
                "stage_diagnostic_schema"
            ],
            "stage_diagnostic_sha256": fields[
                "stage_diagnostic_sha256"
            ],
            "alias_content_sha256": fields["alias_content_sha256"],
            "proof_authority": fields["proof_authority"],
            "token": fields["_token"],
        }

    def admission_boundary_body(value: Any) -> Mapping[str, Any]:
        fields = boundary_mapping(value, stage_admission_type)
        alias_content = []
        for alias in fields["aliases"]:
            alias_fields = boundary_mapping(
                alias, stage_alias_handle_type
            )
            alias_content.append(alias_fields["alias_content_sha256"])
        return {
            "schema": STAGE_ADMISSION_SCHEMA,
            "use_index": fields["use_index"],
            "stage_use_sha256": fields["stage_use_sha256"],
            "alias_content_sha256": alias_content,
            "content_sha256": fields["content_sha256"],
            "proof_authority": fields["proof_authority"],
            "token": fields["_token"],
        }

    def fast_fields_equal(
        current: Tuple[Any, ...],
        expected: Tuple[Any, ...],
        *,
        identity_indices: Tuple[int, ...],
    ) -> bool:
        if len(current) != len(expected):
            return False
        for index, (left, right) in enumerate(zip(current, expected)):
            if index in identity_indices:
                if left is not right:
                    return False
            elif type(left) is not type(right) or left != right:
                return False
        return True
    required_functions = (
        dense_platform_check,
        dense_prepare,
        dense_validate,
        dense_array_sha256,
        conv_platform_check,
        conv_geometry,
        conv_prepare,
        conv_validate,
        frozen_layer_manifest,
        frozen_immutable_f64,
        derive_source_spec,
        dense_numeric_core_builder,
        validate_dense_numeric_core,
        validate_dense_support_fresh,
    )
    if any(type(value) is not FunctionType for value in required_functions):
        _fail(
            "DEPENDENCY_SUBSTITUTION",
            "a required isolated dependency is not an exact function",
        )
    dependency_error_types = (
        frozen_globals["QueryDualReplayError"],
        dense_globals["QueryDualScalarGuardV51Error"],
    )

    raw_root_manifest, initial_layer_manifests = _root_manifest(
        full_layers, layer_manifest=frozen_layer_manifest
    )
    initial_bounds_manifest = _frame_bounds_manifest(
        full_layers, frame_bounds
    )
    terminal_assert = _assert_id(full_layers)
    if terminal_assert != raw_assert_id:
        _fail(
            "RAW_CONTEXT_MISMATCH",
            "owned ASSERT differs from captured raw ASSERT",
        )
    initial_schedules: Dict[int, Tuple[int, ...]] = {}
    for stage_use in stage_uses:
        reverse_order = _validate_context(
            full_layers=full_layers,
            contexts=contexts,
            stage_use=stage_use,
            assert_id=terminal_assert,
        )
        affine = tuple(
            lid
            for lid in reverse_order
            if full_layers[lid].kind in _AFFINE_KINDS
        )
        if affine != captured_schedules[stage_use.use_index]:
            _fail(
                "RAW_CONTEXT_MISMATCH",
                "owned affine schedule differs from raw capture",
            )
        initial_schedules[stage_use.use_index] = affine

    owned_root_manifest, owned_layer_manifests = _root_manifest(
        full_layers, layer_manifest=frozen_layer_manifest
    )
    owned_bounds_manifest = _frame_bounds_manifest(
        full_layers, frame_bounds
    )
    if (
        owned_root_manifest != raw_root_manifest
        or dict(owned_layer_manifests) != dict(initial_layer_manifests)
        or owned_bounds_manifest != initial_bounds_manifest
    ):
        _fail(
            "RAW_CONTEXT_MISMATCH",
            "owned root/frame snapshot differs from raw anchors",
        )

    module_source = _source_sha256()
    dependency_manifest = _deep_freeze(
        {
            "schema": (
                "act.query_dual_v51b_dependency_implementation.v2"
            ),
            "module_source_sha256": {
                "query_dual_replay": _file_sha256(frozen.__file__),
                "query_dual_scalar_guard_v51": _file_sha256(
                    dense_v51.__file__
                ),
                "query_dual_replay_v51_conv": _file_sha256(
                    conv_v51.__file__
                ),
                "query_dual_v51_authority": _file_sha256(
                    authority.__file__
                ),
                "query_dual_v51b_physical_registry": module_source,
            },
            "module_function_implementation_sha256": {
                name: dict(value)
                for name, value in (
                    dependency_function_manifests.items()
                )
            },
            "module_alias_capture": _PRIVATE_MODULE_ALIAS_MANIFEST,
            "module_view_runtime_seal": {
                "persistent_class_namespace_substitution_rejected": True,
                "operation_entry_and_publication_fingerprint": True,
                "transient_change_dispatch_restore_cycle_closed": False,
                "formal_integration_requirement": (
                    "hidden_runtime_seal_with_callable_closure_locals_"
                    "and_zero_module_or_module_view_references"
                ),
                "proof_authority": False,
            },
            "callable": {
                "dense_platform_check": _callable_name(
                    dense_platform_check
                ),
                "dense_prepare": _callable_name(dense_prepare),
                "dense_validate": _callable_name(dense_validate),
                "dense_array_sha256": _callable_name(
                    dense_array_sha256
                ),
                "conv_platform_check": _callable_name(
                    conv_platform_check
                ),
                "conv_geometry": _callable_name(conv_geometry),
                "conv_prepare": _callable_name(conv_prepare),
                "conv_validate": _callable_name(conv_validate),
                "frozen_layer_manifest": _callable_name(
                    frozen_layer_manifest
                ),
                "frozen_immutable_f64": _callable_name(
                    frozen_immutable_f64
                ),
                "derive_source_spec": _callable_name(
                    derive_source_spec
                ),
                "dense_numeric_core_builder": _callable_name(
                    dense_numeric_core_builder
                ),
                "validate_dense_numeric_core": _callable_name(
                    validate_dense_numeric_core
                ),
                "validate_dense_support_fresh": _callable_name(
                    validate_dense_support_fresh
                ),
            },
            "type": {
                "dense_support": _callable_name(dense_support_type),
                "dense_diagnostics": _callable_name(
                    dense_diagnostics_type
                ),
                "conv_plan": _callable_name(conv_plan_type),
            },
        }
    )
    dependency_implementation = _json_sha256(dependency_manifest)
    dependency_binding_gate()
    platform_sha = dense_platform_check().sha256
    numeric_platform = _json_sha256(
        {
            "dense_v51_platform_sha256": platform_sha,
            "conv_v51_platform": dict(conv_platform_check()),
        }
    )
    dependency_binding_gate()
    owner_pid = _OS_GETPID()
    secret = _SECRETS_TOKEN_BYTES(32)
    capability = object()
    creation_capability = object()
    operation_lock = _THREADING_LOCK()
    state: Dict[str, Any] = {"phase": "OPEN"}
    source_anchors: Dict[Tuple[int, int], _SourceSpec] = {}
    cores_by_key: Dict[str, _CoreRecord] = {}
    aliases_by_occurrence: Dict[Tuple[int, int], _AliasRecord] = {}
    admissions_by_index: Dict[int, _AdmissionRecord] = {}
    admissions_by_identity: Dict[int, _AdmissionRecord] = {}
    events = []
    chain_head = _ZERO_SHA256
    execution_lookup_count = 0
    port_ref: Optional[weakref.ReferenceType[Any]] = None

    def check_pid() -> None:
        if _OS_GETPID() != owner_pid:
            _fail(
                "PROCESS_MISMATCH",
                "frame-local registry capability cannot cross a fork",
            )

    def clear_material() -> None:
        for record in cores_by_key.values():
            record.material = None
            record.numeric_core = None
        cores_by_key.clear()
        source_anchors.clear()

    def poison() -> None:
        if state["phase"] in ("OPEN", "COMMITTED"):
            state["phase"] = "POISONED"

    def check_deadline() -> None:
        if (
            checked_deadline is not None
            and _TIME_MONOTONIC() >= checked_deadline
        ):
            poison()
            raise V51BPhysicalRegistryTimeout()

    class _FactoryDependencyDeadline:
        __slots__ = ()

        def check(self, *, force: bool = False) -> None:
            del force
            check_deadline()

    dependency_deadline = _FactoryDependencyDeadline()

    def require_open() -> None:
        if state["phase"] != "OPEN":
            _fail(
                "INVALID_STATE",
                f"registry state is {state['phase']}, not OPEN",
            )

    def verify_raw_identity() -> None:
        try:
            _validate_exact_external_raw(
                full_layers=raw_full_layers,
                contexts=raw_contexts,
                stage_uses=raw_stage_uses,
                frame_bounds=raw_frame_bounds,
                stage_use_type=stage_use_type,
            )
        except V51BPhysicalRegistryError as exc:
            if exc.code in (
                "INVALID_STAGE_USE",
                "RAW_EXACT_TYPE",
                "RAW_CONTEXT_MISMATCH",
            ):
                _fail(
                    "RAW_CONTEXT_MISMATCH",
                    "raw root/frame exact structure changed after open",
                )
            raise
        if (
            _raw_identity_snapshot(
                full_layers=raw_full_layers,
                contexts=raw_contexts,
                stage_uses=raw_stage_uses,
                frame_bounds=raw_frame_bounds,
            )
            != initial_raw_identity
        ):
            _fail(
                "RAW_CONTEXT_MISMATCH",
                "raw root/frame identity changed after registry open",
            )

    class _OperationGuard:
        __slots__ = ()

        def __enter__(self) -> None:
            check_pid()
            if not operation_lock.acquire(blocking=False):
                _fail(
                    "CONCURRENT_ACCESS",
                    "concurrent registry operation rejected",
                )

        def __exit__(
            self,
            exception_type: Any,
            exception: Any,
            traceback: Any,
        ) -> bool:
            del exception_type, exception, traceback
            operation_lock.release()
            return False

    operation_guard = _OperationGuard()

    def append_event(operation: str, payload: Mapping[str, Any]) -> None:
        nonlocal chain_head
        sequence = len(events)
        frozen_payload = _deep_freeze(payload)
        body = {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "operation": operation,
            "payload": frozen_payload,
            "previous_mac": chain_head,
        }
        mac = _sealed_hmac(secret, "event", body)
        events.append(
            _Event(
                sequence=sequence,
                operation=operation,
                payload=frozen_payload,
                previous_mac=chain_head,
                mac=mac,
            )
        )
        chain_head = mac

    def scan_events() -> Mapping[str, int]:
        previous = _ZERO_SHA256
        counts = {
            "physical_builds": 0,
            "dense_physical_builds": 0,
            "conv_physical_builds": 0,
            "stage_aliases": 0,
            "cross_stage_physical_hits": 0,
            "execution_alias_lookups": execution_lookup_count,
            "admission_full_validations": 0,
            "private_execution_full_validations": 0,
            "commit_full_validations": 0,
            "admitted_stages": 0,
        }
        for index, event in enumerate(events):
            body = {
                "schema": EVENT_SCHEMA,
                "sequence": event.sequence,
                "operation": event.operation,
                "payload": event.payload,
                "previous_mac": event.previous_mac,
            }
            wanted = _sealed_hmac(secret, "event", body)
            if (
                event.sequence != index
                or event.previous_mac != previous
                or not _HMAC_COMPARE_DIGEST(event.mac, wanted)
            ):
                _fail("EVENT_CHAIN_MISMATCH", "event chain authentication failed")
            previous = event.mac
            if event.operation == "CORE_BUILD":
                counts["physical_builds"] += 1
                if event.payload["operator_branch"] == BRANCH_DENSE:
                    counts["dense_physical_builds"] += 1
                else:
                    counts["conv_physical_builds"] += 1
            elif event.operation == "CORE_HIT":
                counts["cross_stage_physical_hits"] += 1
            elif event.operation == "ALIAS_MINT":
                counts["stage_aliases"] += 1
            elif event.operation == "ADMISSION_VALIDATE":
                counts["admission_full_validations"] += 1
            elif event.operation == "PRIVATE_EXECUTION_VALIDATE":
                counts["private_execution_full_validations"] += 1
            elif event.operation == "COMMIT_VALIDATE":
                counts["commit_full_validations"] += 1
            elif event.operation == "STAGE_ADMIT":
                counts["admitted_stages"] += 1
        if previous != chain_head:
            _fail("EVENT_CHAIN_MISMATCH", "event chain head changed")
        return MappingProxyType(counts)

    def current_stats() -> RegistryStats:
        counts = scan_events()
        return make_boundary(
            registry_stats_type,
            physical_builds=counts["physical_builds"],
            dense_physical_builds=counts["dense_physical_builds"],
            conv_physical_builds=counts["conv_physical_builds"],
            stage_aliases=counts["stage_aliases"],
            cross_stage_physical_hits=counts[
                "cross_stage_physical_hits"
            ],
            execution_alias_lookups=counts[
                "execution_alias_lookups"
            ],
            admission_full_validations=counts[
                "admission_full_validations"
            ],
            private_execution_full_validations=counts[
                "private_execution_full_validations"
            ],
            commit_full_validations=counts["commit_full_validations"],
            admitted_stages=counts["admitted_stages"],
            event_count=len(events),
            event_chain_head_sha256=chain_head,
            state=state["phase"],
            proof_authority=False,
        )

    def validate_physical_handle(record: _CoreRecord) -> None:
        value = record.handle
        current_fields = boundary_values(
            value, physical_core_handle_type
        )
        current = physical_boundary_body(value)
        wanted = _sealed_hmac(secret, "physical-handle", current)
        if (
            not fast_fields_equal(
                current_fields,
                record.handle_fast_fields,
                identity_indices=(8,),
            )
            or _canonical(current) != _canonical(record.handle_body)
            or not _HMAC_COMPARE_DIGEST(current_fields[7], wanted)
            or not _HMAC_COMPARE_DIGEST(
                current_fields[7], record.handle_seal
            )
        ):
            _fail(
                "HANDLE_SEAL_MISMATCH",
                "physical handle coordinated replacement detected",
            )

    def validate_alias_handle(record: _AliasRecord) -> None:
        value = record.handle
        current_fields = boundary_values(value, stage_alias_handle_type)
        core = cores_by_key.get(record.physical_key_sha256)
        if core is None or current_fields[4] is not core.handle:
            _fail("HANDLE_SEAL_MISMATCH", "alias physical handle changed")
        validate_physical_handle(core)
        current = alias_boundary_body(value)
        wanted = _sealed_hmac(secret, "stage-alias", current)
        if (
            not fast_fields_equal(
                current_fields,
                record.handle_fast_fields,
                identity_indices=(4, 11),
            )
            or _canonical(current) != _canonical(record.handle_body)
            or not _HMAC_COMPARE_DIGEST(current_fields[10], wanted)
            or not _HMAC_COMPARE_DIGEST(
                current_fields[10], record.handle_seal
            )
        ):
            _fail(
                "HANDLE_SEAL_MISMATCH",
                "stage alias coordinated replacement detected",
            )

    def validate_admission_handle(record: _AdmissionRecord) -> None:
        value = record.handle
        current_fields = boundary_values(value, stage_admission_type)
        if (
            admissions_by_index.get(record.use_index) is not record
            or not fast_fields_equal(
                current_fields,
                record.fast_fields,
                identity_indices=(2, 7),
            )
        ):
            _fail("HANDLE_SEAL_MISMATCH", "admission identity changed")
        if any(
            current_fields[2][index] is not alias_record.handle
            for index, alias_record in enumerate(
                record.alias_by_layer.values()
            )
        ):
            _fail("HANDLE_SEAL_MISMATCH", "admission alias sequence changed")
        for alias_record in record.alias_by_layer.values():
            validate_alias_handle(alias_record)
        current = admission_boundary_body(value)
        wanted = _sealed_hmac(secret, "stage-admission", current)
        if (
            _canonical(current) != _canonical(record.handle_body)
            or not _HMAC_COMPARE_DIGEST(current_fields[6], wanted)
            or not _HMAC_COMPARE_DIGEST(
                current_fields[6], record.handle_seal
            )
        ):
            _fail(
                "HANDLE_SEAL_MISMATCH",
                "stage admission coordinated replacement detected",
            )

    def validate_admission_handle_fast(record: _AdmissionRecord) -> None:
        value = record.handle
        current = boundary_values(value, stage_admission_type)
        if (
            admissions_by_index.get(record.use_index) is not record
            or not fast_fields_equal(
                current,
                record.fast_fields,
                identity_indices=(2, 7),
            )
        ):
            _fail(
                "HANDLE_SEAL_MISMATCH",
                "stage admission fast seal changed",
            )

    def validate_alias_handle_fast(record: _AliasRecord) -> None:
        value = record.handle
        current = boundary_values(value, stage_alias_handle_type)
        core = cores_by_key.get(record.physical_key_sha256)
        if (
            core is None
            or current[4] is not core.handle
            or not fast_fields_equal(
                current,
                record.handle_fast_fields,
                identity_indices=(4, 11),
            )
        ):
            _fail(
                "HANDLE_SEAL_MISMATCH",
                "stage alias fast seal changed",
            )
        core_fields = boundary_values(
            core.handle, physical_core_handle_type
        )
        if (
            not fast_fields_equal(
                core_fields,
                core.handle_fast_fields,
                identity_indices=(8,),
            )
        ):
            _fail(
                "HANDLE_SEAL_MISMATCH",
                "physical handle fast seal changed",
            )

    def source_anchor_hit(anchor: _SourceSpec, layer_id: int) -> bool:
        layer = full_layers.get(layer_id)
        if (
            layer is not anchor.layer
            or layer.params is not anchor.layer_params_identity
            or layer.kind not in _AFFINE_KINDS
            or tuple(layer.preds) != (anchor.locator[1],)
        ):
            return False
        box = frame_bounds.get(anchor.locator[1])
        return bool(
            box is anchor.raw_box
            and box.lb is anchor.raw_lb_identity
            and box.ub is anchor.raw_ub_identity
        )

    def fresh_spec(layer_id: int) -> _SourceSpec:
        dependency_binding_gate()
        result = derive_source_spec(
            full_layers=full_layers,
            frame_bounds=frame_bounds,
            layer_id=layer_id,
            root_content_sha256=root_content,
            raw_root_manifest_sha256=raw_root_manifest,
            frame_content_sha256=frame_content,
            bounds_manifest_sha256=initial_bounds_manifest,
            numeric_contract_sha256=numeric_contract,
            numeric_platform_sha256=numeric_platform,
            implementation_sha256=implementation,
            dependency_implementation_sha256=(
                dependency_implementation
            ),
            module_source_sha256=module_source,
            conv_geometry=conv_geometry,
            frozen_layer_type=frozen_layer_type,
            frozen_box_type=frozen_box_type,
            immutable_f64=frozen_immutable_f64,
        )
        dependency_binding_gate()
        return result

    def validate_raw_anchors() -> None:
        dependency_binding_gate()
        verify_raw_identity()
        fresh_root_manifest, fresh_layer_manifests = _root_manifest(
            raw_full_layers, layer_manifest=frozen_layer_manifest
        )
        fresh_bounds_manifest = _frame_bounds_manifest(
            raw_full_layers, raw_frame_bounds
        )
        if (
            not _HMAC_COMPARE_DIGEST(
                fresh_root_manifest, raw_root_manifest
            )
            or dict(fresh_layer_manifests)
            != dict(initial_layer_manifests)
            or not _HMAC_COMPARE_DIGEST(
                fresh_bounds_manifest, initial_bounds_manifest
            )
            or _assert_id(raw_full_layers) != terminal_assert
            or not _HMAC_COMPARE_DIGEST(_source_sha256(), module_source)
        ):
            _fail(
                "RAW_CONTEXT_MISMATCH",
                "raw root or frame bounds changed before publication",
            )
        for raw_stage_use in raw_stage_uses:
            raw_reverse_order = _validate_context(
                full_layers=raw_full_layers,
                contexts=raw_contexts,
                stage_use=raw_stage_use,
                assert_id=terminal_assert,
            )
            raw_schedule = tuple(
                lid
                for lid in raw_reverse_order
                if raw_full_layers[lid].kind in _AFFINE_KINDS
            )
            if (
                raw_schedule
                != initial_schedules[raw_stage_use.use_index]
            ):
                _fail(
                    "RAW_CONTEXT_MISMATCH",
                    "raw affine schedule changed before publication",
                )
        dependency_binding_gate()

    def mint_physical_handle(
        spec: _SourceSpec, core_content_sha256: str
    ) -> Tuple[
        PhysicalCoreHandle, Mapping[str, Any], str, Tuple[Any, ...]
    ]:
        token = _SECRETS_TOKEN_HEX(24)
        provisional = make_boundary(
            physical_core_handle_type,
            operator_branch=spec.operator_branch,
            layer_id=spec.locator[0],
            predecessor_id=spec.locator[1],
            physical_key_sha256=spec.physical_key_sha256,
            core_content_sha256=core_content_sha256,
            proof_authority=False,
            _token=token,
            _seal=_ZERO_SHA256,
            _capability=capability,
        )
        body = physical_boundary_body(provisional)
        seal = _sealed_hmac(secret, "physical-handle", body)
        value = make_boundary(
            physical_core_handle_type,
            operator_branch=spec.operator_branch,
            layer_id=spec.locator[0],
            predecessor_id=spec.locator[1],
            physical_key_sha256=spec.physical_key_sha256,
            core_content_sha256=core_content_sha256,
            proof_authority=False,
            _token=token,
            _seal=seal,
            _capability=capability,
        )
        final_body = _deep_freeze(physical_boundary_body(value))
        return (
            value,
            final_body,
            seal,
            boundary_values(value, physical_core_handle_type),
        )

    def build_core(spec: _SourceSpec) -> _CoreRecord:
        check_deadline()
        dependency_binding_gate()
        if spec.operator_branch == BRANCH_DENSE:
            weight = np.asarray(spec.layer.params["weight"])
            support = dense_prepare(
                weight,
                spec.max_abs,
                binding={
                    "frame_content_sha256": frame_content,
                    "physical_key_sha256": spec.physical_key_sha256,
                    "root_content_sha256": root_content,
                },
                deadline=checked_deadline,
            )
            dependency_binding_gate()
            dense_validate(
                support, weight, platform_sha256=platform_sha
            )
            dependency_binding_gate()
            numeric = dense_numeric_core_builder(
                support, immutable_f64=frozen_immutable_f64
            )
            core_content = _json_sha256(
                {
                    "physical_key_sha256": spec.physical_key_sha256,
                    "operator_branch": BRANCH_DENSE,
                    "dense_numeric_content_sha256": numeric.content_sha256,
                    "proof_authority": False,
                }
            )
            material = support
        else:
            plan = conv_prepare(
                spec.layer,
                spec.effective_box,
                deadline=dependency_deadline,
            )
            dependency_binding_gate()
            conv_validate(plan, deadline=dependency_deadline)
            dependency_binding_gate()
            numeric = None
            core_content = _json_sha256(
                {
                    "physical_key_sha256": spec.physical_key_sha256,
                    "operator_branch": BRANCH_CONV_DENSE,
                    "conv_plan_content_sha256": plan.manifest[
                        "content_sha256"
                    ],
                    "proof_authority": False,
                }
            )
            material = plan
        dependency_binding_gate()
        check_deadline()
        (
            handle,
            handle_body,
            handle_seal,
            handle_fast_fields,
        ) = mint_physical_handle(
            spec, core_content
        )
        record = _CoreRecord(
            spec=spec,
            material=material,
            numeric_core=numeric,
            core_content_sha256=core_content,
            handle=handle,
            handle_body=handle_body,
            handle_seal=handle_seal,
            handle_fast_fields=handle_fast_fields,
        )
        append_event(
            "CORE_BUILD",
            {
                "physical_key_sha256": spec.physical_key_sha256,
                "operator_branch": spec.operator_branch,
                "layer_id": spec.locator[0],
            },
        )
        append_event(
            "ADMISSION_VALIDATE",
            {
                "physical_key_sha256": spec.physical_key_sha256,
                "operator_branch": spec.operator_branch,
            },
        )
        return record

    def alias_semantic_body(
        *,
        stage_use: authority.StageUse,
        spec: _SourceSpec,
        core_content_sha256: str,
        base_diagnostic_sha256: str,
    ) -> Mapping[str, Any]:
        diagnostic_schema = (
            DENSE_STAGE_DIAGNOSTIC_SCHEMA
            if spec.operator_branch == BRANCH_DENSE
            else CONV_STAGE_DIAGNOSTIC_SCHEMA
        )
        diagnostic = _json_sha256(
            {
                "schema": diagnostic_schema,
                "stage_use_sha256": stage_use.stage_use_sha256,
                "use_index": stage_use.use_index,
                "layer_id": spec.locator[0],
                "predecessor_id": spec.locator[1],
                "physical_key_sha256": spec.physical_key_sha256,
                "core_content_sha256": core_content_sha256,
                "base_material_diagnostic_sha256": base_diagnostic_sha256,
                "proof_authority": False,
            }
        )
        body = {
            "schema": STAGE_ALIAS_SCHEMA,
            "stage_use": _stage_use_body(stage_use),
            "root_content_sha256": root_content,
            "raw_root_manifest_sha256": raw_root_manifest,
            "frame_content_sha256": frame_content,
            "bounds_manifest_sha256": initial_bounds_manifest,
            "layer_id": spec.locator[0],
            "predecessor_id": spec.locator[1],
            "operator_branch": spec.operator_branch,
            "physical_key_sha256": spec.physical_key_sha256,
            "core_content_sha256": core_content_sha256,
            "stage_diagnostic_sha256": diagnostic,
            "proof_authority": False,
        }
        frozen_body = _deep_freeze(body)
        return MappingProxyType(
            {
                "body": frozen_body,
                "content_sha256": _json_sha256(body),
                "diagnostic_schema": diagnostic_schema,
                "stage_diagnostic_sha256": diagnostic,
            }
        )

    def mint_alias(
        *,
        stage_use: authority.StageUse,
        spec: _SourceSpec,
        core: _CoreRecord,
    ) -> _AliasRecord:
        if spec.operator_branch == BRANCH_DENSE:
            base_diagnostic = core.material.diagnostics.sha256
        else:
            base_diagnostic = str(
                core.material.manifest["content_sha256"]
            )
        semantic = alias_semantic_body(
            stage_use=stage_use,
            spec=spec,
            core_content_sha256=core.core_content_sha256,
            base_diagnostic_sha256=base_diagnostic,
        )
        token = _SECRETS_TOKEN_HEX(24)
        provisional = make_boundary(
            stage_alias_handle_type,
            use_index=stage_use.use_index,
            stage_use_sha256=stage_use.stage_use_sha256,
            layer_id=spec.locator[0],
            predecessor_id=spec.locator[1],
            physical_core=core.handle,
            stage_diagnostic_schema=semantic["diagnostic_schema"],
            stage_diagnostic_sha256=semantic[
                "stage_diagnostic_sha256"
            ],
            alias_content_sha256=semantic["content_sha256"],
            proof_authority=False,
            _token=token,
            _seal=_ZERO_SHA256,
            _capability=capability,
        )
        body = alias_boundary_body(provisional)
        seal = _sealed_hmac(secret, "stage-alias", body)
        handle = make_boundary(
            stage_alias_handle_type,
            use_index=stage_use.use_index,
            stage_use_sha256=stage_use.stage_use_sha256,
            layer_id=spec.locator[0],
            predecessor_id=spec.locator[1],
            physical_core=core.handle,
            stage_diagnostic_schema=semantic["diagnostic_schema"],
            stage_diagnostic_sha256=semantic[
                "stage_diagnostic_sha256"
            ],
            alias_content_sha256=semantic["content_sha256"],
            proof_authority=False,
            _token=token,
            _seal=seal,
            _capability=capability,
        )
        record = _AliasRecord(
            use_index=stage_use.use_index,
            layer_id=spec.locator[0],
            predecessor_id=spec.locator[1],
            physical_key_sha256=spec.physical_key_sha256,
            alias_body=semantic["body"],
            handle=handle,
            handle_body=_deep_freeze(alias_boundary_body(handle)),
            handle_seal=seal,
            handle_fast_fields=boundary_values(
                handle, stage_alias_handle_type
            ),
        )
        append_event(
            "ALIAS_MINT",
            {
                "use_index": stage_use.use_index,
                "layer_id": spec.locator[0],
                "physical_key_sha256": spec.physical_key_sha256,
                "alias_content_sha256": semantic["content_sha256"],
            },
        )
        return record

    def admit_impl(stage_use: authority.StageUse) -> StageAdmission:
        require_open()
        check_deadline()
        _builtins_binding_gate()
        _factory_public_dependency_gate()
        verify_raw_identity()
        dependency_binding_gate()
        expected_index = len(admissions_by_index)
        if (
            type(stage_use) is not stage_use_type
            or expected_index >= len(raw_stage_uses)
            or stage_use is not raw_stage_uses[expected_index]
        ):
            _fail(
                "STAGE_SEAL_MISMATCH",
                "stage must be the next exact root-registered StageUse",
            )
        stage_use = stage_uses[expected_index]
        reverse_order = _validate_context(
            full_layers=full_layers,
            contexts=contexts,
            stage_use=stage_use,
            assert_id=terminal_assert,
        )
        schedule = tuple(
            lid
            for lid in reverse_order
            if full_layers[lid].kind in _AFFINE_KINDS
        )
        if schedule != initial_schedules[stage_use.use_index]:
            _fail("RAW_CONTEXT_MISMATCH", "stage affine schedule changed")

        stage_alias_records: Dict[int, _AliasRecord] = {}
        for layer_id in schedule:
            check_deadline()
            layer = full_layers[layer_id]
            locator = (layer_id, int(layer.preds[0]))
            anchor = source_anchors.get(locator)
            if anchor is None:
                spec = fresh_spec(layer_id)
                if spec.locator != locator:
                    _fail("RAW_CONTEXT_MISMATCH", "affine locator changed")
                source_anchors[locator] = spec
                core = cores_by_key.get(spec.physical_key_sha256)
                if core is not None:
                    _fail(
                        "PHYSICAL_KEY_COLLISION",
                        "distinct source anchors produced one key",
                    )
                core = build_core(spec)
                cores_by_key[spec.physical_key_sha256] = core
            else:
                if not source_anchor_hit(anchor, layer_id):
                    _fail(
                        "RAW_CONTEXT_MISMATCH",
                        "raw layer/box identity changed on physical hit",
                    )
                spec = anchor
                core = cores_by_key.get(spec.physical_key_sha256)
                if core is None:
                    _fail(
                        "PHYSICAL_REGISTRY_MISMATCH",
                        "source anchor lost its physical core",
                    )
                validate_physical_handle(core)
                append_event(
                    "CORE_HIT",
                    {
                        "use_index": stage_use.use_index,
                        "layer_id": layer_id,
                        "physical_key_sha256": spec.physical_key_sha256,
                    },
                )
            occurrence = (stage_use.use_index, layer_id)
            if occurrence in aliases_by_occurrence:
                _fail("DUPLICATE_ALIAS", "stage affine alias already exists")
            alias_record = mint_alias(
                stage_use=stage_use, spec=spec, core=core
            )
            aliases_by_occurrence[occurrence] = alias_record
            stage_alias_records[layer_id] = alias_record

        alias_tuple = tuple(
            stage_alias_records[layer_id].handle for layer_id in schedule
        )
        content_body = {
            "schema": STAGE_ADMISSION_SCHEMA,
            "stage_use": _stage_use_body(stage_use),
            "affine_layer_ids": list(schedule),
            "alias_content_sha256": [
                stage_alias_records[layer_id].handle_fast_fields[7]
                for layer_id in schedule
            ],
            "proof_authority": False,
        }
        content = _json_sha256(content_body)
        token = _SECRETS_TOKEN_HEX(24)
        provisional = make_boundary(
            stage_admission_type,
            use_index=stage_use.use_index,
            stage_use_sha256=stage_use.stage_use_sha256,
            aliases=alias_tuple,
            content_sha256=content,
            proof_authority=False,
            _token=token,
            _seal=_ZERO_SHA256,
            _capability=capability,
        )
        body = admission_boundary_body(provisional)
        seal = _sealed_hmac(secret, "stage-admission", body)
        handle = make_boundary(
            stage_admission_type,
            use_index=stage_use.use_index,
            stage_use_sha256=stage_use.stage_use_sha256,
            aliases=alias_tuple,
            content_sha256=content,
            proof_authority=False,
            _token=token,
            _seal=seal,
            _capability=capability,
        )
        record = _AdmissionRecord(
            use_index=stage_use.use_index,
            alias_by_layer=MappingProxyType(stage_alias_records),
            handle=handle,
            handle_body=_deep_freeze(admission_boundary_body(handle)),
            handle_seal=seal,
            fast_fields=boundary_values(
                handle, stage_admission_type
            ),
        )
        admissions_by_index[stage_use.use_index] = record
        admissions_by_identity[id(handle)] = record
        append_event(
            "STAGE_ADMIT",
            {
                "use_index": stage_use.use_index,
                "stage_use_sha256": stage_use.stage_use_sha256,
                "affine_count": len(schedule),
                "content_sha256": content,
            },
        )
        verify_raw_identity()
        dependency_binding_gate()
        check_deadline()
        return handle

    def lookup_impl(
        admission: StageAdmission, layer_id: int
    ) -> StageAliasHandle:
        nonlocal execution_lookup_count
        require_open()
        check_deadline()
        if (
            type(layer_id) is not int
            or type(admission) is not stage_admission_type
        ):
            _fail("INVALID_LOOKUP", "lookup requires an exact affine layer id")
        record = admissions_by_identity.get(id(admission))
        if record is None or admission is not record.handle:
            _fail("HANDLE_SEAL_MISMATCH", "foreign admission handle")
        validate_admission_handle_fast(record)
        alias_record = record.alias_by_layer.get(layer_id)
        if alias_record is None:
            _fail("INVALID_LOOKUP", "layer is not affine in this stage")
        validate_alias_handle_fast(alias_record)
        execution_lookup_count += 1
        check_deadline()
        return alias_record.handle

    def validate_core_at_commit(
        record: _CoreRecord, spec: _SourceSpec
    ) -> None:
        check_deadline()
        dependency_binding_gate()
        if spec.operator_branch == BRANCH_DENSE:
            if (
                type(record.material) is not dense_support_type
                or record.numeric_core is None
            ):
                _fail("MATERIAL_SUBSTITUTION", "Dense material disappeared")
            weight = np.asarray(spec.layer.params["weight"])
            dense_validate(
                record.material,
                weight,
                platform_sha256=platform_sha,
            )
            dependency_binding_gate()
            if not _HMAC_COMPARE_DIGEST(
                record.material.max_abs_sha256,
                dense_array_sha256(spec.max_abs),
            ):
                _fail(
                    "MATERIAL_SUBSTITUTION",
                    "Dense material no longer binds the raw frame box",
                )
            fresh_material = dense_prepare(
                weight,
                spec.max_abs,
                binding={
                    "frame_content_sha256": frame_content,
                    "physical_key_sha256": spec.physical_key_sha256,
                    "root_content_sha256": root_content,
                },
                deadline=checked_deadline,
            )
            dependency_binding_gate()
            dense_validate(
                fresh_material,
                weight,
                platform_sha256=platform_sha,
            )
            dependency_binding_gate()
            validate_dense_support_fresh(
                record.material,
                fresh_material,
                support_type=dense_support_type,
                diagnostics_type=dense_diagnostics_type,
            )
            dependency_binding_gate()
            validate_dense_numeric_core(
                record.numeric_core,
                fresh_material,
                immutable_f64=frozen_immutable_f64,
            )
            dependency_binding_gate()
            expected_content = _json_sha256(
                {
                    "physical_key_sha256": spec.physical_key_sha256,
                    "operator_branch": BRANCH_DENSE,
                    "dense_numeric_content_sha256": (
                        record.numeric_core.content_sha256
                    ),
                    "proof_authority": False,
                }
            )
        else:
            if type(record.material) is not conv_plan_type:
                _fail("MATERIAL_SUBSTITUTION", "Conv material disappeared")
            conv_validate(record.material, deadline=dependency_deadline)
            dependency_binding_gate()
            geometry = conv_geometry(spec.layer)
            dependency_binding_gate()
            plan = record.material
            if (
                plan.layer_id != spec.layer.id
                or not np.array_equal(
                    plan.weight, spec.layer.params["weight"]
                )
                or not np.array_equal(plan.support, spec.max_abs)
                or plan.input_shape != geometry["input_shape"]
                or plan.output_shape != geometry["output_shape"]
                or plan.stride != geometry["stride"]
                or plan.padding != geometry["padding"]
                or plan.dilation != geometry["dilation"]
                or plan.groups != geometry["groups"]
            ):
                _fail(
                    "MATERIAL_SUBSTITUTION",
                    "Conv material no longer binds the raw layer/box",
                )
            expected_content = _json_sha256(
                {
                    "physical_key_sha256": spec.physical_key_sha256,
                    "operator_branch": BRANCH_CONV_DENSE,
                    "conv_plan_content_sha256": plan.manifest[
                        "content_sha256"
                    ],
                    "proof_authority": False,
                }
            )
        if not _HMAC_COMPARE_DIGEST(
            record.core_content_sha256, expected_content
        ):
            _fail("MATERIAL_SUBSTITUTION", "physical core content changed")
        validate_physical_handle(record)
        dependency_binding_gate()
        check_deadline()

    def commit_impl() -> PhysicalRegistryCertificate:
        require_open()
        check_deadline()
        dependency_binding_gate()
        validate_raw_anchors()
        if len(admissions_by_index) != len(stage_uses):
            _fail("INCOMPLETE_REGISTRY", "not every stage was admitted")

        # Expected cores are reconstructed from the closure-owned frozen
        # snapshot.  The caller-owned raw anchors were independently walked
        # immediately above and are walked once more before publication.
        stage_schedules: Dict[int, Tuple[int, ...]] = {}
        expected_occurrences: Dict[
            Tuple[int, int], Tuple[int, int]
        ] = {}
        unique_locators: Dict[Tuple[int, int], int] = {}
        for stage_use in stage_uses:
            reverse_order = _validate_context(
                full_layers=full_layers,
                contexts=contexts,
                stage_use=stage_use,
                assert_id=terminal_assert,
            )
            schedule = tuple(
                lid
                for lid in reverse_order
                if full_layers[lid].kind in _AFFINE_KINDS
            )
            if schedule != initial_schedules[stage_use.use_index]:
                _fail(
                    "RAW_CONTEXT_MISMATCH",
                    "commit affine enumeration changed",
                )
            admission_record = admissions_by_index.get(stage_use.use_index)
            if admission_record is None:
                _fail("INCOMPLETE_REGISTRY", "stage admission disappeared")
            if tuple(admission_record.alias_by_layer) != schedule:
                _fail(
                    "PHYSICAL_REGISTRY_MISMATCH",
                    "internal alias schedule differs from raw root",
                )
            stage_schedules[stage_use.use_index] = schedule
            for layer_id in schedule:
                occurrence = (stage_use.use_index, layer_id)
                layer = full_layers[layer_id]
                if (
                    type(layer) is not frozen_layer_type
                    or len(layer.preds) != 1
                ):
                    _fail(
                        "RAW_CONTEXT_MISMATCH",
                        "commit affine locator changed",
                    )
                locator = (layer_id, layer.preds[0])
                expected_occurrences[occurrence] = locator
                unique_locators[locator] = layer_id

        # Physical derivation is exactly once per unique raw locator.  The
        # twenty stage occurrences below only map to these seven fresh specs.
        specs_by_locator: Dict[Tuple[int, int], _SourceSpec] = {}
        expected_specs: Dict[str, _SourceSpec] = {}
        commit_physical_derivations = 0
        for locator in sorted(unique_locators):
            spec = fresh_spec(unique_locators[locator])
            commit_physical_derivations += 1
            if spec.locator != locator:
                _fail(
                    "RAW_CONTEXT_MISMATCH",
                    "fresh physical locator changed",
                )
            specs_by_locator[locator] = spec
            previous = expected_specs.get(spec.physical_key_sha256)
            if previous is None:
                expected_specs[spec.physical_key_sha256] = spec
            elif previous.physical_key_body != spec.physical_key_body:
                _fail(
                    "PHYSICAL_KEY_COLLISION",
                    "commit found a physical key collision",
                )

        # Validate each unique core exactly once before alias diagnostics are
        # consumed.  Dense validation includes an independent fresh rebuild,
        # so a self-consistent forged diagnostic cannot be masked by the
        # subsequent alias comparison.
        validated_physical_keys = tuple(sorted(expected_specs))
        for physical_key in validated_physical_keys:
            record = cores_by_key.get(physical_key)
            if record is None:
                _fail(
                    "PHYSICAL_REGISTRY_MISMATCH",
                    "physical core vanished",
                )
            validate_core_at_commit(record, expected_specs[physical_key])

        for stage_use in stage_uses:
            for layer_id in stage_schedules[stage_use.use_index]:
                occurrence = (stage_use.use_index, layer_id)
                spec = specs_by_locator[expected_occurrences[occurrence]]
                alias_record = aliases_by_occurrence.get(occurrence)
                if (
                    alias_record is None
                    or alias_record.layer_id != layer_id
                    or alias_record.predecessor_id != spec.locator[1]
                    or not _HMAC_COMPARE_DIGEST(
                        alias_record.physical_key_sha256,
                        spec.physical_key_sha256,
                    )
                ):
                    _fail(
                        "PHYSICAL_REGISTRY_MISMATCH",
                        "stage alias does not match raw reconstruction",
                    )
                core_record = cores_by_key.get(spec.physical_key_sha256)
                if core_record is None:
                    _fail(
                        "PHYSICAL_REGISTRY_MISMATCH",
                        "stage alias references no reconstructed core",
                )
                if spec.operator_branch == BRANCH_DENSE:
                    if type(core_record.material) is not dense_support_type:
                        _fail(
                            "MATERIAL_SUBSTITUTION",
                            "Dense material disappeared before alias audit",
                        )
                    base_diagnostic = (
                        core_record.material.diagnostics.sha256
                    )
                else:
                    if type(core_record.material) is not conv_plan_type:
                        _fail(
                            "MATERIAL_SUBSTITUTION",
                            "Conv material disappeared before alias audit",
                        )
                    base_diagnostic = str(
                        core_record.material.manifest[
                            "content_sha256"
                        ]
                    )
                expected_alias = alias_semantic_body(
                    stage_use=stage_use,
                    spec=spec,
                    core_content_sha256=(
                        core_record.core_content_sha256
                    ),
                    base_diagnostic_sha256=base_diagnostic,
                )
                if (
                    _canonical(alias_record.alias_body)
                    != _canonical(expected_alias["body"])
                    or not _HMAC_COMPARE_DIGEST(
                        alias_record.handle_fast_fields[7],
                        expected_alias["content_sha256"],
                    )
                    or not _HMAC_COMPARE_DIGEST(
                        alias_record.handle_fast_fields[6],
                        expected_alias["stage_diagnostic_sha256"],
                    )
                    or alias_record.handle_fast_fields[5]
                    != expected_alias["diagnostic_schema"]
                ):
                    _fail(
                        "PHYSICAL_REGISTRY_MISMATCH",
                        "stage alias semantic body changed",
                    )

        expected_count = len(expected_occurrences)
        if (
            set(expected_specs) != set(cores_by_key)
            or set(expected_occurrences) != set(aliases_by_occurrence)
        ):
            _fail(
                "PHYSICAL_REGISTRY_MISMATCH",
                "physical core/alias set differs from raw reconstruction",
            )
        counts = scan_events()
        expected_hits = expected_count - len(expected_specs)
        if (
            counts["physical_builds"] != len(expected_specs)
            or counts["stage_aliases"] != expected_count
            or counts["cross_stage_physical_hits"] != expected_hits
            or counts["admission_full_validations"] != len(expected_specs)
            or counts["private_execution_full_validations"] != 0
            or counts["commit_full_validations"] != 0
            or counts["admitted_stages"] != len(stage_uses)
        ):
            _fail(
                "EVENT_COUNT_MISMATCH",
                "authenticated events differ from raw reconstruction",
            )

        for admission_record in admissions_by_index.values():
            validate_admission_handle(admission_record)
        for physical_key in validated_physical_keys:
            append_event(
                "COMMIT_VALIDATE",
                {
                    "physical_key_sha256": physical_key,
                    "operator_branch": expected_specs[
                        physical_key
                    ].operator_branch,
                },
            )
        check_deadline()
        append_event(
            "COMMIT",
            {
                "physical_count": len(expected_specs),
                "alias_count": expected_count,
                "physical_hit_count": expected_hits,
                "execution_alias_lookups": execution_lookup_count,
            },
        )
        stats = current_stats()
        stats_fields = boundary_mapping(stats, registry_stats_type)
        dependency_binding_gate()
        receipt = _deep_freeze(
            {
                "schema": COMMIT_SCHEMA,
                "registry_schema": SCHEMA,
                "numeric_protocol": NUMERIC_PROTOCOL,
                "root_content_sha256": root_content,
                "raw_root_manifest_sha256": raw_root_manifest,
                "frame_content_sha256": frame_content,
                "bounds_manifest_sha256": initial_bounds_manifest,
                "numeric_contract_sha256": numeric_contract,
                "numeric_platform_sha256": numeric_platform,
                "implementation_sha256": implementation,
                "dependency_implementation_sha256": (
                    dependency_implementation
                ),
                "dependency_implementation": dependency_manifest,
                "module_source_sha256": module_source,
                "stage_use_sha256": [
                    value.stage_use_sha256 for value in stage_uses
                ],
                "physical_key_sha256": sorted(expected_specs),
                "commit_physical_derivations": (
                    commit_physical_derivations
                ),
                "event_count": stats_fields["event_count"],
                "event_chain_head_sha256": (
                    stats_fields["event_chain_head_sha256"]
                ),
                "proof_authority": False,
            }
        )
        certificate = make_boundary(
            physical_registry_certificate_type,
            physical_builds=stats_fields["physical_builds"],
            dense_physical_builds=stats_fields[
                "dense_physical_builds"
            ],
            conv_physical_builds=stats_fields[
                "conv_physical_builds"
            ],
            stage_aliases=stats_fields["stage_aliases"],
            cross_stage_physical_hits=(
                stats_fields["cross_stage_physical_hits"]
            ),
            execution_alias_lookups=stats_fields[
                "execution_alias_lookups"
            ],
            admission_full_validations=(
                stats_fields["admission_full_validations"]
            ),
            private_execution_full_validations=(
                stats_fields["private_execution_full_validations"]
            ),
            commit_full_validations=stats_fields[
                "commit_full_validations"
            ],
            receipt=receipt,
            proof_authority=False,
        )
        validate_raw_anchors()
        dependency_binding_gate()
        check_deadline()
        state["phase"] = "COMMITTED"
        return certificate

    def cleanup() -> None:
        if _OS_GETPID() != owner_pid:
            return
        if not operation_lock.acquire(blocking=False):
            return
        try:
            clear_material()
            aliases_by_occurrence.clear()
            admissions_by_index.clear()
            admissions_by_identity.clear()
            events.clear()
            state["phase"] = "ABORTED"
        finally:
            operation_lock.release()

    def require_port(value: Any) -> None:
        if (
            port_ref is None
            or port_ref() is not value
            or type(value) is not port_abi["type"]
        ):
            _fail("HANDLE_SEAL_MISMATCH", "foreign registry port")

    class _RegistryPort(_NoCopy):
        __slots__ = ("__weakref__",)

        def __init__(self, creation_token: Any):
            if creation_token is not creation_capability:
                _fail("HANDLE_SEAL_MISMATCH", "registry port is factory-only")

        def admit_stage(
            self, stage_use: authority.StageUse
        ) -> StageAdmission:
            require_port(self)
            public_abi_gate()
            with operation_guard:
                try:
                    private_view_abi_gate()
                    result = admit_impl(stage_use)
                    private_view_abi_gate()
                    return result
                except V51BPhysicalRegistryError:
                    poison()
                    raise
                except dependency_error_types as exc:
                    poison()
                    if getattr(exc, "code", "") == "DEADLINE_EXPIRED":
                        raise V51BPhysicalRegistryTimeout() from exc
                    raise V51BPhysicalRegistryError(
                        "DEPENDENCY_FAILURE",
                        f"material admission failed: {exc}",
                    ) from exc
                except Exception as exc:
                    poison()
                    raise V51BPhysicalRegistryError(
                        "INTERNAL_FAILURE",
                        f"unexpected admission failure: {type(exc).__name__}",
                    ) from exc

        def lookup_execution_alias(
            self, admission: StageAdmission, layer_id: int
        ) -> StageAliasHandle:
            require_port(self)
            public_abi_gate()
            with operation_guard:
                try:
                    private_view_abi_gate()
                    result = lookup_impl(admission, layer_id)
                    private_view_abi_gate()
                    return result
                except V51BPhysicalRegistryError:
                    poison()
                    raise
                except Exception as exc:
                    poison()
                    raise V51BPhysicalRegistryError(
                        "INTERNAL_FAILURE",
                        f"unexpected lookup failure: {type(exc).__name__}",
                    ) from exc

        def stats(self) -> RegistryStats:
            require_port(self)
            public_abi_gate()
            with operation_guard:
                try:
                    private_view_abi_gate()
                    check_deadline()
                    if state["phase"] not in ("OPEN", "COMMITTED"):
                        _fail(
                            "INVALID_STATE",
                            f"registry state is {state['phase']}",
                        )
                    result = current_stats()
                    check_deadline()
                    private_view_abi_gate()
                    return result
                except V51BPhysicalRegistryError:
                    poison()
                    raise
                except Exception as exc:
                    poison()
                    raise V51BPhysicalRegistryError(
                        "INTERNAL_FAILURE",
                        f"unexpected stats failure: {type(exc).__name__}",
                    ) from exc

        def commit(self) -> PhysicalRegistryCertificate:
            require_port(self)
            public_abi_gate()
            with operation_guard:
                try:
                    private_view_abi_gate()
                    result = commit_impl()
                    private_view_abi_gate()
                    return result
                except V51BPhysicalRegistryError:
                    poison()
                    raise
                except dependency_error_types as exc:
                    poison()
                    if getattr(exc, "code", "") == "DEADLINE_EXPIRED":
                        raise V51BPhysicalRegistryTimeout() from exc
                    raise V51BPhysicalRegistryError(
                        "DEPENDENCY_FAILURE",
                        f"commit validation failed: {exc}",
                    ) from exc
                except Exception as exc:
                    poison()
                    raise V51BPhysicalRegistryError(
                        "INTERNAL_FAILURE",
                        f"unexpected commit failure: {type(exc).__name__}",
                    ) from exc

        def abort(self) -> None:
            require_port(self)
            public_abi_gate()
            with operation_guard:
                try:
                    private_view_abi_gate()
                    clear_material()
                    aliases_by_occurrence.clear()
                    admissions_by_index.clear()
                    admissions_by_identity.clear()
                    events.clear()
                    state["phase"] = "ABORTED"
                    private_view_abi_gate()
                except V51BPhysicalRegistryError:
                    poison()
                    raise

    if (
        checked_deadline is not None
        and _TIME_MONOTONIC() >= checked_deadline
    ):
        clear_material()
        raise V51BPhysicalRegistryTimeout()
    port_abi["type"] = _RegistryPort
    port_abi["snapshot"] = class_binding_snapshot(_RegistryPort)
    private_view_abi_gate()
    public_abi_gate()
    port = object.__new__(_RegistryPort)
    port_ref = _WEAKREF_REF(port)
    _WEAKREF_FINALIZE(port, cleanup)
    private_view_abi_gate()
    return port


def _seal_physical_registry_factory(
    implementation: FunctionType,
) -> FunctionType:
    """Clone the factory and every module helper into sealed globals."""

    class _FactoryPrivateModuleView:
        """Factory-owned immutable module namespace."""

        __slots__ = ("_module_name", "_values")

        def __init__(
            self,
            module_name: str,
            values: Dict[str, Any],
            *,
            copy_values: bool = True,
            _setattr: Any = object.__setattr__,
            _mapping_proxy: Any = MappingProxyType,
            _dict: Any = dict,
        ):
            _setattr(self, "_module_name", module_name)
            _setattr(
                self,
                "_values",
                _mapping_proxy(
                    _dict(values) if copy_values else values
                ),
            )

        def __getattribute__(
            self,
            name: str,
            _getattribute: Any = object.__getattribute__,
        ) -> Any:
            return _getattribute(self, name)

        @property
        def __dict__(
            self,
            _getattribute: Any = object.__getattribute__,
        ) -> Mapping[str, Any]:
            return _getattribute(self, "_values")

        def __getattr__(
            self,
            name: str,
            _getattribute: Any = object.__getattribute__,
            _attribute_error: Any = AttributeError,
            _key_error: Any = KeyError,
        ) -> Any:
            try:
                return _getattribute(self, "_values")[name]
            except _key_error as exc:
                module_name = _getattribute(self, "_module_name")
                raise _attribute_error(
                    f"{module_name} has no captured attribute {name}"
                ) from exc

        def __setattr__(
            self,
            name: str,
            value: Any,
            _private_fail: Any = _fail,
        ) -> NoReturn:
            del name, value
            _private_fail(
                "DEPENDENCY_SUBSTITUTION",
                "factory-private module view is immutable",
            )

        def __reduce__(
            self,
            _private_fail: Any = _fail,
        ) -> NoReturn:
            _private_fail(
                "COPY_FORBIDDEN",
                "factory-private module view cannot be serialized",
            )

        def __reduce_ex__(
            self,
            protocol: int,
            _private_fail: Any = _fail,
        ) -> NoReturn:
            del protocol
            _private_fail(
                "COPY_FORBIDDEN",
                "factory-private module view cannot be serialized",
            )

    private_view_type = _FactoryPrivateModuleView
    private_view_type_snapshot = _private_view_type_snapshot(
        private_view_type
    )
    module_globals = globals()
    module_owners = _FACTORY_MODULE_OWNERS
    primary_owners = (
        module_owners["query_dual_v51b_physical_registry"],
        module_owners["query_dual_replay"],
        module_owners["query_dual_scalar_guard_v51"],
        module_owners["query_dual_replay_v51_conv"],
        module_owners["query_dual_v51_authority"],
    )
    (
        private_module_aliases,
        bytecode_attribute_guard,
        module_alias_manifest,
    ) = _build_private_module_alias_registry(
        primary_owners,
        view_type=private_view_type,
    )
    module_globals["_PRIVATE_MODULE_ALIASES_BY_OWNER_ID"] = (
        private_module_aliases
    )
    module_globals["_PRIVATE_MODULE_ALIAS_MANIFEST"] = (
        module_alias_manifest
    )

    frozen_owner = module_owners["query_dual_replay"]
    dense_owner = module_owners["query_dual_scalar_guard_v51"]
    conv_owner = module_owners["query_dual_replay_v51_conv"]
    authority_owner = module_owners["query_dual_v51_authority"]
    frozen_globals = _isolated_module_globals(frozen_owner)
    frozen_view = private_view_type(
        frozen_owner.__name__, frozen_globals
    )
    dense_globals = _isolated_module_globals(
        dense_owner, overrides={"_v3": frozen_view}
    )
    dense_view = private_view_type(
        dense_owner.__name__, dense_globals
    )
    conv_globals = _isolated_module_globals(
        conv_owner, overrides={"frozen": frozen_view}
    )
    conv_view = private_view_type(
        conv_owner.__name__, conv_globals
    )
    authority_globals = _isolated_module_globals(authority_owner)
    authority_view = private_view_type(
        authority_owner.__name__, authority_globals
    )
    private_globals = _isolated_module_globals(
        module_owners["query_dual_v51b_physical_registry"],
        overrides={
            "authority": authority_view,
            "conv_v51": conv_view,
            "dense_v51": dense_view,
            "frozen": frozen_view,
        },
    )
    private_globals["_JSON_DUMPS"] = private_globals["json"].dumps
    private_token_hex = private_globals["secrets"].token_hex
    private_globals["_SECRETS_TOKEN_HEX"] = private_token_hex
    private_globals["_SECRETS_TOKEN_BYTES"] = (
        private_token_hex.__globals__["token_bytes"]
    )
    private_globals["_FACTORY_MODULE_OWNERS"] = module_owners
    private_globals["_FACTORY_PRIVATE_MODULE_VIEW_TYPE"] = (
        private_view_type
    )
    private_globals["_FACTORY_PRIVATE_MODULE_VIEW_TYPE_SNAPSHOT"] = (
        private_view_type_snapshot
    )
    private_globals["_PRIVATE_MODULE_ALIASES_BY_OWNER_ID"] = (
        private_module_aliases
    )
    private_globals["_PRIVATE_MODULE_ALIAS_MANIFEST"] = (
        module_alias_manifest
    )
    guarded_public_names = tuple(
        name
        for name in _PRIVATE_BUILTINS_TEMPLATE
        if (
            not name.startswith("__")
            or name == "__build_class__"
        )
    ) + (
        "MemberDescriptorType",
        "hmac",
        "time",
        "os",
        "_HASHLIB_SHA256",
        "_HMAC_NEW",
        "_HMAC_COMPARE_DIGEST",
        "_JSON_DUMPS",
        "_MATH_ISFINITE",
        "_OS_GETPID",
        "_SECRETS_TOKEN_BYTES",
        "_SECRETS_TOKEN_HEX",
        "_TIME_MONOTONIC",
    )
    private_globals["_FACTORY_PUBLIC_GLOBALS"] = module_globals
    private_globals["_FACTORY_PUBLIC_BINDING_GUARD"] = tuple(
        (
            name,
            name in module_globals,
            module_globals.get(name),
        )
        for name in guarded_public_names
    )
    explicit_attribute_guard = (
        (vars(hmac), hmac.__name__, "new", True, hmac.new),
        (vars(hmac), hmac.__name__, "HMAC", True, hmac.HMAC),
        (
            vars(hmac),
            hmac.__name__,
            "compare_digest",
            True,
            hmac.compare_digest,
        ),
        (
            vars(time),
            time.__name__,
            "monotonic",
            True,
            time.monotonic,
        ),
        (vars(os), os.__name__, "getpid", True, os.getpid),
        (
            vars(hashlib),
            hashlib.__name__,
            "sha256",
            True,
            hashlib.sha256,
        ),
        (
            vars(math),
            math.__name__,
            "isfinite",
            True,
            math.isfinite,
        ),
        (
            vars(secrets),
            secrets.__name__,
            "token_bytes",
            True,
            secrets.token_bytes,
        ),
        (
            vars(secrets),
            secrets.__name__,
            "token_hex",
            True,
            secrets.token_hex,
        ),
    )
    attribute_guard = {
        (id(source), name): (
            source,
            owner_name,
            name,
            existed,
            expected,
        )
        for source, owner_name, name, existed, expected in (
            bytecode_attribute_guard + explicit_attribute_guard
        )
    }
    private_globals["_FACTORY_MODULE_ATTRIBUTE_GUARD"] = tuple(
        attribute_guard[key]
        for key in sorted(
            attribute_guard,
            key=lambda item: (
                attribute_guard[item][1],
                attribute_guard[item][2],
            ),
        )
    )
    sealed_implementation = private_globals[implementation.__name__]
    if (
        type(sealed_implementation) is not FunctionType
        or sealed_implementation.__globals__ is not private_globals
    ):
        _fail(
            "DEPENDENCY_SUBSTITUTION",
            "physical-registry factory could not be sealed",
        )

    def open_v51b_frame_physical_registry(
        *,
        full_layers: Mapping[int, frozen._FrozenLayer],
        contexts: Mapping[Optional[int], frozen._SealedCone],
        stage_uses: Tuple[authority.StageUse, ...],
        frame_bounds: Mapping[int, frozen._Box],
        root_content_sha256: str,
        frame_content_sha256: str,
        numeric_contract_sha256: str,
        implementation_sha256: str,
        deadline: Optional[float],
    ) -> Any:
        return sealed_implementation(
            full_layers=full_layers,
            contexts=contexts,
            stage_uses=stage_uses,
            frame_bounds=frame_bounds,
            root_content_sha256=root_content_sha256,
            frame_content_sha256=frame_content_sha256,
            numeric_contract_sha256=numeric_contract_sha256,
            implementation_sha256=implementation_sha256,
            deadline=deadline,
        )

    public_globals = {
        "__builtins__": dict(_PRIVATE_BUILTINS_TEMPLATE),
        "__name__": module_globals["__name__"],
    }
    sealed_wrapper = FunctionType(
        open_v51b_frame_physical_registry.__code__,
        public_globals,
        open_v51b_frame_physical_registry.__name__,
        open_v51b_frame_physical_registry.__defaults__,
        open_v51b_frame_physical_registry.__closure__,
    )
    sealed_wrapper.__kwdefaults__ = (
        None
        if open_v51b_frame_physical_registry.__kwdefaults__ is None
        else dict(open_v51b_frame_physical_registry.__kwdefaults__)
    )
    sealed_wrapper.__annotations__ = dict(
        open_v51b_frame_physical_registry.__annotations__
    )
    sealed_wrapper.__module__ = module_globals["__name__"]
    sealed_wrapper.__qualname__ = "open_v51b_frame_physical_registry"
    sealed_wrapper.__doc__ = (
        "Open one dependency-sealed frame-local physical registry."
    )
    return sealed_wrapper


open_v51b_frame_physical_registry = _seal_physical_registry_factory(
    _open_v51b_frame_physical_registry_impl
)
del _open_v51b_frame_physical_registry_impl
del _seal_physical_registry_factory
del _factory_public_dependency_gate


__all__ = [
    "BOX_OUTPUT",
    "BOX_RELU_POST",
    "BRANCH_CONV_DENSE",
    "BRANCH_DENSE",
    "COMMIT_SCHEMA",
    "CONV_STAGE_DIAGNOSTIC_SCHEMA",
    "DenseNumericCore",
    "DENSE_STAGE_DIAGNOSTIC_SCHEMA",
    "NUMERIC_PROTOCOL",
    "PhysicalCoreHandle",
    "PhysicalRegistryCertificate",
    "RegistryStats",
    "SCHEMA",
    "StageAdmission",
    "StageAliasHandle",
    "V51BPhysicalRegistryError",
    "V51BPhysicalRegistryTimeout",
    "open_v51b_frame_physical_registry",
]
