"""Controlled tests for the isolated V5.1b physical registry."""

from __future__ import annotations

import builtins
import copy
import gc
import hashlib
import os
import pickle
import sys
import time
import unittest
import weakref
from contextlib import ExitStack
from dataclasses import dataclass, replace
from dis import get_instructions
from types import CodeType, FunctionType, MappingProxyType, ModuleType
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as dense_v51
from act.back_end.hybridz_tf import query_dual_v51_authority as authority
from act.back_end.hybridz_tf import (
    query_dual_v51b_physical_registry as physical,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _f64(value: object, *, name: str = "controlled value") -> np.ndarray:
    return frozen._immutable_f64_array(value, name=name)


def _layer(
    layer_id: int,
    kind: str,
    preds: tuple[int, ...],
    *,
    params: dict[str, object] | None = None,
) -> frozen._FrozenLayer:
    return frozen._FrozenLayer(
        id=layer_id,
        kind=kind,
        preds=preds,
        width=1,
        in_vars=(),
        out_vars=(),
        params=MappingProxyType({} if params is None else params),
    )


def _conv(
    layer_id: int, predecessor_id: int, weight: float
) -> frozen._FrozenLayer:
    return _layer(
        layer_id,
        "CONV2D",
        (predecessor_id,),
        params={
            "weight": _f64(
                np.asarray([[[[weight]]]], dtype=np.float64),
                name=f"conv {layer_id} weight",
            ),
            "bias_channels": _f64(
                np.asarray([0.0], dtype=np.float64),
                name=f"conv {layer_id} bias",
            ),
            "input_shape": (1, 1, 1),
            "output_shape": (1, 1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        },
    )


def _dense(
    layer_id: int, predecessor_id: int, weight: float
) -> frozen._FrozenLayer:
    return _layer(
        layer_id,
        "DENSE",
        (predecessor_id,),
        params={
            "weight": _f64(
                np.asarray([[weight]], dtype=np.float64),
                name=f"dense {layer_id} weight",
            ),
            "bias": _f64(
                np.asarray([0.0], dtype=np.float64),
                name=f"dense {layer_id} bias",
            ),
        },
    )


def _box(lower: float, upper: float, *, name: str) -> frozen._Box:
    return frozen._Box(
        lb=_f64(np.asarray([lower], dtype=np.float64), name=f"{name} lb"),
        ub=_f64(np.asarray([upper], dtype=np.float64), name=f"{name} ub"),
    )


@dataclass(frozen=True)
class _ControlledTopology:
    full_layers: MappingProxyType
    contexts: MappingProxyType
    stage_uses: tuple[authority.StageUse, ...]
    frame_bounds: MappingProxyType


def _five_cone_topology() -> _ControlledTopology:
    # Affine prefix counts at the four target starts are 1, 2, 4, and 6.
    # The final property has all six Conv layers plus the Dense layer: 7.
    layers = {
        0: _layer(0, "INPUT", ()),
        1: _layer(1, "INPUT_SPEC", (0,)),
        2: _conv(2, 1, 1.0),
        3: _layer(3, "RELU", (2,)),
        4: _conv(4, 3, -0.75),
        5: _layer(5, "RELU", (4,)),
        6: _conv(6, 5, 1.125),
        7: _layer(7, "RELU", (6,)),
        8: _conv(8, 7, -1.25),
        9: _layer(9, "RELU", (8,)),
        10: _conv(10, 9, 0.625),
        11: _layer(11, "RELU", (10,)),
        12: _conv(12, 11, 1.375),
        13: _layer(13, "RELU", (12,)),
        14: _dense(14, 13, -0.875),
        15: _layer(15, "ASSERT", (14,)),
    }
    full_layers = MappingProxyType(layers)
    manifests = {
        lid: frozen._layer_manifest(layer)
        for lid, layer in full_layers.items()
    }
    starts = (2, 4, 8, 12, None)
    contexts = MappingProxyType(
        {
            start: frozen._sealed_cone(
                full_layers,
                manifests,
                assert_id=15,
                start_lid=start,
            )
            for start in starts
        }
    )
    stage_uses = (
        authority.StageUse(
            use_index=0,
            stage_kind=authority.STAGE_TARGET,
            stage_index=0,
            target_relu_lid=3,
            cone_start_lid=2,
        ),
        authority.StageUse(
            use_index=1,
            stage_kind=authority.STAGE_TARGET,
            stage_index=1,
            target_relu_lid=5,
            cone_start_lid=4,
        ),
        authority.StageUse(
            use_index=2,
            stage_kind=authority.STAGE_TARGET,
            stage_index=2,
            target_relu_lid=9,
            cone_start_lid=8,
        ),
        authority.StageUse(
            use_index=3,
            stage_kind=authority.STAGE_TARGET,
            stage_index=3,
            target_relu_lid=13,
            cone_start_lid=12,
        ),
        authority.StageUse(
            use_index=4,
            stage_kind=authority.STAGE_PROPERTY,
            stage_index=None,
            target_relu_lid=None,
            cone_start_lid=None,
        ),
    )
    boxes = {
        1: _box(-1.0, 1.0, name="input spec"),
        2: _box(-0.75, 0.8, name="conv 2"),
        3: _box(-0.75, 0.8, name="relu 3 raw"),
        4: _box(-0.7, 0.9, name="conv 4"),
        5: _box(-0.7, 0.9, name="relu 5 raw"),
        6: _box(-0.6, 1.0, name="conv 6"),
        7: _box(-0.6, 1.0, name="relu 7 raw"),
        8: _box(-0.5, 1.1, name="conv 8"),
        9: _box(-0.5, 1.1, name="relu 9 raw"),
        10: _box(-0.4, 1.2, name="conv 10"),
        11: _box(-0.4, 1.2, name="relu 11 raw"),
        12: _box(-0.3, 1.3, name="conv 12"),
        13: _box(-0.3, 1.3, name="relu 13 raw"),
        14: _box(-1.2, 1.1, name="dense 14"),
    }
    return _ControlledTopology(
        full_layers=full_layers,
        contexts=contexts,
        stage_uses=stage_uses,
        frame_bounds=MappingProxyType(boxes),
    )


def _shared_dense_topology() -> _ControlledTopology:
    layers = {
        0: _layer(0, "INPUT", ()),
        1: _layer(1, "INPUT_SPEC", (0,)),
        2: _dense(2, 1, 1.25),
        3: _layer(3, "RELU", (2,)),
        4: _layer(4, "ASSERT", (3,)),
    }
    full_layers = MappingProxyType(layers)
    manifests = {
        lid: frozen._layer_manifest(layer)
        for lid, layer in full_layers.items()
    }
    contexts = MappingProxyType(
        {
            start: frozen._sealed_cone(
                full_layers,
                manifests,
                assert_id=4,
                start_lid=start,
            )
            for start in (2, None)
        }
    )
    stage_uses = (
        authority.StageUse(
            use_index=0,
            stage_kind=authority.STAGE_TARGET,
            stage_index=0,
            target_relu_lid=3,
            cone_start_lid=2,
        ),
        authority.StageUse(
            use_index=1,
            stage_kind=authority.STAGE_PROPERTY,
            stage_index=None,
            target_relu_lid=None,
            cone_start_lid=None,
        ),
    )
    return _ControlledTopology(
        full_layers=full_layers,
        contexts=contexts,
        stage_uses=stage_uses,
        frame_bounds=MappingProxyType(
            {
                1: _box(-1.0, 1.0, name="Dense input"),
                2: _box(-1.25, 1.25, name="Dense output"),
                3: _box(-1.25, 1.25, name="Dense ReLU raw"),
            }
        ),
    )


def _open(
    topology: _ControlledTopology,
    *,
    frame_label: str = "controlled-frame-a",
    deadline: float | None = None,
):
    return physical.open_v51b_frame_physical_registry(
        full_layers=topology.full_layers,
        contexts=topology.contexts,
        stage_uses=topology.stage_uses,
        frame_bounds=topology.frame_bounds,
        root_content_sha256=_sha("controlled-raw-root"),
        frame_content_sha256=_sha(frame_label),
        numeric_contract_sha256=_sha("controlled-v51b-numeric-contract"),
        implementation_sha256=_sha("controlled-v51b-implementation"),
        deadline=time.monotonic() + 60.0 if deadline is None else deadline,
    )


def _admit_all(registry: object, topology: _ControlledTopology):
    return tuple(
        registry.admit_stage(stage_use)
        for stage_use in topology.stage_uses
    )


def _registry_core_records(registry: object):
    port_method = type(registry).admit_stage
    port_cells = dict(
        zip(
            port_method.__code__.co_freevars,
            (cell.cell_contents for cell in port_method.__closure__),
        )
    )
    admit_impl = port_cells["admit_impl"]
    admit_cells = dict(
        zip(
            admit_impl.__code__.co_freevars,
            (cell.cell_contents for cell in admit_impl.__closure__),
        )
    )
    return admit_cells["cores_by_key"]


def _closure_cells(value: FunctionType) -> dict[str, object]:
    return dict(
        zip(
            value.__code__.co_freevars,
            (cell.cell_contents for cell in value.__closure__ or ()),
        )
    )


def _registry_check_deadline(registry: object, method_name: str):
    port_method = type.__getattribute__(
        type(registry), "__dict__"
    )[method_name]
    cells = _closure_cells(port_method)
    if "check_deadline" in cells:
        return cells["check_deadline"]
    implementation = cells[f"{method_name}_impl"]
    return _closure_cells(implementation)["check_deadline"]


def _set_registry_deadline(
    registry: object, method_name: str, deadline: float
) -> None:
    check_deadline = _registry_check_deadline(registry, method_name)
    freevars = check_deadline.__code__.co_freevars
    deadline_cell = check_deadline.__closure__[
        freevars.index("checked_deadline")
    ]
    deadline_cell.cell_contents = deadline


class _DependencyTrap:
    __slots__ = ("calls", "label")

    def __init__(self, label: str):
        self.calls = 0
        self.label = label

    def __call__(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        raise AssertionError(f"perturbed dependency called: {self.label}")

    def __getattr__(self, name: str):
        self.calls += 1
        raise AssertionError(
            f"perturbed dependency read: {self.label}.{name}"
        )


def _dependency_perturbation_cases():
    physical_globals = vars(physical)
    return (
        *(
            (f"physical.{name}", physical_globals, name)
            for name in (
                "enumerate",
                "zip",
                "len",
                "type",
                "id",
                "MemberDescriptorType",
                "object",
                "hmac",
                "time",
                "os",
            )
        ),
        *(
            (f"physical.{name}", physical_globals, name)
            for name in (
                "_HASHLIB_SHA256",
                "_HMAC_NEW",
                "_HMAC_COMPARE_DIGEST",
                "_MATH_ISFINITE",
                "_OS_GETPID",
                "_TIME_MONOTONIC",
            )
        ),
        *(
            (f"builtins.{name}", vars(builtins), name)
            for name in (
                "enumerate",
                "zip",
                "len",
                "type",
                "id",
                "object",
                "__build_class__",
            )
        ),
        ("hmac.new", vars(physical.hmac), "new"),
        ("hmac.HMAC", vars(physical.hmac), "HMAC"),
        (
            "hmac.compare_digest",
            vars(physical.hmac),
            "compare_digest",
        ),
        ("time.monotonic", vars(physical.time), "monotonic"),
        ("os.getpid", vars(physical.os), "getpid"),
        ("hashlib.sha256", vars(physical.hashlib), "sha256"),
        ("math.isfinite", vars(physical.math), "isfinite"),
    )


def _numeric_module_attribute_perturbation_cases():
    numpy_attributes = (
        "asarray",
        "nextafter",
        "any",
        "all",
        "ascontiguousarray",
        "array_equal",
        "abs",
        "maximum",
        "prod",
        "isfinite",
        "dtype",
        "frombuffer",
        "longdouble",
        "float64",
        "ndarray",
    )
    return (
        *(
            (f"numpy.{name}", vars(np), name)
            for name in numpy_attributes
        ),
        (
            "math.isfinite",
            vars(physical.math),
            "isfinite",
        ),
        ("json.dumps", vars(physical.json), "dumps"),
        (
            "hashlib.sha256",
            vars(physical.hashlib),
            "sha256",
        ),
        (
            "hmac.compare_digest",
            vars(physical.hmac),
            "compare_digest",
        ),
        ("hmac.new", vars(physical.hmac), "new"),
    )


def _recursive_codes(value: CodeType) -> tuple[CodeType, ...]:
    result = [value]
    for constant in value.co_consts:
        if type(constant) is CodeType:
            result.extend(_recursive_codes(constant))
    return tuple(result)


def _install_perturbation(
    owner_globals: dict[str, object],
    name: str,
    replacement: object,
) -> tuple[bool, object | None]:
    existed = dict.__contains__(owner_globals, name)
    original = (
        dict.__getitem__(owner_globals, name) if existed else None
    )
    dict.__setitem__(owner_globals, name, replacement)
    return existed, original


def _restore_perturbation(
    owner_globals: dict[str, object],
    name: str,
    existed: bool,
    original: object | None,
) -> None:
    if existed:
        dict.__setitem__(owner_globals, name, original)
    else:
        dict.__delitem__(owner_globals, name)


def _hot_path_profile_targets():
    python_functions = (
        physical._raw_identity_snapshot,
        physical._identity_tree,
        physical._root_manifest,
        physical._frame_bounds_manifest,
        physical._array_sha256,
        physical._json_sha256,
        physical._json_bytes,
        physical._sealed_hmac,
        physical._dense_numeric_core,
        physical._validate_dense_numeric_core,
        physical._derive_source_spec,
        physical._module_binding_snapshot,
        physical._module_function_manifest,
        frozen._layer_manifest,
        conv_v51._geometry,
        conv_v51._validate_plan,
        dense_v51._validate_support,
    )
    python_codes = {
        value.__code__: value.__qualname__
        for value in python_functions
    }
    c_functions = (
        physical._HASHLIB_SHA256,
        physical._HMAC_NEW,
        physical._HMAC_COMPARE_DIGEST,
        physical._MATH_ISFINITE,
    )
    c_function_ids = {
        id(value): getattr(value, "__qualname__", repr(value))
        for value in c_functions
    }
    return python_codes, c_function_ids


def _hot_path_profiler(
    python_codes: dict[object, str],
    c_function_ids: dict[int, str],
    hits: dict[str, int],
):
    def profile(frame, event, arg):
        label = None
        if event == "call":
            label = python_codes.get(frame.f_code)
        elif event == "c_call":
            label = c_function_ids.get(id(arg))
        if label is not None:
            hits[label] = hits.get(label, 0) + 1

    return profile


def _replacement_field_value(value: object) -> object:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1000
    if type(value) is str:
        return "f" * 64 if len(value) == 64 else f"{value}-forged"
    if type(value) is tuple:
        return value + (value[0],) if value else (object(),)
    return object()


class QueryDualV51bPhysicalRegistryTests(unittest.TestCase):
    def test_factory_and_nested_functions_use_private_fixed_globals(self):
        public_factory = physical.open_v51b_frame_physical_registry
        public_globals = vars(physical)
        self.assertIsNot(public_factory.__globals__, public_globals)
        self.assertIsNot(
            public_factory.__globals__["__builtins__"],
            vars(builtins),
        )
        implementation = _closure_cells(public_factory)[
            "sealed_implementation"
        ]
        private_globals = implementation.__globals__
        self.assertIsNot(private_globals, public_globals)
        self.assertIsNot(
            private_globals["__builtins__"], vars(builtins)
        )
        for name in (
            "enumerate",
            "zip",
            "len",
            "type",
            "id",
            "object",
            "__build_class__",
        ):
            self.assertIs(
                private_globals["__builtins__"][name],
                vars(builtins)[name],
            )
        for value in private_globals.values():
            if (
                type(value) is FunctionType
                and value.__module__ == physical.__name__
            ):
                self.assertIs(value.__globals__, private_globals)

        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="private-global-audit")
        public_dependency_globals = tuple(
            vars(module)
            for module in (
                physical,
                frozen,
                dense_v51,
                conv_v51,
                authority,
            )
        )
        pending = [
            value
            for value in type.__getattribute__(
                type(registry), "__dict__"
            ).values()
            if type(value) is FunctionType
        ]
        seen = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            self.assertTrue(
                all(
                    value.__globals__ is not module_globals
                    for module_globals in public_dependency_globals
                )
            )
            self.assertIsNot(
                value.__globals__.get("__builtins__"),
                vars(builtins),
            )
            for cell in value.__closure__ or ():
                contents = cell.cell_contents
                if type(contents) is FunctionType:
                    pending.append(contents)
        registry.abort()

    def test_recursive_bytecode_module_alias_capture_is_complete(self):
        public_factory = physical.open_v51b_frame_physical_registry
        implementation = _closure_cells(public_factory)[
            "sealed_implementation"
        ]
        sealed_globals = implementation.__globals__
        self.assertEqual(
            [
                name
                for name, value in sealed_globals.items()
                if type(value) is ModuleType
            ],
            [],
        )

        expected_numpy_attributes = {
            "asarray",
            "nextafter",
            "any",
            "all",
            "ascontiguousarray",
            "array_equal",
            "abs",
            "maximum",
            "prod",
            "isfinite",
            "dtype",
            "frombuffer",
            "longdouble",
            "float64",
            "ndarray",
        }
        manifest = physical._PRIVATE_MODULE_ALIAS_MANIFEST
        numpy_attributes = set()
        captured_modules = set()
        for aliases in manifest.values():
            for value in aliases.values():
                captured_modules.add(value["module"])
                if value["module"] == np.__name__:
                    numpy_attributes.update(
                        path.split(".", 1)[0]
                        for path in value["attribute_paths"]
                    )
        self.assertTrue(
            expected_numpy_attributes.issubset(numpy_attributes)
        )
        self.assertTrue(
            {
                "hashlib",
                "hmac",
                "json",
                "math",
                "numpy",
            }.issubset(captured_modules)
        )

        captured_type = type(sealed_globals["np"])
        self.assertIsNot(captured_type, ModuleType)
        self.assertIsNot(captured_type, physical._CapturedModuleNamespace)
        captured_namespace = type.__getattribute__(
            captured_type, "__dict__"
        )
        self.assertTrue(
            {
                "__getattribute__",
                "__getattr__",
                "__dict__",
                "__setattr__",
                "__reduce_ex__",
            }.issubset(captured_namespace)
        )
        self.assertIs(type(sealed_globals["json"].dumps), FunctionType)
        self.assertIsNot(
            sealed_globals["json"].dumps.__globals__,
            vars(physical.json),
        )
        self.assertIs(
            sealed_globals["_JSON_DUMPS"],
            sealed_globals["json"].dumps,
        )
        self.assertIs(
            type(sealed_globals["np"].array_equal), FunctionType
        )
        self.assertIsNot(
            sealed_globals["np"].array_equal,
            np.array_equal,
        )
        for name in ("frozen", "dense_v51", "conv_v51", "authority"):
            view = sealed_globals[name]
            self.assertIs(type(view), captured_type)
            self.assertEqual(
                [
                    key
                    for key, value in vars(view).items()
                    if type(value) is ModuleType
                ],
                [],
            )

        topology = _shared_dense_topology()
        registry = _open(
            topology, frame_label="recursive-module-alias-audit"
        )
        roots = [
            value
            for value in sealed_globals.values()
            if (
                type(value) is FunctionType
                and value.__globals__ is sealed_globals
            )
        ]
        roots.extend(
            value
            for value in type.__getattribute__(
                type(registry), "__dict__"
            ).values()
            if type(value) is FunctionType
        )
        public_dependency_globals = tuple(
            vars(module)
            for module in (
                physical,
                frozen,
                dense_v51,
                conv_v51,
                authority,
                physical.json,
                physical.secrets,
            )
        )
        pending = list(roots)
        seen = set()
        isolated_globals = {}
        while pending:
            function = pending.pop()
            if id(function) in seen:
                continue
            seen.add(id(function))
            function_globals = function.__globals__
            self.assertTrue(
                all(
                    function_globals is not public_globals
                    for public_globals in public_dependency_globals
                )
            )
            self.assertEqual(
                [
                    name
                    for name, value in function_globals.items()
                    if type(value) is ModuleType
                ],
                [],
                function.__qualname__,
            )
            isolated_globals[id(function_globals)] = function_globals
            for cell in function.__closure__ or ():
                contents = cell.cell_contents
                if type(contents) is FunctionType:
                    pending.append(contents)
            for code in _recursive_codes(function.__code__):
                instructions = tuple(get_instructions(code))
                for index, instruction in enumerate(instructions):
                    if instruction.opname != "LOAD_GLOBAL":
                        continue
                    candidate = function_globals.get(
                        instruction.argval
                    )
                    self.assertIsNot(
                        type(candidate),
                        ModuleType,
                        (
                            f"{function.__qualname__} loads live module "
                            f"{instruction.argval}"
                        ),
                    )
                    following = index + 1
                    while (
                        type(candidate) is captured_type
                        and following < len(instructions)
                        and instructions[following].opname
                        in ("LOAD_ATTR", "LOAD_METHOD")
                    ):
                        try:
                            candidate = getattr(
                                candidate,
                                instructions[following].argval,
                            )
                        except AttributeError:
                            candidate = None
                            break
                        following += 1
                    if type(candidate) is FunctionType:
                        pending.append(candidate)
        self.assertGreaterEqual(len(isolated_globals), 6)
        registry.abort()

    def test_numeric_module_attribute_preopen_changes_reject_and_restore(
        self,
    ):
        topology = _shared_dense_topology()
        hashes = {
            "root_content_sha256": _sha("numeric-preopen-root"),
            "frame_content_sha256": _sha("numeric-preopen-frame"),
            "numeric_contract_sha256": _sha("numeric-preopen-contract"),
            "implementation_sha256": _sha(
                "numeric-preopen-implementation"
            ),
        }
        for label, owner_globals, name in (
            _numeric_module_attribute_perturbation_cases()
        ):
            with self.subTest(dependency=label):
                trap = _DependencyTrap(label)
                existed, original = _install_perturbation(
                    owner_globals, name, trap
                )
                try:
                    with self.assertRaises(
                        physical.V51BPhysicalRegistryError
                    ) as captured:
                        physical.open_v51b_frame_physical_registry(
                            full_layers=topology.full_layers,
                            contexts=topology.contexts,
                            stage_uses=topology.stage_uses,
                            frame_bounds=topology.frame_bounds,
                            deadline=time.monotonic() + 60.0,
                            **hashes,
                        )
                finally:
                    _restore_perturbation(
                        owner_globals,
                        name,
                        existed,
                        original,
                    )
                self.assertEqual(
                    captured.exception.code,
                    "DEPENDENCY_SUBSTITUTION",
                )
                self.assertEqual(trap.calls, 0)
                recovered = _open(
                    topology, frame_label=f"restored-{label}"
                )
                admission = recovered.admit_stage(
                    topology.stage_uses[0]
                )
                self.assertEqual(len(admission.aliases), 1)
                recovered.abort()

    def test_numeric_module_attribute_postopen_changes_never_dispatch(
        self,
    ):
        for label, owner_globals, name in (
            _numeric_module_attribute_perturbation_cases()
        ):
            with self.subTest(dependency=label, operation="admission"):
                topology = _shared_dense_topology()
                registry = _open(
                    topology, frame_label=f"postopen-admit-{label}"
                )
                first = registry.admit_stage(topology.stage_uses[0])
                trap = _DependencyTrap(label)
                existed, original = _install_perturbation(
                    owner_globals, name, trap
                )
                try:
                    self.assertIs(
                        registry.lookup_execution_alias(first, 2),
                        first.aliases[0],
                    )
                    self.assertEqual(
                        registry.stats().execution_alias_lookups, 1
                    )
                    with self.assertRaises(
                        physical.V51BPhysicalRegistryError
                    ) as captured:
                        registry.admit_stage(topology.stage_uses[1])
                finally:
                    _restore_perturbation(
                        owner_globals,
                        name,
                        existed,
                        original,
                    )
                self.assertEqual(
                    captured.exception.code,
                    "DEPENDENCY_SUBSTITUTION",
                )
                self.assertEqual(trap.calls, 0)
                registry.abort()

            with self.subTest(dependency=label, operation="commit"):
                topology = _shared_dense_topology()
                registry = _open(
                    topology, frame_label=f"postopen-commit-{label}"
                )
                _admit_all(registry, topology)
                trap = _DependencyTrap(label)
                existed, original = _install_perturbation(
                    owner_globals, name, trap
                )
                try:
                    with self.assertRaises(
                        physical.V51BPhysicalRegistryError
                    ) as captured:
                        registry.commit()
                finally:
                    _restore_perturbation(
                        owner_globals,
                        name,
                        existed,
                        original,
                    )
                self.assertEqual(
                    captured.exception.code,
                    "DEPENDENCY_SUBSTITUTION",
                )
                self.assertEqual(trap.calls, 0)
                registry.abort()

    def test_postopen_numpy_asarray_counting_wrapper_is_never_called(
        self,
    ):
        topology = _shared_dense_topology()
        registry = _open(
            topology, frame_label="postopen-numpy-asarray-counting"
        )
        owner_globals = vars(np)
        original = owner_globals["asarray"]
        state = {
            "calls": 0,
            "patched": False,
            "restored": False,
        }

        def counting_asarray(*args, **kwargs):
            state["calls"] += 1
            return original(*args, **kwargs)

        def perturb_inside_private_derivation(frame, event, arg):
            del arg
            if (
                frame.f_code.co_name == "_derive_source_spec"
                and frame.f_globals is not vars(physical)
            ):
                if event == "call" and not state["patched"]:
                    owner_globals["asarray"] = counting_asarray
                    state["patched"] = True
                elif (
                    event == "return"
                    and state["patched"]
                    and not state["restored"]
                ):
                    owner_globals["asarray"] = original
                    state["restored"] = True
            return perturb_inside_private_derivation

        previous_trace = sys.gettrace()
        try:
            sys.settrace(perturb_inside_private_derivation)
            admission = registry.admit_stage(topology.stage_uses[0])
        finally:
            sys.settrace(previous_trace)
            owner_globals["asarray"] = original
            if state["patched"]:
                state["restored"] = True
        self.assertEqual(len(admission.aliases), 1)
        self.assertTrue(state["patched"])
        self.assertTrue(state["restored"])
        self.assertEqual(state["calls"], 0)
        registry.abort()

    def test_public_legacy_module_view_type_change_is_inert(self):
        topology = _shared_dense_topology()
        registry = _open(
            topology, frame_label="public-legacy-view-type-inert"
        )
        implementation = _closure_cells(
            physical.open_v51b_frame_physical_registry
        )["sealed_implementation"]
        runtime_view_type = type(implementation.__globals__["np"])
        self.assertIsNot(
            runtime_view_type, physical._CapturedModuleNamespace
        )

        legacy_type = physical._CapturedModuleNamespace
        legacy_namespace = type.__getattribute__(
            legacy_type, "__dict__"
        )
        original_getattr = legacy_namespace["__getattr__"]
        original_asarray = np.asarray
        calls = {"getattr": 0, "asarray": 0}

        def counting_asarray(*args, **kwargs):
            calls["asarray"] += 1
            return original_asarray(*args, **kwargs)

        def changed_getattr(self, name):
            calls["getattr"] += 1
            if name == "asarray":
                return counting_asarray
            return original_getattr(self, name)

        type.__setattr__(
            legacy_type, "__getattr__", changed_getattr
        )
        try:
            admission = registry.admit_stage(
                topology.stage_uses[0]
            )
        finally:
            type.__setattr__(
                legacy_type, "__getattr__", original_getattr
            )
        self.assertEqual(len(admission.aliases), 1)
        self.assertEqual(calls, {"getattr": 0, "asarray": 0})
        registry.abort()

    def test_private_module_view_namespace_changes_reject_before_dispatch(
        self,
    ):
        implementation = _closure_cells(
            physical.open_v51b_frame_physical_registry
        )["sealed_implementation"]
        private_type = type(implementation.__globals__["np"])
        guarded_names = (
            "__getattribute__",
            "__getattr__",
            "__dict__",
            "__setattr__",
            "__reduce_ex__",
        )
        for name in guarded_names:
            with self.subTest(binding=name):
                topology = _shared_dense_topology()
                registry = _open(
                    topology,
                    frame_label=f"private-view-binding-{name}",
                )
                namespace = type.__getattribute__(
                    private_type, "__dict__"
                )
                original = namespace[name]
                trap = _DependencyTrap(name)
                calls = {"getattr": 0, "asarray": 0}
                if name == "__getattr__":
                    original_asarray = np.asarray

                    def counting_asarray(*args, **kwargs):
                        calls["asarray"] += 1
                        return original_asarray(*args, **kwargs)

                    def changed_getattr(self, attribute):
                        calls["getattr"] += 1
                        if attribute == "asarray":
                            return counting_asarray
                        return original(self, attribute)

                    replacement = changed_getattr
                else:
                    replacement = trap
                if name == "__dict__":
                    method = original.fget
                    original_code = method.__code__

                    def forged_dict(
                        self, _getattribute=None
                    ):
                        del self, _getattribute
                        raise AssertionError(
                            "forged private view __dict__ called"
                        )

                    method.__code__ = forged_dict.__code__
                else:
                    type.__setattr__(
                        private_type, name, replacement
                    )
                try:
                    with self.assertRaises(
                        physical.V51BPhysicalRegistryError
                    ) as captured:
                        registry.admit_stage(
                            topology.stage_uses[0]
                        )
                finally:
                    if name == "__dict__":
                        method.__code__ = original_code
                    else:
                        type.__setattr__(
                            private_type, name, original
                        )
                self.assertEqual(
                    captured.exception.code,
                    "DEPENDENCY_SUBSTITUTION",
                )
                self.assertEqual(trap.calls, 0)
                self.assertEqual(calls, {"getattr": 0, "asarray": 0})
                with self.assertRaisesRegex(
                    physical.V51BPhysicalRegistryError,
                    "INVALID_STATE",
                ):
                    registry.stats()
                registry.abort()

        with self.subTest(binding="__getattr__.__code__"):
            topology = _shared_dense_topology()
            registry = _open(
                topology,
                frame_label="private-view-getattr-code",
            )
            namespace = type.__getattribute__(
                private_type, "__dict__"
            )
            method = namespace["__getattr__"]
            original_code = method.__code__

            def forged_getattr(
                self,
                name,
                _getattribute=None,
                _attribute_error=None,
                _key_error=None,
            ):
                del self, name, _getattribute, _attribute_error, _key_error
                raise AssertionError("forged private view method called")

            method.__code__ = forged_getattr.__code__
            try:
                with self.assertRaises(
                    physical.V51BPhysicalRegistryError
                ) as captured:
                    registry.admit_stage(topology.stage_uses[0])
            finally:
                method.__code__ = original_code
            self.assertEqual(
                captured.exception.code,
                "DEPENDENCY_SUBSTITUTION",
            )
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "INVALID_STATE",
            ):
                registry.stats()
            registry.abort()

    def test_private_view_all_method_defaults_use_strong_anchors(
        self,
    ):
        # CPython can reuse the address of a released defaults tuple.  This
        # first reproduces the allocator condition that made an id-only
        # snapshot insufficient.
        values = (object(), object(), object())
        released = tuple(list(values))
        released_id = id(released)
        del released
        reused = False
        for _ in range(100_000):
            candidate = tuple(list(values))
            if id(candidate) == released_id:
                reused = True
                break
            del candidate
        self.assertTrue(reused)

        implementation = _closure_cells(
            physical.open_v51b_frame_physical_registry
        )["sealed_implementation"]
        private_type = type(implementation.__globals__["np"])
        snapshot = implementation.__globals__[
            "_FACTORY_PRIVATE_MODULE_VIEW_TYPE_SNAPSHOT"
        ]
        method_snapshots = {
            entry[0]: entry for entry in snapshot[3]
        }
        guarded_names = (
            "__getattribute__",
            "__getattr__",
            "__dict__",
            "__setattr__",
            "__reduce_ex__",
        )
        numeric_codes = {
            dense_v51.prepare_dense_support_v51.__code__,
            dense_v51._validate_support.__code__,
            conv_v51.prepare_dense_conv_v51_plan.__code__,
            conv_v51._geometry.__code__,
            conv_v51._validate_plan.__code__,
            physical._dense_numeric_body.__code__,
            physical._dense_numeric_core.__code__,
            physical._validate_dense_numeric_core.__code__,
        }

        for name in guarded_names:
            with self.subTest(binding=f"{name}.__defaults__"):
                topology = _shared_dense_topology()
                registry = _open(
                    topology,
                    frame_label=f"private-default-anchor-{name}",
                )
                namespace = type.__getattribute__(
                    private_type, "__dict__"
                )
                item = namespace[name]
                function = item.fget if type(item) is property else item
                method_snapshot = method_snapshots[name]
                original_defaults = function.__defaults__
                self.assertIs(method_snapshot[2], function)
                self.assertIs(method_snapshot[3], function.__code__)
                self.assertIs(method_snapshot[4], original_defaults)
                self.assertIsNotNone(original_defaults)

                trap = _DependencyTrap(f"{name} changed default")
                changed_items = list(original_defaults)
                changed_items[0] = trap
                replacement = tuple(changed_items)
                self.assertIsNot(replacement, original_defaults)
                # The snapshot's strong reference makes reuse of this live
                # defaults tuple impossible even under allocator pressure.
                original_id = id(original_defaults)
                self.assertTrue(
                    all(
                        id(tuple(changed_items)) != original_id
                        for _ in range(10_000)
                    )
                )
                numeric_calls = []

                def profile(frame, event, arg):
                    del arg
                    if event == "call" and frame.f_code in numeric_codes:
                        numeric_calls.append(frame.f_code.co_name)

                previous_profile = sys.getprofile()
                function.__defaults__ = replacement
                try:
                    sys.setprofile(profile)
                    with self.assertRaises(
                        physical.V51BPhysicalRegistryError
                    ) as captured:
                        registry.admit_stage(topology.stage_uses[0])
                finally:
                    sys.setprofile(previous_profile)
                    function.__defaults__ = original_defaults
                self.assertEqual(
                    captured.exception.code,
                    "DEPENDENCY_SUBSTITUTION",
                )
                self.assertEqual(trap.calls, 0)
                self.assertEqual(numeric_calls, [])
                with self.assertRaisesRegex(
                    physical.V51BPhysicalRegistryError,
                    "INVALID_STATE",
                ):
                    registry.stats()
                registry.abort()

    def test_strong_snapshots_check_exact_mutable_contents(self):
        implementation = _closure_cells(
            physical.open_v51b_frame_physical_registry
        )["sealed_implementation"]
        private_type = type(implementation.__globals__["np"])
        namespace = type.__getattribute__(private_type, "__dict__")
        function = namespace["__getattr__"]
        original_kwdefaults = function.__kwdefaults__
        anchor = object()
        controlled_kwdefaults = {"probe": anchor}
        function.__kwdefaults__ = controlled_kwdefaults
        try:
            snapshot = physical._private_view_type_snapshot(private_type)
            self.assertTrue(
                physical._private_view_type_matches(
                    private_type, snapshot
                )
            )
            self.assertIs(snapshot[3][1][6], controlled_kwdefaults)
            controlled_kwdefaults["probe"] = object()
            self.assertFalse(
                physical._private_view_type_matches(
                    private_type, snapshot
                )
            )
        finally:
            function.__kwdefaults__ = original_kwdefaults

        class Boundary:
            marker = object()

        class_snapshot = physical._class_binding_snapshot(Boundary)
        class_namespace = type.__getattribute__(Boundary, "__dict__")
        marker_index = tuple(class_namespace.keys()).index("marker")
        original_marker = class_namespace["marker"]
        self.assertIs(
            class_snapshot[0][2][marker_index], original_marker
        )
        type.__setattr__(Boundary, "marker", object())
        self.assertFalse(
            physical._class_binding_matches(Boundary, class_snapshot)
        )

        module_snapshot = physical._module_binding_snapshot(
            vars(physical)
        )
        fail_entry = next(
            entry for entry in module_snapshot if entry[0] == "_fail"
        )
        self.assertIs(fail_entry[2], physical._fail)
        self.assertIs(fail_entry[4], physical._fail.__code__)

        topology = _shared_dense_topology()
        raw_snapshot = physical._raw_identity_snapshot(
            full_layers=topology.full_layers,
            contexts=topology.contexts,
            stage_uses=topology.stage_uses,
            frame_bounds=topology.frame_bounds,
        )
        self.assertIs(raw_snapshot[1], topology.full_layers)
        self.assertIs(raw_snapshot[4], topology.contexts)
        self.assertIs(raw_snapshot[7], topology.stage_uses)
        self.assertIs(raw_snapshot[10], topology.frame_bounds)
        self.assertIs(
            raw_snapshot[2][0][2],
            topology.full_layers[raw_snapshot[2][0][0]],
        )
        first_bound_id = raw_snapshot[11][0][0]
        self.assertIs(
            raw_snapshot[11][0][2],
            topology.frame_bounds[first_bound_id],
        )
        bound_tree = raw_snapshot[11][0][3]
        self.assertEqual(bound_tree[0], "ndarray")
        self.assertIs(
            bound_tree[2],
            topology.frame_bounds[first_bound_id].lb,
        )

    def test_private_module_view_post_operation_gate_rejects_change(
        self,
    ):
        topology = _shared_dense_topology()
        registry = _open(
            topology, frame_label="private-view-post-operation-gate"
        )
        admission = registry.admit_stage(topology.stage_uses[0])
        implementation = _closure_cells(
            physical.open_v51b_frame_physical_registry
        )["sealed_implementation"]
        private_type = type(implementation.__globals__["np"])
        namespace = type.__getattribute__(private_type, "__dict__")
        original_getattr = namespace["__getattr__"]
        trap = _DependencyTrap("private post-operation __getattr__")
        state = {"patched": False}

        def patch_after_entry(frame, event, arg):
            del arg
            if (
                event == "call"
                and frame.f_code.co_name == "lookup_impl"
                and not state["patched"]
            ):
                type.__setattr__(
                    private_type, "__getattr__", trap
                )
                state["patched"] = True
            return patch_after_entry

        previous_trace = sys.gettrace()
        try:
            sys.settrace(patch_after_entry)
            with self.assertRaises(
                physical.V51BPhysicalRegistryError
            ) as captured:
                registry.lookup_execution_alias(admission, 2)
        finally:
            sys.settrace(previous_trace)
            type.__setattr__(
                private_type, "__getattr__", original_getattr
            )
        self.assertTrue(state["patched"])
        self.assertEqual(
            captured.exception.code,
            "DEPENDENCY_SUBSTITUTION",
        )
        self.assertEqual(trap.calls, 0)
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "INVALID_STATE",
        ):
            registry.stats()
        registry.abort()

    def test_preopen_dependency_perturbation_matrix_is_fail_closed(self):
        hashes = {
            "root_content_sha256": _sha("preopen-root"),
            "frame_content_sha256": _sha("preopen-frame"),
            "numeric_contract_sha256": _sha("preopen-numeric"),
            "implementation_sha256": _sha("preopen-implementation"),
        }
        for label, owner_globals, name in (
            _dependency_perturbation_cases()
        ):
            with self.subTest(dependency=label):
                topology = _shared_dense_topology()
                deadline = time.monotonic() + 60.0
                trap = _DependencyTrap(label)
                registry = None
                returned = None
                rejection = None
                existed, original = _install_perturbation(
                    owner_globals, name, trap
                )
                try:
                    try:
                        registry = (
                            physical.open_v51b_frame_physical_registry(
                                full_layers=topology.full_layers,
                                contexts=topology.contexts,
                                stage_uses=topology.stage_uses,
                                frame_bounds=topology.frame_bounds,
                                deadline=deadline,
                                **hashes,
                            )
                        )
                        admission = registry.admit_stage(
                            topology.stage_uses[0]
                        )
                        returned = registry.lookup_execution_alias(
                            admission, 2
                        )
                        registry.stats()
                    except physical.V51BPhysicalRegistryError as exc:
                        rejection = exc.code
                finally:
                    _restore_perturbation(
                        owner_globals,
                        name,
                        existed,
                        original,
                    )
                self.assertEqual(trap.calls, 0)
                if rejection is not None:
                    self.assertEqual(
                        rejection, "DEPENDENCY_SUBSTITUTION"
                    )
                else:
                    self.assertIs(returned, admission.aliases[0])
                if registry is not None:
                    registry.abort()

    def test_postopen_persistent_perturbation_matrix_is_inert(self):
        for label, owner_globals, name in (
            _dependency_perturbation_cases()
        ):
            with self.subTest(dependency=label):
                topology = _shared_dense_topology()
                registry = _open(
                    topology, frame_label=f"persistent-{label}"
                )
                admission = registry.admit_stage(
                    topology.stage_uses[0]
                )
                expected_alias = admission.aliases[0]
                trap = _DependencyTrap(label)
                existed, original = _install_perturbation(
                    owner_globals, name, trap
                )
                try:
                    returned = registry.lookup_execution_alias(
                        admission, 2
                    )
                    stats = registry.stats()
                finally:
                    _restore_perturbation(
                        owner_globals,
                        name,
                        existed,
                        original,
                    )
                self.assertEqual(trap.calls, 0)
                self.assertIs(returned, expected_alias)
                self.assertEqual(stats.execution_alias_lookups, 1)
                self.assertFalse(stats.proof_authority)
                registry.abort()

    def test_postopen_full_validation_perturbations_reject_before_call(
        self,
    ):
        for label, owner_globals, name in (
            _dependency_perturbation_cases()
        ):
            with self.subTest(dependency=label):
                topology = _shared_dense_topology()
                registry = _open(
                    topology, frame_label=f"full-gate-{label}"
                )
                trap = _DependencyTrap(label)
                rejection = None
                existed, original = _install_perturbation(
                    owner_globals, name, trap
                )
                try:
                    try:
                        registry.admit_stage(topology.stage_uses[0])
                    except physical.V51BPhysicalRegistryError as exc:
                        rejection = exc.code
                finally:
                    _restore_perturbation(
                        owner_globals,
                        name,
                        existed,
                        original,
                    )
                self.assertEqual(trap.calls, 0)
                self.assertEqual(
                    rejection, "DEPENDENCY_SUBSTITUTION"
                )
                registry.abort()

    def test_postopen_transient_aba_perturbation_matrix_is_inert(self):
        for label, owner_globals, name in (
            _dependency_perturbation_cases()
        ):
            with self.subTest(dependency=label):
                topology = _shared_dense_topology()
                registry = _open(topology, frame_label=f"aba-{label}")
                admission = registry.admit_stage(
                    topology.stage_uses[0]
                )
                expected_alias = admission.aliases[0]
                trap = _DependencyTrap(label)

                def run_aba(code_name, operation):
                    state = {"patched": False, "restored": False}
                    existed = None
                    original = None

                    def perturb(frame, event, arg):
                        nonlocal existed, original
                        del arg
                        if frame.f_code.co_name == code_name:
                            if event == "call" and not state["patched"]:
                                existed, original = (
                                    _install_perturbation(
                                        owner_globals, name, trap
                                    )
                                )
                                state["patched"] = True
                            elif (
                                event == "return"
                                and state["patched"]
                                and not state["restored"]
                            ):
                                _restore_perturbation(
                                    owner_globals,
                                    name,
                                    existed,
                                    original,
                                )
                                state["restored"] = True
                        return perturb

                    previous_trace = sys.gettrace()
                    try:
                        sys.settrace(perturb)
                        result = operation()
                    finally:
                        sys.settrace(previous_trace)
                        if state["patched"] and not state["restored"]:
                            _restore_perturbation(
                                owner_globals,
                                name,
                                existed,
                                original,
                            )
                            state["restored"] = True
                    self.assertTrue(state["patched"])
                    self.assertTrue(state["restored"])
                    return result

                returned = run_aba(
                    "lookup_execution_alias",
                    lambda: registry.lookup_execution_alias(
                        admission, 2
                    ),
                )
                stats = run_aba("stats", registry.stats)
                self.assertEqual(trap.calls, 0)
                self.assertIs(returned, expected_alias)
                self.assertEqual(stats.execution_alias_lookups, 1)
                self.assertFalse(stats.proof_authority)
                registry.abort()

    def test_exact_enumerate_bypass_repro_is_rejected_fast(self):
        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="enumerate-bypass-repro")
        admission = registry.admit_stage(topology.stage_uses[0])
        alias = admission.aliases[0]
        object.__setattr__(alias, "layer_id", 999)
        forged_calls = {"count": 0}

        def forged_enumerate(value):
            del value
            forged_calls["count"] += 1
            return ()

        prohibited = (
            (physical, "_raw_identity_snapshot"),
            (physical, "_root_manifest"),
            (physical, "_frame_bounds_manifest"),
            (physical, "_array_sha256"),
            (physical, "_json_sha256"),
            (physical, "_sealed_hmac"),
            (physical, "_derive_source_spec"),
            (physical, "_dense_numeric_core"),
            (physical, "_validate_dense_numeric_core"),
            (conv_v51, "_geometry"),
            (conv_v51, "_validate_plan"),
            (dense_v51, "_validate_support"),
        )
        probes = []
        python_codes, c_function_ids = _hot_path_profile_targets()
        profile_hits = {}
        physical_globals = vars(physical)
        existed, original = _install_perturbation(
            physical_globals, "enumerate", forged_enumerate
        )
        try:
            with ExitStack() as stack:
                for owner, name in prohibited:
                    probes.append(
                        stack.enter_context(
                            mock.patch.object(
                                owner,
                                name,
                                side_effect=AssertionError(
                                    f"repro invoked prohibited {name}"
                                ),
                            )
                        )
                    )
                previous_profile = sys.getprofile()
                try:
                    sys.setprofile(
                        _hot_path_profiler(
                            python_codes,
                            c_function_ids,
                            profile_hits,
                        )
                    )
                    with self.assertRaisesRegex(
                        physical.V51BPhysicalRegistryError,
                        "HANDLE_SEAL_MISMATCH",
                    ):
                        registry.lookup_execution_alias(admission, 2)
                finally:
                    sys.setprofile(previous_profile)
        finally:
            _restore_perturbation(
                physical_globals,
                "enumerate",
                existed,
                original,
            )
        self.assertEqual(forged_calls["count"], 0)
        self.assertTrue(all(probe.call_count == 0 for probe in probes))
        self.assertEqual(profile_hits, {})
        registry.abort()

    def test_open_rejects_external_subclasses_before_dynamic_reads(self):
        class StaticLayer(frozen._FrozenLayer):
            pass

        class DynamicLayer(frozen._FrozenLayer):
            reads = 0

            def __getattribute__(self, name):
                if name in {
                    "id",
                    "kind",
                    "preds",
                    "width",
                    "in_vars",
                    "out_vars",
                    "params",
                }:
                    type(self).reads += 1
                return super().__getattribute__(name)

        class StaticBox(frozen._Box):
            pass

        class DynamicBox(frozen._Box):
            reads = 0

            def __getattribute__(self, name):
                if name in {"lb", "ub"}:
                    type(self).reads += 1
                return super().__getattribute__(name)

        class StaticArray(np.ndarray):
            pass

        class DynamicArray(np.ndarray):
            reads = 0

            def __getattribute__(self, name):
                if name in {
                    "base",
                    "dtype",
                    "flags",
                    "ndim",
                    "shape",
                    "strides",
                }:
                    type(self).reads += 1
                return super().__getattribute__(name)

        class StaticDict(dict):
            pass

        class DynamicDict(dict):
            getter_reads = 0
            items_calls = 0
            iterator_calls = 0

            def __getattribute__(self, name):
                if name in {"items", "keys", "values"}:
                    type(self).getter_reads += 1
                return super().__getattribute__(name)

            def items(self):
                type(self).items_calls += 1
                return super().items()

            def __iter__(self):
                type(self).iterator_calls += 1
                return super().__iter__()

        class StaticCone(frozen._SealedCone):
            pass

        class DynamicCone(frozen._SealedCone):
            reads = 0

            def __getattribute__(self, name):
                if name in {
                    "start_lid",
                    "layers",
                    "reverse_order",
                    "output_id",
                    "output_width",
                    "start_mode",
                    "input_spec_id",
                    "replay_net_sha256",
                    "manifest_sha256",
                }:
                    type(self).reads += 1
                return super().__getattribute__(name)

        class StaticStageUse(authority.StageUse):
            pass

        class DynamicStageUse(authority.StageUse):
            reads = 0

            def __getattribute__(self, name):
                if name in {
                    "use_index",
                    "stage_kind",
                    "stage_index",
                    "target_relu_lid",
                    "cone_start_lid",
                    "stage_use_sha256",
                }:
                    type(self).reads += 1
                return super().__getattribute__(name)

        def expect_rejected(
            topology: _ControlledTopology,
            *,
            counter_type: type | None = None,
            counter_names: tuple[str, ...] = ("reads",),
        ) -> None:
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError, "RAW_EXACT_TYPE"
            ):
                _open(topology, frame_label="exact-type-rejection")
            if counter_type is not None:
                for counter_name in counter_names:
                    self.assertEqual(
                        getattr(counter_type, counter_name),
                        0,
                        f"{counter_type.__name__}.{counter_name}",
                    )

        for layer_type in (StaticLayer, DynamicLayer):
            topology = _five_cone_topology()
            source = topology.full_layers[2]
            replacement = layer_type(
                id=source.id,
                kind=source.kind,
                preds=source.preds,
                width=source.width,
                in_vars=source.in_vars,
                out_vars=source.out_vars,
                params=source.params,
            )
            if layer_type is DynamicLayer:
                DynamicLayer.reads = 0
            layers = dict(topology.full_layers)
            layers[2] = replacement
            expect_rejected(
                _ControlledTopology(
                    full_layers=MappingProxyType(layers),
                    contexts=topology.contexts,
                    stage_uses=topology.stage_uses,
                    frame_bounds=topology.frame_bounds,
                ),
                counter_type=(
                    DynamicLayer if layer_type is DynamicLayer else None
                ),
            )

        for box_type in (StaticBox, DynamicBox):
            topology = _five_cone_topology()
            source = topology.frame_bounds[1]
            replacement = box_type(lb=source.lb, ub=source.ub)
            if box_type is DynamicBox:
                DynamicBox.reads = 0
            bounds = dict(topology.frame_bounds)
            bounds[1] = replacement
            expect_rejected(
                _ControlledTopology(
                    full_layers=topology.full_layers,
                    contexts=topology.contexts,
                    stage_uses=topology.stage_uses,
                    frame_bounds=MappingProxyType(bounds),
                ),
                counter_type=(
                    DynamicBox if box_type is DynamicBox else None
                ),
            )

        for array_type in (StaticArray, DynamicArray):
            topology = _five_cone_topology()
            source = topology.frame_bounds[1]
            lower = np.asarray(source.lb).view(array_type)
            lower.setflags(write=False)
            if array_type is DynamicArray:
                DynamicArray.reads = 0
            bounds = dict(topology.frame_bounds)
            bounds[1] = frozen._Box(lb=lower, ub=source.ub)
            expect_rejected(
                _ControlledTopology(
                    full_layers=topology.full_layers,
                    contexts=topology.contexts,
                    stage_uses=topology.stage_uses,
                    frame_bounds=MappingProxyType(bounds),
                ),
                counter_type=(
                    DynamicArray if array_type is DynamicArray else None
                ),
            )

        for backing_type in (StaticDict, DynamicDict):
            topology = _five_cone_topology()
            backing = backing_type()
            dict.update(backing, dict(topology.full_layers))
            if backing_type is DynamicDict:
                DynamicDict.getter_reads = 0
                DynamicDict.items_calls = 0
                DynamicDict.iterator_calls = 0
            expect_rejected(
                _ControlledTopology(
                    full_layers=MappingProxyType(backing),
                    contexts=topology.contexts,
                    stage_uses=topology.stage_uses,
                    frame_bounds=topology.frame_bounds,
                ),
                counter_type=(
                    DynamicDict if backing_type is DynamicDict else None
                ),
                counter_names=(
                    "getter_reads",
                    "items_calls",
                    "iterator_calls",
                ),
            )

        for cone_type in (StaticCone, DynamicCone):
            topology = _five_cone_topology()
            source = topology.contexts[2]
            replacement = cone_type(
                start_lid=source.start_lid,
                layers=source.layers,
                reverse_order=source.reverse_order,
                output_id=source.output_id,
                output_width=source.output_width,
                start_mode=source.start_mode,
                input_spec_id=source.input_spec_id,
                replay_net_sha256=source.replay_net_sha256,
                manifest_sha256=source.manifest_sha256,
            )
            if cone_type is DynamicCone:
                DynamicCone.reads = 0
            contexts = dict(topology.contexts)
            contexts[2] = replacement
            expect_rejected(
                _ControlledTopology(
                    full_layers=topology.full_layers,
                    contexts=MappingProxyType(contexts),
                    stage_uses=topology.stage_uses,
                    frame_bounds=topology.frame_bounds,
                ),
                counter_type=(
                    DynamicCone if cone_type is DynamicCone else None
                ),
            )

        for use_type in (StaticStageUse, DynamicStageUse):
            topology = _five_cone_topology()
            source = topology.stage_uses[0]
            replacement = use_type(
                use_index=source.use_index,
                stage_kind=source.stage_kind,
                stage_index=source.stage_index,
                target_relu_lid=source.target_relu_lid,
                cone_start_lid=source.cone_start_lid,
            )
            if use_type is DynamicStageUse:
                DynamicStageUse.reads = 0
            expect_rejected(
                _ControlledTopology(
                    full_layers=topology.full_layers,
                    contexts=topology.contexts,
                    stage_uses=(
                        replacement,
                        *topology.stage_uses[1:],
                    ),
                    frame_bounds=topology.frame_bounds,
                ),
                counter_type=(
                    DynamicStageUse
                    if use_type is DynamicStageUse
                    else None
                ),
            )

    def test_open_rejects_nonexact_critical_geometry_scalars(self):
        class IntSubclass(int):
            pass

        class StringSubclass(str):
            pass

        class TupleSubclass(tuple):
            pass

        replacements = (
            ("groups", True),
            ("groups", IntSubclass(1)),
            ("stride", TupleSubclass((1, 1))),
        )
        for parameter, replacement in replacements:
            with self.subTest(parameter=parameter, value=type(replacement)):
                topology = _five_cone_topology()
                source = topology.full_layers[2]
                params = dict(source.params)
                params[parameter] = replacement
                layers = dict(topology.full_layers)
                layers[2] = frozen._FrozenLayer(
                    id=source.id,
                    kind=source.kind,
                    preds=source.preds,
                    width=source.width,
                    in_vars=source.in_vars,
                    out_vars=source.out_vars,
                    params=MappingProxyType(params),
                )
                with self.assertRaisesRegex(
                    physical.V51BPhysicalRegistryError,
                    "RAW_EXACT_TYPE",
                ):
                    _open(
                        _ControlledTopology(
                            full_layers=MappingProxyType(layers),
                            contexts=topology.contexts,
                            stage_uses=topology.stage_uses,
                            frame_bounds=topology.frame_bounds,
                        ),
                        frame_label="nonexact-geometry",
                    )

        topology = _five_cone_topology()
        source = topology.full_layers[2]
        layers = dict(topology.full_layers)
        layers[2] = frozen._FrozenLayer(
            id=source.id,
            kind=StringSubclass(source.kind),
            preds=source.preds,
            width=source.width,
            in_vars=source.in_vars,
            out_vars=source.out_vars,
            params=source.params,
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError, "RAW_EXACT_TYPE"
        ):
            _open(
                _ControlledTopology(
                    full_layers=MappingProxyType(layers),
                    contexts=topology.contexts,
                    stage_uses=topology.stage_uses,
                    frame_bounds=topology.frame_bounds,
                ),
                frame_label="nonexact-string",
            )

    def test_controlled_five_cone_exact_counts_and_validator_boundary(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        admissions = _admit_all(registry, topology)
        self.assertEqual(
            tuple(len(value.aliases) for value in admissions),
            (1, 2, 4, 6, 7),
        )

        aliases = tuple(
            alias
            for admission in admissions
            for alias in admission.aliases
        )
        self.assertEqual(len(aliases), 20)
        self.assertEqual(len({id(value) for value in aliases}), 20)
        self.assertEqual(
            len({id(value.physical_core) for value in aliases}), 7
        )
        self.assertEqual(
            len(
                {
                    value.physical_core.physical_key_sha256
                    for value in aliases
                }
            ),
            7,
        )
        for alias in aliases:
            expected_schema = (
                physical.DENSE_STAGE_DIAGNOSTIC_SCHEMA
                if alias.physical_core.operator_branch
                == physical.BRANCH_DENSE
                else physical.CONV_STAGE_DIAGNOSTIC_SCHEMA
            )
            self.assertEqual(
                alias.stage_diagnostic_schema, expected_schema
            )

        dense_aliases = tuple(
            value for value in aliases if value.layer_id == 14
        )
        self.assertEqual(len(dense_aliases), 1)
        layer_two_aliases = tuple(
            value for value in aliases if value.layer_id == 2
        )
        self.assertEqual(len(layer_two_aliases), 5)
        self.assertEqual(
            len(
                {
                    value.stage_diagnostic_sha256
                    for value in layer_two_aliases
                }
            ),
            5,
        )

        for admission in admissions:
            for alias in admission.aliases:
                returned = registry.lookup_execution_alias(
                    admission, alias.layer_id
                )
                self.assertIs(returned, alias)
        for alias in admissions[-1].aliases:
            registry.lookup_execution_alias(
                admissions[-1], alias.layer_id
            )
        before_commit = registry.stats()
        self.assertEqual(before_commit.physical_builds, 7)
        self.assertEqual(before_commit.conv_physical_builds, 6)
        self.assertEqual(before_commit.dense_physical_builds, 1)
        self.assertEqual(before_commit.stage_aliases, 20)
        self.assertEqual(before_commit.cross_stage_physical_hits, 13)
        self.assertEqual(before_commit.execution_alias_lookups, 27)
        self.assertEqual(before_commit.admission_full_validations, 7)
        self.assertEqual(
            before_commit.private_execution_full_validations, 0
        )
        self.assertEqual(before_commit.commit_full_validations, 0)
        object.__setattr__(before_commit, "physical_builds", 700)

        certificate = registry.commit()

        self.assertFalse(certificate.proof_authority)
        self.assertEqual(certificate.physical_builds, 7)
        self.assertEqual(certificate.conv_physical_builds, 6)
        self.assertEqual(certificate.dense_physical_builds, 1)
        self.assertEqual(certificate.stage_aliases, 20)
        self.assertEqual(certificate.cross_stage_physical_hits, 13)
        self.assertEqual(certificate.execution_alias_lookups, 27)
        self.assertEqual(certificate.admission_full_validations, 7)
        self.assertEqual(
            certificate.private_execution_full_validations, 0
        )
        self.assertEqual(certificate.commit_full_validations, 7)
        self.assertFalse(certificate.receipt["proof_authority"])
        dependency_digest = certificate.receipt[
            "dependency_implementation_sha256"
        ]
        self.assertEqual(len(dependency_digest), 64)
        self.assertEqual(
            dependency_digest,
            physical._json_sha256(
                certificate.receipt["dependency_implementation"]
            ),
        )
        runtime_seal = certificate.receipt[
            "dependency_implementation"
        ]["module_view_runtime_seal"]
        self.assertTrue(
            runtime_seal[
                "persistent_class_namespace_substitution_rejected"
            ]
        )
        self.assertTrue(
            runtime_seal[
                "operation_entry_and_publication_fingerprint"
            ]
        )
        self.assertFalse(
            runtime_seal[
                "transient_change_dispatch_restore_cycle_closed"
            ]
        )
        self.assertFalse(runtime_seal["proof_authority"])

    def test_all_twenty_seven_lookups_are_strict_o1_alias_checks(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="strict-o1-lookups")
        admissions = _admit_all(registry, topology)
        before = registry.stats()
        self.assertEqual(before.admission_full_validations, 7)
        self.assertEqual(before.private_execution_full_validations, 0)
        prohibited = (
            (physical, "_raw_identity_snapshot"),
            (physical, "_root_manifest"),
            (physical, "_frame_bounds_manifest"),
            (physical, "_array_sha256"),
            (physical, "_json_sha256"),
            (physical, "_json_bytes"),
            (physical, "_sealed_hmac"),
            (physical, "_dense_numeric_core"),
            (physical, "_validate_dense_numeric_core"),
            (physical, "_derive_source_spec"),
            (physical, "_module_binding_snapshot"),
            (physical, "_module_function_manifest"),
            (frozen, "_layer_manifest"),
            (conv_v51, "_geometry"),
            (conv_v51, "_validate_plan"),
            (dense_v51, "_validate_support"),
        )
        probes = []
        with ExitStack() as stack:
            for owner, name in prohibited:
                probes.append(
                    stack.enter_context(
                        mock.patch.object(
                            owner,
                            name,
                            side_effect=AssertionError(
                                f"lookup invoked prohibited {name}"
                            ),
                        )
                    )
                )
            for admission in admissions:
                for alias in admission.aliases:
                    self.assertIs(
                        registry.lookup_execution_alias(
                            admission, alias.layer_id
                        ),
                        alias,
                    )
            for alias in admissions[-1].aliases:
                self.assertIs(
                    registry.lookup_execution_alias(
                        admissions[-1], alias.layer_id
                    ),
                    alias,
                )
        self.assertTrue(all(probe.call_count == 0 for probe in probes))
        after = registry.stats()
        self.assertEqual(after.execution_alias_lookups, 27)
        self.assertEqual(after.admission_full_validations, 7)
        self.assertEqual(after.private_execution_full_validations, 0)
        registry.abort()

    def test_one_hundred_thousand_lookup_microbenchmark_stays_fast_only(self):
        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="lookup-100k-micro")
        admission = registry.admit_stage(topology.stage_uses[0])
        expected_alias = admission.aliases[0]
        prohibited = (
            (physical, "_raw_identity_snapshot"),
            (physical, "_root_manifest"),
            (physical, "_frame_bounds_manifest"),
            (physical, "_array_sha256"),
            (physical, "_json_sha256"),
            (physical, "_sealed_hmac"),
            (physical, "_derive_source_spec"),
            (conv_v51, "_geometry"),
            (conv_v51, "_validate_plan"),
            (dense_v51, "_validate_support"),
        )
        probes = []
        python_codes, c_function_ids = _hot_path_profile_targets()
        profile_hits = {}
        started = time.perf_counter()
        with ExitStack() as stack:
            for owner, name in prohibited:
                probes.append(
                    stack.enter_context(
                        mock.patch.object(
                            owner,
                            name,
                            side_effect=AssertionError(
                                f"100k lookup invoked prohibited {name}"
                            ),
                        )
                    )
                )
            previous_profile = sys.getprofile()
            try:
                sys.setprofile(
                    _hot_path_profiler(
                        python_codes,
                        c_function_ids,
                        profile_hits,
                    )
                )
                for _ in range(100_000):
                    self.assertIs(
                        registry.lookup_execution_alias(admission, 2),
                        expected_alias,
                    )
            finally:
                sys.setprofile(previous_profile)
        elapsed = time.perf_counter() - started
        self.assertGreater(elapsed, 0.0)
        self.assertTrue(all(probe.call_count == 0 for probe in probes))
        self.assertEqual(profile_hits, {})
        stats = registry.stats()
        self.assertEqual(stats.execution_alias_lookups, 100_000)
        self.assertEqual(stats.admission_full_validations, 1)
        self.assertEqual(stats.private_execution_full_validations, 0)
        registry.abort()

    def test_commit_derives_exactly_seven_unique_physical_keys(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="seven-commit-keys")
        _admit_all(registry, topology)
        certificate = registry.commit()
        self.assertEqual(certificate.physical_builds, 7)
        self.assertEqual(
            len(certificate.receipt["physical_key_sha256"]), 7
        )
        self.assertEqual(
            certificate.receipt["commit_physical_derivations"], 7
        )

    def test_post_open_dependency_replacement_fails_before_dispatch(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="dependency-anchors")
        original_conv_prepare = conv_v51.prepare_dense_conv_v51_plan

        def corrupt_conv_plan(*args, **kwargs):
            plan = original_conv_prepare(*args, **kwargs)
            object.__setattr__(plan, "groups", plan.groups + 1)
            return plan

        def accept_corrupt_conv_plan(*args, **kwargs):
            del args, kwargs
            return None

        with mock.patch.object(
            conv_v51,
            "prepare_dense_conv_v51_plan",
            side_effect=corrupt_conv_plan,
        ) as corrupt_prepare, mock.patch.object(
            conv_v51,
            "_validate_plan",
            side_effect=accept_corrupt_conv_plan,
        ) as no_op_validator, mock.patch.object(
            dense_v51,
            "prepare_dense_support_v51",
            side_effect=AssertionError(
                "post-open malicious Dense prepare was called"
            ),
        ) as dense_prepare, mock.patch.object(
            dense_v51,
            "_validate_support",
            side_effect=AssertionError(
                "post-open malicious Dense validator was called"
            ),
        ) as dense_validator:
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                registry.admit_stage(topology.stage_uses[0])
        self.assertEqual(
            (
                corrupt_prepare.call_count,
                no_op_validator.call_count,
                dense_prepare.call_count,
                dense_validator.call_count,
            ),
            (0, 0, 0, 0),
        )
        clean_registry = _open(
            topology, frame_label="dependency-anchor-receipt"
        )
        _admit_all(clean_registry, topology)
        certificate = clean_registry.commit()
        self.assertEqual(certificate.physical_builds, 7)
        self.assertEqual(certificate.commit_full_validations, 7)
        dependency_manifest = certificate.receipt[
            "dependency_implementation"
        ]
        self.assertEqual(
            certificate.receipt[
                "dependency_implementation_sha256"
            ],
            physical._json_sha256(dependency_manifest),
        )
        self.assertIn("conv_prepare", dependency_manifest["callable"])
        self.assertIn("conv_validate", dependency_manifest["callable"])
        function_manifest = dependency_manifest[
            "module_function_implementation_sha256"
        ]
        self.assertIn(
            "_dot_up_l_matrix",
            function_manifest["query_dual_replay_v51_conv"],
        )
        self.assertIn(
            "check_v51_platform",
            function_manifest["query_dual_scalar_guard_v51"],
        )
        self.assertIn(
            "_json_sha256",
            function_manifest["query_dual_v51_authority"],
        )
        self.assertIn(
            "validate_stage_use",
            function_manifest["query_dual_v51_authority"],
        )
        self.assertIn(
            "_derive_source_spec",
            function_manifest[
                "query_dual_v51b_physical_registry"
            ],
        )
        self.assertIn(
            "_array_sha256",
            function_manifest[
                "query_dual_v51b_physical_registry"
            ],
        )

    def test_factory_captures_exact_private_dependency_functions(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="private-function-capture")
        port_method = type(registry).admit_stage
        port_cells = dict(
            zip(
                port_method.__code__.co_freevars,
                (
                    cell.cell_contents
                    for cell in port_method.__closure__
                ),
            )
        )
        admit_impl = port_cells["admit_impl"]
        admit_cells = dict(
            zip(
                admit_impl.__code__.co_freevars,
                (
                    cell.cell_contents
                    for cell in admit_impl.__closure__
                ),
            )
        )
        build_core = admit_cells["build_core"]
        build_cells = dict(
            zip(
                build_core.__code__.co_freevars,
                (
                    cell.cell_contents
                    for cell in build_core.__closure__
                ),
            )
        )
        dense_prepare = build_cells["dense_prepare"]
        private_platform_check = dense_prepare.__globals__[
            "check_v51_platform"
        ]
        fresh_spec = admit_cells["fresh_spec"]
        fresh_cells = dict(
            zip(
                fresh_spec.__code__.co_freevars,
                (
                    cell.cell_contents
                    for cell in fresh_spec.__closure__
                ),
            )
        )
        private_deriver = fresh_cells["derive_source_spec"]
        self.assertIs(type(dense_prepare), FunctionType)
        self.assertIs(type(private_platform_check), FunctionType)
        self.assertIsNot(
            private_platform_check, dense_v51.check_v51_platform
        )
        self.assertIs(type(private_deriver), FunctionType)
        self.assertIsNot(private_deriver.__globals__, vars(physical))
        self.assertIsNot(
            private_deriver.__globals__["_array_sha256"],
            physical._array_sha256,
        )
        registry.abort()

    def test_transitive_conv_helper_replacement_is_fail_closed(self):
        topology = _five_cone_topology()
        forged_calls = {"count": 0}

        def forged_zero_support(left, right, *, deadline):
            forged_calls["count"] += 1
            deadline.check(force=True)
            return (
                np.zeros(
                    (left.shape[0], right.shape[1]),
                    dtype=np.float64,
                ),
                np.zeros(
                    (left.shape[0], right.shape[1]),
                    dtype=np.bool_,
                ),
            )

        admission_registry = _open(
            topology, frame_label="transitive-helper-admission"
        )
        with mock.patch.object(
            conv_v51,
            "_dot_up_l_matrix",
            side_effect=forged_zero_support,
        ):
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                admission_registry.admit_stage(topology.stage_uses[0])
        self.assertEqual(forged_calls["count"], 0)

        commit_registry = _open(
            topology, frame_label="transitive-helper-commit"
        )
        _admit_all(commit_registry, topology)
        with mock.patch.object(
            conv_v51,
            "_dot_up_l_matrix",
            side_effect=forged_zero_support,
        ):
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                commit_registry.commit()
        self.assertEqual(forged_calls["count"], 0)

    def test_transient_helper_aba_cannot_enter_isolated_execution(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="transient-helper-aba")
        original_helper = conv_v51._dot_up_l_matrix
        state = {"patched": False, "restored": False, "forged": 0}

        def forged_zero_support(left, right, *, deadline):
            del left, right, deadline
            state["forged"] += 1
            raise AssertionError("source-module forged helper executed")

        def helper_aba(frame, event, arg):
            del arg
            if (
                frame.f_code.co_name == "_dot_up_l_matrix"
                and frame.f_globals is not vars(conv_v51)
            ):
                if event == "call" and not state["patched"]:
                    conv_v51._dot_up_l_matrix = forged_zero_support
                    state["patched"] = True
                elif (
                    event == "line"
                    and state["patched"]
                    and not state["restored"]
                ):
                    conv_v51._dot_up_l_matrix = original_helper
                    state["restored"] = True
            return helper_aba

        previous_trace = sys.gettrace()
        try:
            sys.settrace(helper_aba)
            admission = registry.admit_stage(topology.stage_uses[0])
        finally:
            sys.settrace(previous_trace)
            conv_v51._dot_up_l_matrix = original_helper
        self.assertEqual(len(admission.aliases), 1)
        self.assertTrue(state["patched"])
        self.assertTrue(state["restored"])
        self.assertEqual(state["forged"], 0)
        registry.abort()

    def test_dense_and_frozen_transitive_helpers_are_fail_closed(self):
        cases = (
            (
                dense_v51,
                "_dot_up_longdouble_unchecked",
                _shared_dense_topology,
            ),
            (
                frozen,
                "_array_digest",
                _five_cone_topology,
            ),
        )
        for owner, helper_name, topology_factory in cases:
            with self.subTest(
                module=owner.__name__, helper=helper_name
            ):
                topology = topology_factory()
                registry = _open(
                    topology,
                    frame_label=f"transitive-{helper_name}",
                )
                forged_calls = {"count": 0}

                def forged_helper(*args, **kwargs):
                    del args, kwargs
                    forged_calls["count"] += 1
                    raise AssertionError(
                        f"forged transitive helper {helper_name} executed"
                    )

                with mock.patch.object(
                    owner, helper_name, side_effect=forged_helper
                ):
                    with self.assertRaisesRegex(
                        physical.V51BPhysicalRegistryError,
                        "DEPENDENCY_SUBSTITUTION",
                    ):
                        registry.admit_stage(topology.stage_uses[0])
                self.assertEqual(forged_calls["count"], 0)

    def test_conv_and_frozen_exact_function_forgery_is_fail_closed(self):
        cases = (
            (conv_v51, "_dot_up_l_matrix"),
            (frozen, "_array_digest"),
        )
        for owner, helper_name in cases:
            with self.subTest(
                module=owner.__name__, helper=helper_name
            ):
                topology = _five_cone_topology()
                registry = _open(
                    topology,
                    frame_label=f"exact-function-{helper_name}",
                )
                forged_calls = {"count": 0}

                if owner is conv_v51:
                    def forged_code(
                        left,
                        right,
                        *,
                        deadline,
                        _counter=forged_calls,
                    ):
                        del left, right, deadline
                        _counter["count"] += 1
                        raise AssertionError(
                            "forged exact Conv helper executed"
                        )
                else:
                    def forged_code(value, _counter=forged_calls):
                        del value
                        _counter["count"] += 1
                        raise AssertionError(
                            "forged exact frozen helper executed"
                        )

                replacement = FunctionType(
                    forged_code.__code__,
                    vars(owner),
                    forged_code.__name__,
                    forged_code.__defaults__,
                    None,
                )
                replacement.__kwdefaults__ = (
                    None
                    if forged_code.__kwdefaults__ is None
                    else dict(forged_code.__kwdefaults__)
                )
                self.assertIs(type(replacement), FunctionType)
                self.assertIs(replacement.__globals__, vars(owner))
                original = getattr(owner, helper_name)
                try:
                    setattr(owner, helper_name, replacement)
                    with self.assertRaisesRegex(
                        physical.V51BPhysicalRegistryError,
                        "DEPENDENCY_SUBSTITUTION",
                    ):
                        registry.admit_stage(topology.stage_uses[0])
                finally:
                    setattr(owner, helper_name, original)
                self.assertEqual(forged_calls["count"], 0)

    def test_dense_platform_lru_wrapper_has_no_live_helper_path(self):
        topology = _shared_dense_topology()
        registry = _open(
            topology, frame_label="dense-platform-wrapper-isolation"
        )
        dense_v51.check_v51_platform.cache_clear()
        forged_calls = {"count": 0}

        def forged_longdouble_text(value):
            del value
            forged_calls["count"] += 1
            raise AssertionError("public Dense platform helper executed")

        with mock.patch.object(
            dense_v51,
            "_longdouble_text",
            side_effect=forged_longdouble_text,
        ):
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                registry.admit_stage(topology.stage_uses[0])
        self.assertEqual(forged_calls["count"], 0)

    def test_authority_stage_use_copy_never_calls_public_post_init(self):
        topology = _five_cone_topology()

        def forbidden_post_init(value):
            del value
            raise AssertionError("public StageUse.__post_init__ executed")

        with mock.patch.object(
            authority.StageUse,
            "__post_init__",
            autospec=True,
            side_effect=forbidden_post_init,
        ) as post_init:
            registry = _open(
                topology, frame_label="constructorless-stage-use"
            )
            _admit_all(registry, topology)
            certificate = registry.commit()
        self.assertEqual(post_init.call_count, 0)
        self.assertEqual(certificate.physical_builds, 7)
        self.assertEqual(certificate.commit_full_validations, 7)

    def test_authority_helper_replacement_is_fail_closed(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="authority-helper-gate")
        forged_calls = {"count": 0}

        def forged_json_sha256(value):
            del value
            forged_calls["count"] += 1
            return "0" * 64

        with mock.patch.object(
            authority,
            "_json_sha256",
            side_effect=forged_json_sha256,
        ):
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                registry.admit_stage(topology.stage_uses[0])
        self.assertEqual(forged_calls["count"], 0)

    def test_physical_key_helper_replacement_is_fail_closed(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="physical-helper-gate")
        forged_calls = {"count": 0}

        def forged_array_sha256(value):
            del value
            forged_calls["count"] += 1
            return "0" * 64

        with mock.patch.object(
            physical,
            "_array_sha256",
            side_effect=forged_array_sha256,
        ):
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                registry.admit_stage(topology.stage_uses[0])
        self.assertEqual(forged_calls["count"], 0)

    def test_physical_key_helper_aba_cannot_enter_isolated_derivation(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="physical-helper-aba")
        source_globals = vars(physical)
        original_helper = physical._array_sha256
        previous_trace = sys.gettrace()
        state = {"patched": False, "restored": False, "forged": 0}

        def forged_array_sha256(value):
            del value
            state["forged"] += 1
            raise AssertionError("public physical hash helper executed")

        def trace_derive(frame, event, arg):
            del arg
            if (
                frame.f_code.co_name == "_derive_source_spec"
                and frame.f_globals is not source_globals
            ):
                if event == "call" and not state["patched"]:
                    physical._array_sha256 = forged_array_sha256
                    state["patched"] = True
                elif event == "return" and state["patched"]:
                    physical._array_sha256 = original_helper
                    state["restored"] = True
            return trace_derive

        try:
            sys.settrace(trace_derive)
            admission = registry.admit_stage(topology.stage_uses[0])
        finally:
            sys.settrace(previous_trace)
            physical._array_sha256 = original_helper
        self.assertEqual(len(admission.aliases), 1)
        self.assertTrue(state["patched"])
        self.assertTrue(state["restored"])
        self.assertEqual(state["forged"], 0)
        registry.abort()

    def test_commit_rebuilds_dense_support_from_raw_math(self):
        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="dense-fresh-rebuild")
        _admit_all(registry, topology)
        cores = _registry_core_records(registry)
        record = next(iter(cores.values()))
        support = record.material
        forged_binding = tuple(
            sorted((*support.binding, ("forged_binding", "yes")))
        )
        diagnostic_values = dict(support.diagnostics.items)
        diagnostic_values["binding_sha256"] = (
            dense_v51._canonical_digest(forged_binding)
        )
        diagnostic_items = tuple(sorted(diagnostic_values.items()))
        forged_diagnostics = dense_v51.V51Diagnostics(
            items=diagnostic_items,
            sha256=dense_v51._canonical_digest(diagnostic_items),
        )
        forged_support = replace(
            support,
            binding=forged_binding,
            diagnostics=forged_diagnostics,
        )
        dense_v51._validate_support(
            forged_support,
            np.asarray(record.spec.layer.params["weight"]),
            platform_sha256=diagnostic_values["platform_sha256"],
        )
        record.material = forged_support
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "MATERIAL_SUBSTITUTION",
        ):
            registry.commit()

    def test_dense_numeric_core_is_shared_but_stage_diagnostics_are_not(self):
        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="shared-dense-frame")
        target = registry.admit_stage(topology.stage_uses[0])
        self.assertEqual(registry.stats().admission_full_validations, 1)
        property_stage = registry.admit_stage(topology.stage_uses[1])
        after_hit = registry.stats()
        self.assertEqual(after_hit.physical_builds, 1)
        self.assertEqual(after_hit.admission_full_validations, 1)
        self.assertEqual(after_hit.cross_stage_physical_hits, 1)
        admissions = (target, property_stage)
        target_alias = admissions[0].aliases[0]
        property_alias = admissions[1].aliases[0]
        self.assertIs(
            target_alias.physical_core, property_alias.physical_core
        )
        self.assertNotEqual(
            target_alias.stage_diagnostic_sha256,
            property_alias.stage_diagnostic_sha256,
        )
        self.assertEqual(
            target_alias.stage_diagnostic_schema,
            physical.DENSE_STAGE_DIAGNOSTIC_SCHEMA,
        )
        before_commit = registry.stats()
        self.assertEqual(before_commit.physical_builds, 1)
        self.assertEqual(before_commit.dense_physical_builds, 1)
        self.assertEqual(before_commit.stage_aliases, 2)
        self.assertEqual(before_commit.cross_stage_physical_hits, 1)
        certificate = registry.commit()
        self.assertEqual(certificate.commit_full_validations, 1)

    def test_second_frame_has_no_physical_handle_or_key_sharing(self):
        topology = _five_cone_topology()
        first = _open(topology, frame_label="first-frame")
        second = _open(topology, frame_label="second-frame")
        first_admissions = _admit_all(first, topology)
        second_admissions = _admit_all(second, topology)
        first_by_layer = {
            alias.layer_id: alias.physical_core
            for admission in first_admissions
            for alias in admission.aliases
        }
        second_by_layer = {
            alias.layer_id: alias.physical_core
            for admission in second_admissions
            for alias in admission.aliases
        }
        self.assertEqual(set(first_by_layer), set(second_by_layer))
        for layer_id in first_by_layer:
            self.assertIsNot(
                first_by_layer[layer_id], second_by_layer[layer_id]
            )
            self.assertNotEqual(
                first_by_layer[layer_id].physical_key_sha256,
                second_by_layer[layer_id].physical_key_sha256,
            )
        first_certificate = first.commit()
        second_certificate = second.commit()
        self.assertEqual(first_certificate.physical_builds, 7)
        self.assertEqual(second_certificate.physical_builds, 7)
        self.assertEqual(
            first_certificate.receipt[
                "dependency_implementation_sha256"
            ],
            second_certificate.receipt[
                "dependency_implementation_sha256"
            ],
        )

    def test_open_to_first_admission_raw_toctou_is_rejected(self):
        root_topology = _five_cone_topology()
        root_registry = _open(
            root_topology, frame_label="root-toctou-frame"
        )
        layer = root_topology.full_layers[2]
        changed = dict(layer.params)
        changed["weight"] = _f64(
            np.asarray([[[[1.75]]]], dtype=np.float64),
            name="pre-admission substituted weight",
        )
        object.__setattr__(layer, "params", MappingProxyType(changed))
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "RAW_CONTEXT_MISMATCH",
        ):
            root_registry.admit_stage(root_topology.stage_uses[0])

        box_topology = _five_cone_topology()
        box_registry = _open(
            box_topology, frame_label="box-toctou-frame"
        )
        object.__setattr__(
            box_topology.frame_bounds[1],
            "ub",
            _f64(
                np.asarray([0.5], dtype=np.float64),
                name="pre-admission substituted upper",
            ),
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "RAW_CONTEXT_MISMATCH",
        ):
            box_registry.admit_stage(box_topology.stage_uses[0])

    def test_mappingproxy_backing_mutation_is_rejected_on_fast_hit(self):
        topology = _five_cone_topology()
        layer = topology.full_layers[2]
        retained_backing = dict(layer.params)
        object.__setattr__(
            layer, "params", MappingProxyType(retained_backing)
        )
        registry = _open(topology, frame_label="backing-mutation-frame")
        registry.admit_stage(topology.stage_uses[0])
        retained_backing["weight"] = _f64(
            np.asarray([[[[2.0]]]], dtype=np.float64),
            name="mapping backing substituted weight",
        )
        with mock.patch.object(
            physical,
            "_array_sha256",
            side_effect=AssertionError(
                "identity gate should reject before array hashing"
            ),
        ):
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "RAW_CONTEXT_MISMATCH",
            ):
                registry.admit_stage(topology.stage_uses[1])

    def test_invalid_frame_level_stage_use_sets_are_rejected(self):
        topology = _shared_dense_topology()
        two_properties = (
            authority.StageUse(
                use_index=0,
                stage_kind=authority.STAGE_PROPERTY,
                stage_index=None,
                target_relu_lid=None,
                cone_start_lid=None,
            ),
            authority.StageUse(
                use_index=1,
                stage_kind=authority.STAGE_PROPERTY,
                stage_index=None,
                target_relu_lid=None,
                cone_start_lid=None,
            ),
        )
        invalid_properties = _ControlledTopology(
            full_layers=topology.full_layers,
            contexts=topology.contexts,
            stage_uses=two_properties,
            frame_bounds=topology.frame_bounds,
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError, "INVALID_STAGE_USE"
        ):
            _open(invalid_properties, frame_label="two-properties")

        five = _five_cone_topology()
        duplicate_target = authority.StageUse(
            use_index=1,
            stage_kind=authority.STAGE_TARGET,
            stage_index=0,
            target_relu_lid=5,
            cone_start_lid=4,
        )
        invalid_targets = _ControlledTopology(
            full_layers=five.full_layers,
            contexts=five.contexts,
            stage_uses=(
                five.stage_uses[0],
                duplicate_target,
                *five.stage_uses[2:],
            ),
            frame_bounds=five.frame_bounds,
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError, "INVALID_STAGE_USE"
        ):
            _open(invalid_targets, frame_label="duplicate-target-index")

    def test_open_recomputes_stage_use_sha_without_authority_helper(self):
        topology = _shared_dense_topology()
        object.__setattr__(
            topology.stage_uses[0], "stage_use_sha256", "0" * 64
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError, "INVALID_STAGE_USE"
        ):
            _open(topology, frame_label="forged-stage-use-sha")

    def test_coordinated_public_handle_replacement_is_rejected(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        first = registry.admit_stage(topology.stage_uses[0])
        second = registry.admit_stage(topology.stage_uses[1])
        victim = first.aliases[0]
        replacement = second.aliases[0]
        # Replace every visible and hidden dataclass field with a correctly
        # sealed different alias.  The closure-owned occurrence identity must
        # still reject it.
        for name in (
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
        ):
            object.__setattr__(victim, name, getattr(replacement, name))
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "HANDLE_SEAL_MISMATCH",
        ):
            registry.lookup_execution_alias(first, 2)

    def test_every_public_handle_slot_replacement_is_rejected(self):
        fields_by_kind = {
            "admission": (
                "use_index",
                "stage_use_sha256",
                "aliases",
                "content_sha256",
                "proof_authority",
                "_token",
                "_seal",
                "_capability",
            ),
            "alias": (
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
            "core": (
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
        }
        for kind, field_names in fields_by_kind.items():
            for field_name in field_names:
                with self.subTest(kind=kind, field=field_name):
                    topology = _shared_dense_topology()
                    registry = _open(
                        topology,
                        frame_label=f"slot-{kind}-{field_name}",
                    )
                    admission = registry.admit_stage(
                        topology.stage_uses[0]
                    )
                    alias = admission.aliases[0]
                    values = {
                        "admission": admission,
                        "alias": alias,
                        "core": alias.physical_core,
                    }
                    victim = values[kind]
                    current = getattr(victim, field_name)
                    object.__setattr__(
                        victim,
                        field_name,
                        _replacement_field_value(current),
                    )
                    with self.assertRaisesRegex(
                        physical.V51BPhysicalRegistryError,
                        "HANDLE_SEAL_MISMATCH",
                    ):
                        registry.lookup_execution_alias(admission, 2)

    def test_boundary_class_property_and_method_substitution_is_rejected(self):
        cases = (
            (physical.StageAdmission, "use_index"),
            (physical.StageAliasHandle, "alias_content_sha256"),
            (physical.PhysicalCoreHandle, "physical_key_sha256"),
        )
        for boundary_type, field_name in cases:
            with self.subTest(
                boundary_type=boundary_type.__name__, field=field_name
            ):
                topology = _shared_dense_topology()
                registry = _open(
                    topology,
                    frame_label=(
                        f"class-property-{boundary_type.__name__}"
                    ),
                )
                admission = registry.admit_stage(
                    topology.stage_uses[0]
                )
                original = type.__getattribute__(
                    boundary_type, "__dict__"
                )[field_name]
                try:
                    type.__setattr__(
                        boundary_type,
                        field_name,
                        property(lambda value: 0),
                    )
                    with self.assertRaisesRegex(
                        physical.V51BPhysicalRegistryError,
                        "PUBLIC_ABI_SUBSTITUTION",
                    ):
                        registry.lookup_execution_alias(admission, 2)
                finally:
                    type.__setattr__(
                        boundary_type, field_name, original
                    )

        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="class-method-substitution")
        admission = registry.admit_stage(topology.stage_uses[0])
        boundary_type = physical.StageAliasHandle
        had_original = "__getattribute__" in type.__getattribute__(
            boundary_type, "__dict__"
        )
        original = type.__getattribute__(
            boundary_type, "__dict__"
        ).get("__getattribute__")

        def forged_getattribute(value, name):
            return object.__getattribute__(value, name)

        try:
            type.__setattr__(
                boundary_type,
                "__getattribute__",
                forged_getattribute,
            )
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "PUBLIC_ABI_SUBSTITUTION",
            ):
                registry.lookup_execution_alias(admission, 2)
        finally:
            if had_original:
                type.__setattr__(
                    boundary_type, "__getattribute__", original
                )
            else:
                type.__delattr__(boundary_type, "__getattribute__")

    def test_transient_class_descriptor_aba_cannot_affect_slot_authority(self):
        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="class-descriptor-aba")
        admission = registry.admit_stage(topology.stage_uses[0])
        expected_alias = admission.aliases[0]
        boundary_type = physical.StageAdmission
        original = type.__getattribute__(
            boundary_type, "__dict__"
        )["use_index"]
        previous_trace = sys.gettrace()
        state = {"patched": False, "restored": False, "forged": 0}

        def forged_use_index(value):
            del value
            state["forged"] += 1
            return 999

        def descriptor_aba(frame, event, arg):
            del arg
            if frame.f_code.co_name == "boundary_values":
                if event == "call" and not state["patched"]:
                    type.__setattr__(
                        boundary_type,
                        "use_index",
                        property(forged_use_index),
                    )
                    state["patched"] = True
                elif event == "return" and state["patched"]:
                    type.__setattr__(
                        boundary_type, "use_index", original
                    )
                    state["restored"] = True
            return descriptor_aba

        try:
            sys.settrace(descriptor_aba)
            returned = registry.lookup_execution_alias(admission, 2)
        finally:
            sys.settrace(previous_trace)
            type.__setattr__(boundary_type, "use_index", original)
        self.assertIs(returned, expected_alias)
        self.assertTrue(state["patched"])
        self.assertTrue(state["restored"])
        self.assertEqual(state["forged"], 0)
        registry.abort()

    def test_public_type_binding_and_port_method_substitution_is_rejected(self):
        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="module-type-substitution")
        admission = registry.admit_stage(topology.stage_uses[0])
        with mock.patch.object(
            physical, "StageAliasHandle", object()
        ):
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "PUBLIC_ABI_SUBSTITUTION",
            ):
                registry.lookup_execution_alias(admission, 2)

        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="port-method-substitution")
        port_type = type(registry)
        original = type.__getattribute__(
            port_type, "__dict__"
        )["lookup_execution_alias"]
        forged_calls = {"count": 0}

        def forged_lookup(value, admission, layer_id):
            del value, admission, layer_id
            forged_calls["count"] += 1
            return object()

        try:
            type.__setattr__(
                port_type, "lookup_execution_alias", forged_lookup
            )
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "PUBLIC_ABI_SUBSTITUTION",
            ):
                registry.stats()
        finally:
            type.__setattr__(
                port_type, "lookup_execution_alias", original
            )
        self.assertEqual(forged_calls["count"], 0)
        registry.abort()

    def test_handle_class_transplant_and_cross_registry_transplant_fail(self):
        port_topology = _shared_dense_topology()
        port = _open(port_topology, frame_label="port-class-transplant")
        port_type = type(port)
        counterfeit = object.__new__(port_type)
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "HANDLE_SEAL_MISMATCH",
        ):
            port_type.stats(counterfeit)

        class PortTwin(port_type):
            __slots__ = ()

        try:
            object.__setattr__(port, "__class__", PortTwin)
        except TypeError:
            port.abort()
        else:
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "HANDLE_SEAL_MISMATCH",
            ):
                port.stats()

        topology = _shared_dense_topology()
        registry = _open(topology, frame_label="class-transplant")
        admission = registry.admit_stage(topology.stage_uses[0])

        class AdmissionTwin(physical.StageAdmission):
            __slots__ = ()

        try:
            object.__setattr__(admission, "__class__", AdmissionTwin)
        except TypeError:
            pass
        else:
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "INVALID_LOOKUP",
            ):
                registry.lookup_execution_alias(admission, 2)

        topology = _shared_dense_topology()
        first = _open(topology, frame_label="transplant-first")
        second = _open(topology, frame_label="transplant-second")
        first_admission = first.admit_stage(topology.stage_uses[0])
        second_admission = second.admit_stage(topology.stage_uses[0])
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "HANDLE_SEAL_MISMATCH",
        ):
            second.lookup_execution_alias(first_admission, 2)
        self.assertIs(
            first.lookup_execution_alias(first_admission, 2),
            first_admission.aliases[0],
        )

        third = _open(topology, frame_label="transplant-third")
        third_admission = third.admit_stage(topology.stage_uses[0])
        object.__setattr__(
            third_admission.aliases[0],
            "physical_core",
            second_admission.aliases[0].physical_core,
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "HANDLE_SEAL_MISMATCH",
        ):
            third.lookup_execution_alias(third_admission, 2)

    def test_commit_rejects_raw_root_replacement(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        _admit_all(registry, topology)
        layer = topology.full_layers[2]
        changed = dict(layer.params)
        changed["weight"] = _f64(
            np.asarray([[[[1.5]]]], dtype=np.float64),
            name="substituted raw weight",
        )
        object.__setattr__(layer, "params", MappingProxyType(changed))
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "RAW_CONTEXT_MISMATCH",
        ):
            registry.commit()

    def test_commit_rejects_raw_frame_box_replacement(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        _admit_all(registry, topology)
        raw_box = topology.frame_bounds[3]
        object.__setattr__(
            raw_box,
            "lb",
            _f64(
                np.asarray([-0.25], dtype=np.float64),
                name="substituted raw lower",
            ),
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "RAW_CONTEXT_MISMATCH",
        ):
            registry.commit()

    def test_commit_entry_normalizes_mutated_stage_use_and_poisons(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="mutated-stage-frame")
        _admit_all(registry, topology)
        object.__setattr__(topology.stage_uses[0], "use_index", 99)
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "RAW_CONTEXT_MISMATCH",
        ):
            registry.commit()
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError, "INVALID_STATE"
        ):
            registry.stats()

    def test_raw_mutation_during_commit_cannot_cross_publication_gate(self):
        topology = _five_cone_topology()
        registry = _open(topology, frame_label="commit-race-frame")
        _admit_all(registry, topology)
        changed = {"done": False}

        def mutate_after_certificate(frame, event, arg):
            del arg
            if (
                event == "line"
                and frame.f_code.co_name == "commit_impl"
                and "certificate" in frame.f_locals
                and not changed["done"]
            ):
                changed["done"] = True
                layer = topology.full_layers[2]
                params = dict(layer.params)
                params["weight"] = _f64(
                    np.asarray([[[[2.25]]]], dtype=np.float64),
                    name="mid-commit substituted raw weight",
                )
                object.__setattr__(
                    layer, "params", MappingProxyType(params)
                )
            return mutate_after_certificate

        previous_trace = sys.gettrace()
        try:
            sys.settrace(mutate_after_certificate)
            with self.assertRaisesRegex(
                physical.V51BPhysicalRegistryError,
                "RAW_CONTEXT_MISMATCH",
            ):
                registry.commit()
        finally:
            sys.settrace(previous_trace)
        self.assertTrue(changed["done"])

    def test_expired_deadline_fails_creation_and_pre_commit_gate(self):
        topology = _five_cone_topology()
        with self.assertRaises(physical.V51BPhysicalRegistryTimeout):
            _open(topology, deadline=time.monotonic() - 1.0)

        deadline = time.monotonic() + 100.0
        registry = _open(topology, deadline=deadline)
        _admit_all(registry, topology)
        _set_registry_deadline(
            registry, "commit", time.monotonic() - 1.0
        )
        with self.assertRaises(physical.V51BPhysicalRegistryTimeout):
            registry.commit()

        # Expiry after receipt construction must still prevent publication.
        late_deadline = time.monotonic() + 100.0
        late_registry = _open(topology, deadline=late_deadline)
        _admit_all(late_registry, topology)
        changed = {"done": False}

        def expire_after_certificate(frame, event, arg):
            del arg
            if (
                event == "line"
                and frame.f_code.co_name == "commit_impl"
                and "certificate" in frame.f_locals
                and not changed["done"]
            ):
                _set_registry_deadline(
                    late_registry,
                    "commit",
                    time.monotonic() - 1.0,
                )
                changed["done"] = True
            return expire_after_certificate

        previous_trace = sys.gettrace()
        try:
            sys.settrace(expire_after_certificate)
            with self.assertRaises(
                physical.V51BPhysicalRegistryTimeout
            ):
                late_registry.commit()
        finally:
            sys.settrace(previous_trace)
        self.assertTrue(changed["done"])
        _set_registry_deadline(
            late_registry, "stats", time.monotonic() + 100.0
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError, "INVALID_STATE"
        ):
            late_registry.stats()

    def test_stats_has_a_post_scan_deadline_publication_gate(self):
        topology = _five_cone_topology()
        deadline = time.monotonic() + 100.0
        registry = _open(
            topology,
            frame_label="stats-deadline-frame",
            deadline=deadline,
        )
        registry.admit_stage(topology.stage_uses[0])
        changed = {"done": False}

        def expire_after_scan(frame, event, arg):
            del arg
            if (
                event == "line"
                and frame.f_code.co_name == "stats"
                and "result" in frame.f_locals
                and not changed["done"]
            ):
                _set_registry_deadline(
                    registry,
                    "stats",
                    time.monotonic() - 1.0,
                )
                changed["done"] = True
            return expire_after_scan

        previous_trace = sys.gettrace()
        try:
            sys.settrace(expire_after_scan)
            with self.assertRaises(physical.V51BPhysicalRegistryTimeout):
                registry.stats()
        finally:
            sys.settrace(previous_trace)
        self.assertTrue(changed["done"])

    def test_public_pid_entry_replacement_cannot_affect_registry(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        with mock.patch.object(
            physical.os,
            "getpid",
            side_effect=AssertionError("public os.getpid executed"),
        ) as forged_getpid:
            self.assertEqual(registry.stats().state, "OPEN")
        self.assertEqual(forged_getpid.call_count, 0)
        self.assertEqual(registry.stats().state, "OPEN")
        registry.abort()

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_actual_fork_cannot_use_parent_registry_capability(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                registry.stats()
            except physical.V51BPhysicalRegistryError as exc:
                payload = exc.code.encode("ascii")
            except BaseException:
                payload = b"UNEXPECTED"
            else:
                payload = b"ACCEPTED"
            os.write(write_fd, payload)
            os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        payload = os.read(read_fd, 256)
        os.close(read_fd)
        _, status = os.waitpid(child_pid, 0)
        self.assertEqual(status, 0)
        self.assertEqual(payload, b"PROCESS_MISMATCH")
        self.assertEqual(registry.stats().state, "OPEN")
        registry.abort()

    def test_registry_and_all_handles_are_noncopyable(self):
        topology = _shared_dense_topology()
        registry = _open(topology)
        admission = registry.admit_stage(topology.stage_uses[0])
        alias = admission.aliases[0]
        stats = registry.stats()
        registry.admit_stage(topology.stage_uses[1])
        certificate = registry.commit()
        values = (
            registry,
            admission,
            alias,
            alias.physical_core,
            stats,
            certificate,
        )
        for value in values:
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(
                    physical.V51BPhysicalRegistryError,
                    "COPY_FORBIDDEN",
                ):
                    copy.copy(value)
                with self.assertRaisesRegex(
                    physical.V51BPhysicalRegistryError,
                    "COPY_FORBIDDEN",
                ):
                    copy.deepcopy(value)
                with self.assertRaisesRegex(
                    physical.V51BPhysicalRegistryError,
                    "COPY_FORBIDDEN",
                ):
                    pickle.dumps(value)

        # A certificate is terminal non-authoritative diagnostic output.  A
        # reflective mutation cannot alter closure-owned committed state.
        object.__setattr__(certificate, "physical_builds", 1000)
        self.assertEqual(certificate.physical_builds, 1000)
        self.assertEqual(registry.stats().physical_builds, 1)
        self.assertFalse(registry.stats().proof_authority)

    def test_registry_port_and_material_closure_are_collectable(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        admission = registry.admit_stage(topology.stage_uses[0])
        self.assertEqual(len(admission.aliases), 1)
        port_method = type(registry).admit_stage
        port_cells = dict(
            zip(
                port_method.__code__.co_freevars,
                (
                    cell.cell_contents
                    for cell in port_method.__closure__
                ),
            )
        )
        admit_impl = port_cells["admit_impl"]
        admit_cells = dict(
            zip(
                admit_impl.__code__.co_freevars,
                (
                    cell.cell_contents
                    for cell in admit_impl.__closure__
                ),
            )
        )
        cores = admit_cells["cores_by_key"]
        material = next(iter(cores.values())).material
        material_reference = weakref.ref(material)
        self.assertIsNotNone(material_reference())
        reference = weakref.ref(registry)
        del material, cores, admit_cells, admit_impl, port_cells, port_method
        del registry
        for _ in range(3):
            gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(material_reference())
        # Handles can outlive cleanup only as inert, non-authoritative values.
        self.assertFalse(admission.proof_authority)
        self.assertFalse(admission.aliases[0].proof_authority)

    def test_equal_but_reconstructed_stage_use_is_not_an_admission_capability(self):
        topology = _five_cone_topology()
        registry = _open(topology)
        original = topology.stage_uses[0]
        reconstructed = authority.StageUse(
            use_index=original.use_index,
            stage_kind=original.stage_kind,
            stage_index=original.stage_index,
            target_relu_lid=original.target_relu_lid,
            cone_start_lid=original.cone_start_lid,
        )
        self.assertEqual(
            reconstructed.stage_use_sha256, original.stage_use_sha256
        )
        with self.assertRaisesRegex(
            physical.V51BPhysicalRegistryError,
            "STAGE_SEAL_MISMATCH",
        ):
            registry.admit_stage(reconstructed)


if __name__ == "__main__":
    unittest.main()
