"""Focused gates for the non-authoritative V5.1b prepared adapter."""

from __future__ import annotations

import builtins
import copy
import dis
import gc
import hashlib
import os
import pickle
import threading
import time
import types
import unittest
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_prepared_adapter as adapter,
)
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as dense_v51


_FROZEN_KERNEL_SHA256 = (
    "6f72ee6f1f301c9818dcc2ab754346a4cdb08ff072b1b3f18fb3557c55228329"
)


def _end(seconds: float = 30.0) -> float:
    return float(time.monotonic() + seconds)


def _sha(character: str) -> str:
    return character * 64


def _binding(
    kind: str,
    *,
    layer_id: int = 2,
    predecessor_id: int = 1,
    physical: str = "3",
) -> tuple[object, ...]:
    if kind == adapter.BRANCH_DENSE:
        lb_sha = adapter.ZERO_SHA256
        ub_sha = adapter.ZERO_SHA256
        geometry_sha = adapter.ZERO_SHA256
    else:
        lb_sha = _sha("8")
        ub_sha = _sha("9")
        geometry_sha = _sha("a")
    return (
        adapter.RAW_BINDING_TAG,
        _sha("1"),
        _sha("2"),
        _sha(physical),
        kind,
        layer_id,
        predecessor_id,
        _sha("4"),
        lb_sha,
        ub_sha,
        geometry_sha,
        _sha("5"),
        _sha("6"),
    )


def _dense_fixture(deadline: float):
    weight = np.ascontiguousarray(
        np.asarray(
            [[1.0, -2.0, 0.5], [0.25, 4.0, -1.0]],
            dtype=np.float64,
        )
    )
    max_abs = np.ascontiguousarray(
        np.asarray([2.0, 1.0, 3.0], dtype=np.float64)
    )
    coefficients = np.ascontiguousarray(
        np.asarray(
            [[1.0, -0.5], [0.25, 2.0], [0.0, -0.0]],
            dtype=np.float64,
        )
    )
    support = dense_v51.prepare_dense_support_v51(
        weight, max_abs, deadline=deadline
    )
    return weight, max_abs, coefficients, support


def _conv_fixture(deadline: float):
    weight = np.ascontiguousarray(
        np.asarray([[[[1.0]]], [[[-0.5]]]], dtype=np.float64)
    )
    lb = np.ascontiguousarray(-np.ones(4, dtype=np.float64))
    ub = np.ascontiguousarray(np.ones(4, dtype=np.float64))
    input_shape = (1, 1, 4)
    output_shape = (2, 1, 4)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    layer = frozen._FrozenLayer(
        id=2,
        kind="CONV2D",
        preds=(1,),
        width=8,
        in_vars=(),
        out_vars=(),
        params=MappingProxyType(
            {
                "weight": weight,
                "bias_channels": np.zeros(2, dtype=np.float64),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
        ),
    )
    box = frozen._Box(lb=lb, ub=ub)
    plan = conv_v51.prepare_dense_conv_v51_plan(
        layer, box, deadline=frozen._Deadline(end=deadline)
    )
    coefficients = np.ascontiguousarray(
        np.linspace(-1.0, 1.0, 24, dtype=np.float64).reshape(3, 8)
    )
    raw = {
        "layer_id": 2,
        "weight": weight,
        "predecessor_lb": lb,
        "predecessor_ub": ub,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    return raw, coefficients, plan


def _closure_values(function):
    function = getattr(function, "__wrapped__", function)
    return {
        name: cell.cell_contents
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or ()
        )
    }


def _closure_cell(function, name):
    function = getattr(function, "__wrapped__", function)
    for freevar, cell in zip(
        function.__code__.co_freevars, function.__closure__ or ()
    ):
        if freevar == name:
            return cell
    raise AssertionError(
        "{} is not captured by {}".format(name, function.__name__)
    )


def _exact_clone(value, **changes):
    clone = object.__new__(type(value))
    clone.__dict__.update(value.__dict__)
    clone.__dict__.update(changes)
    return clone


def _immutable_copy(value):
    contiguous = np.ascontiguousarray(value)
    payload = contiguous.tobytes(order="C")
    return np.frombuffer(payload, dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _adapter_operation_lock(port):
    return _closure_values(port.stats.__func__)["operation_lock"]


def _dense_dispatch(port):
    adapter_cells = _closure_values(port.admit_dense_prepared.__func__)
    frozen_method = adapter_cells["_frozen_admit_dense"]
    guarded_cells = _closure_values(frozen_method.__func__)
    frozen_cells = _closure_values(guarded_cells["operation"])
    return frozen_cells["_prepare_dense_public"]


def _conv_dispatch(port):
    adapter_cells = _closure_values(port.admit_conv_prepared.__func__)
    frozen_method = adapter_cells["_frozen_admit_conv"]
    guarded_cells = _closure_values(frozen_method.__func__)
    frozen_cells = _closure_values(guarded_cells["operation"])
    return frozen_cells["_prepare_conv_public"]


def _assert_no_global_opcodes(testcase, root_code, *, minimum_codes=1):
    pending = [root_code]
    visited = 0
    while pending:
        code = pending.pop()
        visited += 1
        for instruction in dis.get_instructions(code):
            testcase.assertNotIn(
                instruction.opname,
                {
                    "LOAD_BUILD_CLASS",
                    "LOAD_GLOBAL",
                    "STORE_GLOBAL",
                    "DELETE_GLOBAL",
                },
                (code.co_name, instruction.opname, instruction.argval),
            )
        pending.extend(
            value
            for value in code.co_consts
            if type(value) is types.CodeType
        )
    testcase.assertGreaterEqual(visited, minimum_codes)


class PreparedAdapterAbiTests(unittest.TestCase):
    def test_frozen_sha_and_exact_closure_abi(self):
        path = Path(private.__file__).resolve()
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            _FROZEN_KERNEL_SHA256,
        )
        factory = private.create_private_numeric_kernel
        self.assertIs(type(factory), types.FunctionType)
        self.assertEqual(
            factory.__code__.co_freevars,
            ("implementation", "sealed_dependencies"),
        )
        self.assertIs(type(factory.__closure__), tuple)
        self.assertEqual(len(factory.__closure__), 2)
        implementation = factory.__closure__[0].cell_contents
        sealed = factory.__closure__[1].cell_contents
        self.assertIs(type(implementation), tuple)
        self.assertEqual(len(implementation), 7)
        self.assertIs(type(implementation[0]), types.CodeType)
        self.assertEqual(implementation[0].co_freevars, ())
        self.assertIs(type(implementation[1]), str)
        self.assertIsNone(implementation[2])
        self.assertIsNone(implementation[3])
        self.assertIs(implementation[4], types.FunctionType)
        self.assertIs(implementation[5], dict)
        self.assertIs(type(implementation[6]), str)
        self.assertIs(type(sealed), tuple)
        self.assertEqual(len(sealed), 5)
        self.assertIs(type(sealed[4]), tuple)
        self.assertEqual(len(sealed[4]), 44)
        self.assertIs(
            sealed[4][8], dense_v51.prepare_dense_support_v51
        )
        self.assertIs(
            sealed[4][9], conv_v51.prepare_dense_conv_v51_plan
        )
        self.assertIs(sealed[4][43], False)

    def test_adapter_sealed_implementation_has_no_global_opcode(self):
        factory = adapter.create_prepared_numeric_adapter
        cells = _closure_values(factory)
        self.assertIn("implementation_code", cells)
        implementation_code = cells["implementation_code"]
        self.assertIs(type(implementation_code), types.CodeType)
        self.assertEqual(implementation_code.co_freevars, ())
        _assert_no_global_opcodes(
            self, implementation_code, minimum_codes=40
        )

    def test_each_factory_has_no_runtime_class_builder(self):
        factory_cells = _closure_values(
            adapter.create_prepared_numeric_adapter
        )
        self.assertNotIn(
            "__build_class__",
            dict(factory_cells["trusted_builtins"]),
        )
        first = adapter.create_prepared_numeric_adapter(deadline=_end())
        first_globals = first.stats.__func__.__globals__
        self.assertNotIn("__build_class__", first_globals["__builtins__"])
        second = adapter.create_prepared_numeric_adapter(deadline=_end())
        self.assertEqual(second.stats()["state"], "OPEN")
        self.assertIsNot(
            first_globals,
            second.stats.__func__.__globals__,
        )

    def test_operation_function_closure_graph_has_no_module_view(self):
        port = adapter.create_prepared_numeric_adapter(deadline=_end())
        dispatch_cells = _closure_values(_dense_dispatch(port))
        consume = dispatch_cells["_consume"]
        self.assertNotIn("_direct", consume.__code__.co_freevars)
        self.assertNotIn(
            "_expected_dense_prepare", consume.__code__.co_freevars
        )
        self.assertNotIn(
            "_expected_conv_prepare", consume.__code__.co_freevars
        )
        pending = [
            port.admit_dense_prepared,
            port.admit_conv_prepared,
            port.execute_dense,
            port.execute_conv,
            port.stats,
            port.close,
        ]
        adapter_globals_id = id(port.stats.__func__.__globals__)
        frozen_globals_id = id(
            _closure_values(
                port.admit_dense_prepared.__func__
            )["_frozen_admit_dense"].__func__.__globals__
        )
        operation_globals_ids = {
            adapter_globals_id,
            frozen_globals_id,
        }
        seen = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            self.assertNotIsInstance(value, types.ModuleType)
            self.assertIsNot(
                value, dense_v51.prepare_dense_support_v51
            )
            self.assertIsNot(
                value, conv_v51.prepare_dense_conv_v51_plan
            )
            self.assertIsNot(value, dense_v51.DenseV51Support)
            self.assertIsNot(value, conv_v51.DenseConvV51Plan)
            self.assertIsNot(value, conv_v51._OffsetSupport)
            self.assertIsNot(value, frozen._FrozenLayer)
            self.assertIsNot(value, frozen._Box)
            self.assertIsNot(value, frozen._Deadline)
            # Weak references are terminal in this strong-referent audit.
            # In particular, do not call one and count its referent.
            if type(value) is weakref.ReferenceType:
                continue
            if type(value) is types.MethodType:
                pending.append(value.__func__)
                continue
            if type(value) is types.FunctionType:
                if id(value.__globals__) not in operation_globals_ids:
                    continue
                _assert_no_global_opcodes(self, value.__code__)
                pending.extend(
                    cell.cell_contents
                    for cell in value.__closure__ or ()
                )
                continue
            if type(value) in (tuple, list):
                pending.extend(value)
                continue
            if type(value) is dict or type(value) is MappingProxyType:
                pending.extend(value.values())

    def test_prepared_kernel_closure_uses_private_types_and_true_mode(self):
        port = adapter.create_prepared_numeric_adapter(deadline=_end())
        adapter_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        dense_adapter_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        dense_copy_cells = _closure_values(
            dense_adapter_cells["_copy_dense_support"]
        )
        conv_copy_cells = _closure_values(
            adapter_cells["_copy_conv_plan"]
        )
        offset_scan_cells = _closure_values(
            conv_copy_cells["_scan_conv_offset"]
        )
        type_refs = (
            (
                dense_copy_cells["_DenseSupportPublicRef"],
                dense_v51.DenseV51Support,
            ),
            (
                conv_copy_cells["_ConvPlanPublicRef"],
                conv_v51.DenseConvV51Plan,
            ),
            (
                offset_scan_cells["_ConvOffsetPublicRef"],
                conv_v51._OffsetSupport,
            ),
        )
        for reference, expected_type in type_refs:
            self.assertIs(type(reference), weakref.ReferenceType)
            self.assertIs(reference(), expected_type)
        frozen_conv = adapter_cells["_frozen_admit_conv"]
        conv_guarded_cells = _closure_values(frozen_conv.__func__)
        conv_cells = _closure_values(conv_guarded_cells["operation"])
        dense_guarded_cells = _closure_values(
            dense_adapter_cells["_frozen_admit_dense"].__func__
        )
        dense_cells = _closure_values(dense_guarded_cells["operation"])
        self.assertIs(conv_cells["_prepared_mode"], True)
        self.assertIsNot(conv_cells["_FrozenLayer"], frozen._FrozenLayer)
        self.assertIsNot(conv_cells["_Box"], frozen._Box)
        self.assertIsNot(conv_cells["_Deadline"], frozen._Deadline)
        self.assertIsNot(
            dense_cells["_DenseSupport"], dense_v51.DenseV51Support
        )
        self.assertIsNot(
            conv_cells["_ConvPlan"], conv_v51.DenseConvV51Plan
        )
        self.assertIsNot(
            conv_cells["_ConvOffset"], conv_v51._OffsetSupport
        )
        self.assertIs(
            conv_cells["_prepare_conv_public"], _conv_dispatch(port)
        )
        self.assertIs(
            dense_cells["_prepare_dense_public"], _dense_dispatch(port)
        )

    def test_factory_rejects_pre_factory_dependency_identity_change(self):
        replacement = lambda **kwargs: kwargs
        with mock.patch.object(
            private, "create_private_numeric_kernel", replacement
        ):
            with self.assertRaisesRegex(
                adapter.PreparedNumericAdapterError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                adapter.create_prepared_numeric_adapter(deadline=_end())
        with mock.patch.object(np, "frombuffer", replacement):
            with self.assertRaisesRegex(
                adapter.PreparedNumericAdapterError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                adapter.create_prepared_numeric_adapter(deadline=_end())
        with mock.patch.object(builtins, "memoryview", replacement):
            with self.assertRaisesRegex(
                adapter.PreparedNumericAdapterError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                adapter.create_prepared_numeric_adapter(deadline=_end())

    def test_factory_rejects_custom_module_view_before_getattribute(self):
        calls = []
        original_type = private.__class__

        class ChangedModule(types.ModuleType):
            def __getattribute__(self, name):
                calls.append(name)
                raise AssertionError("custom module view dispatched")

        try:
            private.__class__ = ChangedModule
            with self.assertRaisesRegex(
                adapter.PreparedNumericAdapterError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                adapter.create_prepared_numeric_adapter(deadline=_end())
        finally:
            private.__class__ = original_type
        self.assertEqual(calls, [])

    def test_factory_failure_graph_discards_fresh_implementation(self):
        replacement = lambda value: value
        caught = None
        with mock.patch.object(builtins, "memoryview", replacement):
            try:
                adapter.create_prepared_numeric_adapter(deadline=_end())
            except adapter.PreparedNumericAdapterError as exc:
                caught = exc
        self.assertIsNotNone(caught)
        self.assertEqual(caught.code, "DEPENDENCY_SUBSTITUTION")
        self.assertIsNone(caught.__cause__)
        self.assertIsNone(caught.__context__)
        adapter_path = Path(adapter.__file__).resolve()
        names = []
        traceback = caught.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if Path(frame.f_code.co_filename).resolve() == adapter_path:
                names.append(frame.f_code.co_name)
                for value in frame.f_locals.values():
                    self.assertNotIsInstance(value, np.ndarray)
            traceback = traceback.tb_next
        self.assertEqual(names, ["create_prepared_numeric_adapter"])

    def test_exact_deadline_and_terminal_stats(self):
        for value in (1, True, np.float64(_end()), float("inf")):
            with self.subTest(value=repr(value)):
                with self.assertRaises(
                    adapter.PreparedNumericAdapterError
                ):
                    adapter.create_prepared_numeric_adapter(
                        deadline=value
                    )
        with self.assertRaises(
            adapter.PreparedNumericAdapterTimeout
        ):
            adapter.create_prepared_numeric_adapter(
                deadline=float(time.monotonic() - 1.0)
            )
        port = adapter.create_prepared_numeric_adapter(deadline=_end())
        port.close()
        stats = port.stats()
        self.assertIs(type(stats), MappingProxyType)
        self.assertEqual(stats["state"], "CLOSED")
        self.assertFalse(stats["proof_authority"])

    def test_prepared_conv_skips_public_layer_box_deadline_construction(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        calls = []

        def changed_init(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("changed public constructor called")

        with mock.patch.object(
            frozen._FrozenLayer, "__init__", changed_init
        ), mock.patch.object(
            frozen._Box, "__init__", changed_init
        ), mock.patch.object(
            frozen._Deadline, "__init__", changed_init
        ):
            locator = port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                prepared_plan=plan,
            )
        self.assertEqual(calls, [])
        stats = port.stats()
        self.assertEqual(stats["state"], "OPEN")
        self.assertEqual(stats["dispatch_arms"], 1)
        self.assertEqual(stats["dispatch_consumes"], 1)
        self.assertEqual(stats["locator_count"], 1)
        self.assertFalse(locator.proof_authority)


class PreparedAdapterAdmissionTests(unittest.TestCase):
    def test_dense_and_conv_match_original_kernel_all_raw_fields(self):
        deadline = _end()
        weight, max_abs, dense_coefficients, support = _dense_fixture(
            deadline
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        dense_locator = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=support,
            tile_width=2,
        )
        actual_dense = port.execute_dense(
            dense_locator, dense_coefficients
        )
        original = private.create_private_numeric_kernel(deadline=_end())
        original_dense_locator = original.admit_dense(
            weight=weight,
            predecessor_max_abs=max_abs,
            tile_width=2,
        )
        expected_dense = original.execute_dense(
            original_dense_locator, dense_coefficients
        )
        self.assertEqual(actual_dense, expected_dense)

        raw, conv_coefficients, plan = _conv_fixture(deadline)
        conv_locator = port.admit_conv_prepared(
            **raw,
            raw_binding=_binding(
                adapter.BRANCH_CONV_DENSE, physical="7"
            ),
            prepared_plan=plan,
        )
        actual_conv = port.execute_conv(
            conv_locator, conv_coefficients
        )
        original_conv_locator = original.admit_conv(**raw)
        expected_conv = original.execute_conv(
            original_conv_locator, conv_coefficients
        )
        self.assertEqual(actual_conv, expected_conv)
        for result in (actual_dense, actual_conv):
            self.assertIs(type(result), tuple)
            self.assertIs(result[1], False)
            for frame in result[2:]:
                self.assertIs(type(frame), tuple)
                self.assertIs(type(frame[0]), bytes)
                self.assertIs(type(frame[1]), tuple)
                self.assertIs(type(frame[2]), bytes)

    def test_original_public_preparers_are_call_zero_at_admission(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        raw, _, plan = _conv_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        with mock.patch.object(
            dense_v51,
            "prepare_dense_support_v51",
            side_effect=AssertionError("public Dense preparer called"),
        ) as dense_public, mock.patch.object(
            conv_v51,
            "prepare_dense_conv_v51_plan",
            side_effect=AssertionError("public Conv preparer called"),
        ) as conv_public:
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
            port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(
                    adapter.BRANCH_CONV_DENSE, physical="7"
                ),
                prepared_plan=plan,
            )
        dense_public.assert_not_called()
        conv_public.assert_not_called()

    def test_post_factory_memoryview_substitution_is_ignored(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        calls = []

        def changed_memoryview(value):
            calls.append(value)
            raise AssertionError("changed memoryview was called")

        with mock.patch.object(
            builtins, "memoryview", changed_memoryview
        ):
            locator = port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
        self.assertEqual(calls, [])
        self.assertFalse(locator.proof_authority)

    def test_public_type_identity_rebinding_rejects_before_arm(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        replacement_type = type("DenseV51Support", (), {})
        replacement = replacement_type()
        replacement.__dict__.update(support.__dict__)
        with mock.patch.object(
            dense_v51, "DenseV51Support", replacement_type
        ):
            with self.assertRaisesRegex(
                adapter.PreparedNumericAdapterError,
                "INVALID_PREPARED_TYPE",
            ):
                port.admit_dense_prepared(
                    weight=weight,
                    predecessor_max_abs=max_abs,
                    raw_binding=_binding(adapter.BRANCH_DENSE),
                    prepared_support=replacement,
                )
        stats = port.stats()
        self.assertEqual(stats["state"], "POISONED")
        self.assertEqual(stats["dispatch_arms"], 0)
        self.assertEqual(stats["dispatch_consumes"], 0)

    def test_successful_admission_releases_public_prepared_objects(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        raw, _, plan = _conv_fixture(deadline)
        support_ref = weakref.ref(support)
        plan_ref = weakref.ref(plan)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        dense_locator = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=support,
        )
        conv_locator = port.admit_conv_prepared(
            **raw,
            raw_binding=_binding(
                adapter.BRANCH_CONV_DENSE, physical="7"
            ),
            prepared_plan=plan,
        )
        del support
        del plan
        for _ in range(4):
            gc.collect()
            if support_ref() is None and plan_ref() is None:
                break
        self.assertIsNone(support_ref())
        self.assertIsNone(plan_ref())
        self.assertEqual(port.stats()["locator_count"], 2)
        self.assertFalse(dense_locator.proof_authority)
        self.assertFalse(conv_locator.proof_authority)

    def test_prepared_arrays_and_raw_binding_use_distinct_private_copies(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        raw, _, plan = _conv_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        dense_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        conv_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        private_support = dense_cells["_copy_dense_support"](support)
        private_plan = conv_cells["_copy_conv_plan"](plan)
        pairs = [
            (private_support.support_upper, support.support_upper),
            (private_plan.weight, plan.weight),
            (private_plan.support, plan.support),
        ]
        for private_array, public_array in pairs:
            self.assertIsNot(private_array, public_array)
            self.assertFalse(private_array.flags.writeable)
            self.assertFalse(private_array.flags.owndata)
            base = private_array
            while type(base) is np.ndarray:
                base = base.base
            self.assertIs(type(base), bytes)
        for private_offset, public_offset in zip(
            private_plan.offsets, plan.offsets
        ):
            for name in (
                "output_h_indices",
                "output_w_indices",
                "targets",
                "support_flat",
                "channel_support_flat",
                "support_activity_flat",
            ):
                private_array = getattr(private_offset, name)
                public_array = getattr(public_offset, name)
                self.assertIsNot(private_array, public_array)
                self.assertFalse(private_array.flags.writeable)

        raw_binding = _binding(adapter.BRANCH_DENSE)
        locator = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=raw_binding,
            prepared_support=support,
        )
        mint_cells = _closure_values(dense_cells["_mint_locator"])
        records = mint_cells["records"]
        record = next(iter(records.values()))
        self.assertEqual(record[1], raw_binding)
        self.assertIsNot(record[1], raw_binding)
        self.assertFalse(locator.proof_authority)

    def test_writable_prepared_storage_rejects_before_dispatch(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        writable = np.ascontiguousarray(support.support_upper.copy())
        wrong = _exact_clone(support, support_upper=writable)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "INVALID_PREPARED_VALUE",
        ):
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=wrong,
            )
        stats = port.stats()
        self.assertEqual(stats["dispatch_arms"], 0)
        self.assertEqual(stats["dispatch_consumes"], 0)
        self.assertEqual(stats["frozen_material_count"], 0)

    def test_dtype_metadata_cannot_alias_a_canonical_dtype(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        metadata_dtype = np.dtype(
            np.float64, metadata={"prepared-toy": "different"}
        )
        payload = support.support_upper.tobytes(order="C")
        changed = np.frombuffer(payload, dtype=metadata_dtype)
        self.assertEqual(changed.dtype, np.dtype(np.float64))
        self.assertIsNot(changed.dtype, np.dtype(np.float64))
        wrong = _exact_clone(support, support_upper=changed)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "INVALID_PREPARED_VALUE",
        ):
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=wrong,
            )
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_aggregate_budget_rejects_before_first_copy(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        copy_dense = admit_cells["_copy_dense_support"]
        copy_cells = _closure_values(copy_dense)
        account = copy_cells["_account_private_array"]
        _closure_cell(
            account, "_MAX_PREPARED_RAW_BYTES"
        ).cell_contents = support.support_upper.nbytes - 1
        copy_calls = []

        def forbidden_copy(ticket):
            copy_calls.append(ticket)
            raise AssertionError("copy began before aggregate admission")

        _closure_cell(
            copy_dense, "_copy_ticketed_array"
        ).cell_contents = forbidden_copy
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
        self.assertEqual(copy_calls, [])
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_conv_scans_every_offset_before_first_copy(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        first = plan.offsets[0]
        writable = np.ascontiguousarray(first.support_flat.copy())
        wrong_offset = _exact_clone(first, support_flat=writable)
        wrong_plan = _exact_clone(
            plan,
            offsets=(first, wrong_offset),
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        copy_plan = admit_cells["_copy_conv_plan"]
        copy_calls = []

        def forbidden_copy(ticket):
            copy_calls.append(ticket)
            raise AssertionError("copy began before offset scan finished")

        _closure_cell(
            copy_plan, "_copy_ticketed_array"
        ).cell_contents = forbidden_copy
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "INVALID_PREPARED_VALUE",
        ):
            port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                prepared_plan=wrong_plan,
            )
        self.assertEqual(copy_calls, [])
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_conv_offset_cap_rejects_before_first_copy(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        copy_plan = admit_cells["_copy_conv_plan"]
        _closure_cell(
            copy_plan, "_MAX_PREPARED_OFFSETS"
        ).cell_contents = 0
        copy_calls = []

        def forbidden_copy(ticket):
            copy_calls.append(ticket)
            raise AssertionError("copy began before offset-count admission")

        _closure_cell(
            copy_plan, "_copy_ticketed_array"
        ).cell_contents = forbidden_copy
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                prepared_plan=plan,
            )
        self.assertEqual(copy_calls, [])
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_fixed_resource_contract_values_and_real_offset_cap(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        raw, _, plan = _conv_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        dense_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        copy_dense = dense_cells["_copy_dense_support"]
        copy_dense_cells = _closure_values(copy_dense)
        scan = copy_dense_cells["_scan_private_array"]
        account = copy_dense_cells["_account_private_array"]
        conv_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        copy_plan = conv_cells["_copy_conv_plan"]
        self.assertEqual(
            _closure_cell(
                scan, "_MAX_PREPARED_ARRAY_BYTES"
            ).cell_contents,
            1 << 30,
        )
        self.assertEqual(
            _closure_cell(
                account, "_MAX_PREPARED_RAW_BYTES"
            ).cell_contents,
            1 << 28,
        )
        self.assertEqual(
            _closure_cell(
                copy_plan, "_MAX_PREPARED_OFFSETS"
            ).cell_contents,
            65536,
        )
        del weight
        del max_abs
        del support

        wrong_plan = _exact_clone(
            plan,
            offsets=(plan.offsets[0],) * 65537,
        )
        copy_calls = []
        scan_calls = []

        def forbidden_copy(ticket):
            copy_calls.append(ticket)
            raise AssertionError("fixed offset cap allowed a copy")

        def forbidden_scan(value):
            scan_calls.append(value)
            raise AssertionError("fixed offset cap allowed offset scanning")

        _closure_cell(
            copy_plan, "_copy_ticketed_array"
        ).cell_contents = forbidden_copy
        _closure_cell(
            copy_plan, "_scan_conv_offset"
        ).cell_contents = forbidden_scan
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                prepared_plan=wrong_plan,
            )
        self.assertEqual(copy_calls, [])
        self.assertEqual(scan_calls, [])
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_fixed_aggregate_budget_counts_reused_array_views(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        payload = bytes(1 << 20)
        i64 = np.frombuffer(payload, dtype=np.int64)
        f64 = np.frombuffer(payload, dtype=np.float64)
        boolean = np.frombuffer(payload, dtype=np.bool_)
        first = plan.offsets[0]
        large_offset = _exact_clone(
            first,
            output_h_indices=i64,
            output_w_indices=i64,
            targets=i64,
            support_flat=f64,
            channel_support_flat=f64,
            support_activity_flat=boolean,
        )
        wrong_plan = _exact_clone(
            plan,
            offsets=(large_offset,) * 43,
        )
        per_offset_nbytes = sum(
            value.nbytes
            for value in (
                i64,
                i64,
                i64,
                f64,
                f64,
                boolean,
            )
        )
        base_nbytes = plan.weight.nbytes + plan.support.nbytes
        self.assertLessEqual(
            base_nbytes + 42 * per_offset_nbytes,
            1 << 28,
        )
        self.assertGreater(
            base_nbytes + 43 * per_offset_nbytes,
            1 << 28,
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        conv_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        copy_plan = conv_cells["_copy_conv_plan"]
        copy_calls = []

        def forbidden_copy(ticket):
            copy_calls.append(ticket)
            raise AssertionError("aggregate budget allowed a copy")

        _closure_cell(
            copy_plan, "_copy_ticketed_array"
        ).cell_contents = forbidden_copy
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                prepared_plan=wrong_plan,
            )
        self.assertEqual(copy_calls, [])
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_per_array_gib_cap_uses_pure_checked_integer_metadata(self):
        deadline = _end()
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        dense_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        copy_dense = dense_cells["_copy_dense_support"]
        scan = _closure_values(copy_dense)["_scan_private_array"]
        checked = _closure_values(scan)["_checked_prepared_nbytes"]
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            checked(
                (((1 << 30) // 8) + 1,),
                8,
                "large toy array",
            )
        self.assertEqual(
            checked((1 << 27,), 8, "boundary toy array"),
            1 << 30,
        )
        self.assertEqual(checked((1 << 60, 0), 8, "zero toy"), 0)

    def test_copy_memory_error_is_normalized(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        copy_dense = admit_cells["_copy_dense_support"]
        copy_ticket = _closure_values(copy_dense)[
            "_copy_ticketed_array"
        ]

        def no_memory(value):
            del value
            raise MemoryError()

        _closure_cell(
            copy_ticket, "_memoryview"
        ).cell_contents = no_memory
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_numpy_array_memory_error_is_normalized(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        copy_dense = admit_cells["_copy_dense_support"]
        copy_ticket = _closure_values(copy_dense)[
            "_copy_ticketed_array"
        ]
        copy_cells = _closure_values(copy_ticket)
        array_memory_error = copy_cells["_ArrayMemoryError"]

        def no_array_memory(*args, **kwargs):
            del args
            del kwargs
            raise array_memory_error((1,), np.dtype(np.float64))

        _closure_cell(
            copy_ticket, "_np_frombuffer"
        ).cell_contents = no_array_memory
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_metadata_memory_error_is_normalized(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_function = port.admit_dense_prepared.__func__

        def no_metadata_memory(value):
            del value
            raise MemoryError()

        _closure_cell(
            admit_function, "_copy_dense_support"
        ).cell_contents = no_metadata_memory
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "RESOURCE_LIMIT",
        ):
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_deadline_before_and_after_copy_never_publishes(self):
        for expiry_point in ("before", "after"):
            with self.subTest(expiry_point=expiry_point):
                deadline = _end()
                weight, max_abs, _, support = _dense_fixture(deadline)
                port = adapter.create_prepared_numeric_adapter(
                    deadline=deadline
                )
                admit_cells = _closure_values(
                    port.admit_dense_prepared.__func__
                )
                copy_dense = admit_cells["_copy_dense_support"]
                copy_cells = _closure_values(copy_dense)
                gate = copy_cells["_prepared_copy_gate"]
                deadline_cell = _closure_cell(gate, "owner_deadline")
                copy_calls = []
                if expiry_point == "before":
                    original_scan = copy_cells[
                        "_scan_private_array"
                    ]

                    def expiring_scan(*args, **kwargs):
                        ticket = original_scan(*args, **kwargs)
                        deadline_cell.cell_contents = float(
                            time.monotonic() - 1.0
                        )
                        return ticket

                    _closure_cell(
                        copy_dense, "_scan_private_array"
                    ).cell_contents = expiring_scan
                    original_copy = copy_cells[
                        "_copy_ticketed_array"
                    ]

                    def counting_copy(ticket):
                        copy_calls.append(ticket)
                        return original_copy(ticket)

                    _closure_cell(
                        copy_dense, "_copy_ticketed_array"
                    ).cell_contents = counting_copy
                else:
                    original_copy = copy_cells[
                        "_copy_ticketed_array"
                    ]

                    def expiring_copy(ticket):
                        result = original_copy(ticket)
                        copy_calls.append(ticket)
                        deadline_cell.cell_contents = float(
                            time.monotonic() - 1.0
                        )
                        return result

                    _closure_cell(
                        copy_dense, "_copy_ticketed_array"
                    ).cell_contents = expiring_copy
                with self.assertRaises(
                    adapter.PreparedNumericAdapterTimeout
                ):
                    port.admit_dense_prepared(
                        weight=weight,
                        predecessor_max_abs=max_abs,
                        raw_binding=_binding(adapter.BRANCH_DENSE),
                        prepared_support=support,
                    )
                self.assertEqual(
                    len(copy_calls),
                    0 if expiry_point == "before" else 1,
                )
                stats = port.stats()
                self.assertEqual(stats["dispatch_arms"], 0)
                self.assertEqual(stats["locator_count"], 0)

    def test_conv_each_offset_scan_has_a_deadline_checkpoint(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        two_offsets = _exact_clone(
            plan,
            offsets=(plan.offsets[0], plan.offsets[0]),
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        copy_plan = admit_cells["_copy_conv_plan"]
        copy_cells = _closure_values(copy_plan)
        original_scan = copy_cells["_scan_conv_offset"]
        gate = copy_cells["_prepared_copy_gate"]
        deadline_cell = _closure_cell(gate, "owner_deadline")
        scan_calls = []
        copy_calls = []

        def expiring_scan(value):
            scanned = original_scan(value)
            scan_calls.append(value)
            if len(scan_calls) == 1:
                deadline_cell.cell_contents = float(
                    time.monotonic() - 1.0
                )
            return scanned

        def forbidden_copy(ticket):
            copy_calls.append(ticket)
            raise AssertionError("copy began after scan deadline")

        _closure_cell(
            copy_plan, "_scan_conv_offset"
        ).cell_contents = expiring_scan
        _closure_cell(
            copy_plan, "_copy_ticketed_array"
        ).cell_contents = forbidden_copy
        with self.assertRaises(
            adapter.PreparedNumericAdapterTimeout
        ):
            port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                prepared_plan=two_offsets,
            )
        self.assertEqual(len(scan_calls), 1)
        self.assertEqual(copy_calls, [])
        self.assertEqual(port.stats()["dispatch_arms"], 0)

    def test_conv_each_offset_materialization_has_a_pid_checkpoint(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        two_offsets = _exact_clone(
            plan,
            offsets=(plan.offsets[0], plan.offsets[0]),
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admit_cells = _closure_values(
            port.admit_conv_prepared.__func__
        )
        copy_plan = admit_cells["_copy_conv_plan"]
        copy_cells = _closure_values(copy_plan)
        original_copy_offset = copy_cells[
            "_copy_scanned_conv_offset"
        ]
        gate = copy_cells["_prepared_copy_gate"]
        getpid_cell = _closure_cell(gate, "_getpid")
        original_getpid = getpid_cell.cell_contents
        owner_pid = _closure_values(gate)["owner_pid"]
        emit_mismatch = [False]
        mismatch_emitted = [False]
        copy_calls = []

        def scripted_getpid():
            if emit_mismatch[0] and not mismatch_emitted[0]:
                mismatch_emitted[0] = True
                return owner_pid + 1
            return owner_pid

        def copying_then_change_pid(scanned):
            result = original_copy_offset(scanned)
            copy_calls.append(scanned)
            if len(copy_calls) == 1:
                emit_mismatch[0] = True
            return result

        getpid_cell.cell_contents = scripted_getpid
        _closure_cell(
            copy_plan, "_copy_scanned_conv_offset"
        ).cell_contents = copying_then_change_pid
        try:
            with self.assertRaisesRegex(
                adapter.PreparedNumericAdapterError,
                "FORKED_PROCESS",
            ):
                port.admit_conv_prepared(
                    **raw,
                    raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                    prepared_plan=two_offsets,
                )
        finally:
            getpid_cell.cell_contents = original_getpid
        self.assertTrue(mismatch_emitted[0])
        self.assertEqual(len(copy_calls), 1)
        stats = port.stats()
        self.assertEqual(stats["dispatch_arms"], 0)
        self.assertEqual(stats["state"], "POISONED")

    def test_numerical_equivalence_is_separate_from_provenance(self):
        deadline = _end()
        weight, max_abs, coefficients, support = _dense_fixture(deadline)
        changed_support = _exact_clone(
            support,
            weight_sha256=_sha("a"),
            max_abs_sha256=_sha("b"),
            support_sha256=_sha("c"),
            binding=(("registry-provenance", "different"),),
            diagnostics=object(),
        )
        raw, conv_coefficients, plan = _conv_fixture(deadline)
        changed_plan = replace(
            plan,
            manifest=MappingProxyType(
                {"registry_provenance": "different"}
            ),
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        dense_locator = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=changed_support,
        )
        conv_locator = port.admit_conv_prepared(
            **raw,
            raw_binding=_binding(
                adapter.BRANCH_CONV_DENSE, physical="7"
            ),
            prepared_plan=changed_plan,
        )
        self.assertEqual(
            port.execute_dense(dense_locator, coefficients)[0],
            b"act.v51b.private.dense-result.v1",
        )
        self.assertEqual(
            port.execute_conv(conv_locator, conv_coefficients)[0],
            b"act.v51b.private.conv-result.v1",
        )

    def test_wrong_dense_fields_fail_after_consume_and_release_prepared(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        wrong_array = np.frombuffer(
            np.zeros_like(support.support_upper).tobytes(),
            dtype=np.float64,
        )
        wrong = replace(support, support_upper=wrong_array)
        reference = weakref.ref(wrong)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        try:
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=wrong,
            )
        except adapter.PreparedNumericAdapterError as exc:
            self.assertEqual(exc.code, "POST_CONSUME_FAILURE")
        else:
            self.fail("wrong prepared Dense support was accepted")
        stats = port.stats()
        self.assertEqual(stats["state"], "POISONED")
        self.assertEqual(stats["slot_state"], "IDLE")
        self.assertEqual(stats["dispatch_arms"], 1)
        self.assertEqual(stats["dispatch_consumes"], 1)
        self.assertEqual(stats["post_consume_failures"], 1)
        self.assertEqual(stats["locator_count"], 0)
        self.assertEqual(stats["frozen_material_count"], 0)
        del wrong
        for _ in range(3):
            gc.collect()
        self.assertIsNone(reference())

    def test_each_dense_numeric_field_is_independently_rebuilt(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        changed_support = np.ascontiguousarray(
            support.support_upper.copy()
        )
        changed_support[0] = changed_support[0] + 1.0
        changed_support = _immutable_copy(changed_support)

        def changed_exponent(value):
            return 0 if value is None else value + 1

        changes = {
            "support_upper": changed_support,
            "box_mass_upper": support.box_mass_upper + 1.0,
            "weight_shape": (
                support.weight_shape[0] + 1,
                support.weight_shape[1],
            ),
            "weight_exponent_min": changed_exponent(
                support.weight_exponent_min
            ),
            "weight_exponent_max": changed_exponent(
                support.weight_exponent_max
            ),
            "support_exponent_min": changed_exponent(
                support.support_exponent_min
            ),
            "support_exponent_max": changed_exponent(
                support.support_exponent_max
            ),
            "max_abs_exponent_min": changed_exponent(
                support.max_abs_exponent_min
            ),
            "max_abs_exponent_max": changed_exponent(
                support.max_abs_exponent_max
            ),
            "global_underflow_risk": (
                not support.global_underflow_risk
            ),
            "global_subnormal_operand": (
                not support.global_subnormal_operand
            ),
            "disjoint_box_mass": not support.disjoint_box_mass,
            "proof_authority": True,
        }
        for field, changed in changes.items():
            with self.subTest(field=field):
                port = adapter.create_prepared_numeric_adapter(
                    deadline=_end()
                )
                wrong = _exact_clone(support, **{field: changed})
                with self.assertRaisesRegex(
                    adapter.PreparedNumericAdapterError,
                    "POST_CONSUME_FAILURE",
                ):
                    port.admit_dense_prepared(
                        weight=weight,
                        predecessor_max_abs=max_abs,
                        raw_binding=_binding(adapter.BRANCH_DENSE),
                        prepared_support=wrong,
                    )
                stats = port.stats()
                self.assertEqual(stats["dispatch_consumes"], 1)
                self.assertEqual(stats["frozen_material_count"], 0)

    def test_wrong_conv_fields_fail_after_consume_without_core(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        wrong = replace(plan, offsets=())
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "POST_CONSUME_FAILURE",
        ):
            port.admit_conv_prepared(
                **raw,
                raw_binding=_binding(adapter.BRANCH_CONV_DENSE),
                prepared_plan=wrong,
            )
        stats = port.stats()
        self.assertEqual(stats["dispatch_consumes"], 1)
        self.assertEqual(stats["post_consume_failures"], 1)
        self.assertEqual(stats["frozen_conv_admissions"], 0)
        self.assertEqual(stats["frozen_material_count"], 0)

    def test_each_conv_plan_and_offset_field_is_independently_rebuilt(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)

        changed_weight = np.ascontiguousarray(plan.weight.copy())
        changed_weight.reshape(-1)[0] += 1.0
        changed_weight = _immutable_copy(changed_weight)
        changed_support = np.ascontiguousarray(plan.support.copy())
        changed_support[0] += 1.0
        changed_support = _immutable_copy(changed_support)
        plan_changes = {
            "layer_id": plan.layer_id + 1,
            "input_shape": (
                plan.input_shape[0] + 1,
                *plan.input_shape[1:],
            ),
            "output_shape": (
                plan.output_shape[0] + 1,
                *plan.output_shape[1:],
            ),
            "stride": (plan.stride[0] + 1, plan.stride[1]),
            "padding": (plan.padding[0] + 1, plan.padding[1]),
            "dilation": (
                plan.dilation[0] + 1,
                plan.dilation[1],
            ),
            "groups": plan.groups + 1,
            "weight": changed_weight,
            "support": changed_support,
            "offsets": (),
            "proof_authority": True,
        }

        first = plan.offsets[0]

        def changed_array(value):
            result = np.ascontiguousarray(value.copy())
            flattened = result.reshape(-1)
            if result.dtype == np.dtype(np.bool_):
                flattened[0] = not bool(flattened[0])
            else:
                flattened[0] += 1
            return _immutable_copy(result)

        offset_changes = {
            "group": first.group + 1,
            "kh": first.kh + 1,
            "kw": first.kw + 1,
            "co_start": first.co_start + 1,
            "co_end": first.co_end + 1,
            "ci_start": first.ci_start + 1,
            "ci_end": first.ci_end + 1,
            "output_h_indices": changed_array(
                first.output_h_indices
            ),
            "output_w_indices": changed_array(
                first.output_w_indices
            ),
            "targets": changed_array(first.targets),
            "support_flat": changed_array(first.support_flat),
            "channel_support_flat": changed_array(
                first.channel_support_flat
            ),
            "support_activity_flat": changed_array(
                first.support_activity_flat
            ),
            "support_sum_upper": first.support_sum_upper + 1.0,
        }
        cases = [
            ("plan.{}".format(field), _exact_clone(plan, **{field: value}))
            for field, value in plan_changes.items()
        ]
        for field, changed in offset_changes.items():
            wrong_offset = _exact_clone(first, **{field: changed})
            cases.append(
                (
                    "offset.{}".format(field),
                    _exact_clone(
                        plan,
                        offsets=(wrong_offset, *plan.offsets[1:]),
                    ),
                )
            )
        for field, wrong in cases:
            with self.subTest(field=field):
                port = adapter.create_prepared_numeric_adapter(
                    deadline=_end()
                )
                with self.assertRaisesRegex(
                    adapter.PreparedNumericAdapterError,
                    "POST_CONSUME_FAILURE",
                ):
                    port.admit_conv_prepared(
                        **raw,
                        raw_binding=_binding(
                            adapter.BRANCH_CONV_DENSE
                        ),
                        prepared_plan=wrong,
                    )
                stats = port.stats()
                self.assertEqual(stats["dispatch_consumes"], 1)
                self.assertEqual(stats["frozen_material_count"], 0)

    def test_raw_binding_exact_shape_and_cross_kind_reject_preconsume(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        cases = (
            list(_binding(adapter.BRANCH_DENSE)),
            _binding(adapter.BRANCH_CONV_DENSE),
            _binding(adapter.BRANCH_DENSE)[:-1],
            (
                *(_binding(adapter.BRANCH_DENSE)[:7]),
                "A" * 64,
                *(_binding(adapter.BRANCH_DENSE)[8:]),
            ),
        )
        for index, raw_binding in enumerate(cases):
            with self.subTest(index=index):
                port = adapter.create_prepared_numeric_adapter(
                    deadline=_end()
                )
                with self.assertRaises(
                    adapter.PreparedNumericAdapterError
                ):
                    port.admit_dense_prepared(
                        weight=weight,
                        predecessor_max_abs=max_abs,
                        raw_binding=raw_binding,
                        prepared_support=support,
                    )
                stats = port.stats()
                self.assertEqual(stats["state"], "POISONED")
                self.assertEqual(stats["dispatch_arms"], 0)
                self.assertEqual(stats["dispatch_consumes"], 0)
                self.assertEqual(stats["rejected_operations"], 1)

    def test_duplicate_physical_binding_fails_after_consume_and_closes_core(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        first = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=support,
        )
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "POST_CONSUME_FAILURE",
        ):
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
        stats = port.stats()
        self.assertEqual(stats["state"], "POISONED")
        self.assertEqual(stats["dispatch_arms"], 2)
        self.assertEqual(stats["dispatch_consumes"], 2)
        self.assertEqual(stats["post_consume_failures"], 1)
        self.assertEqual(stats["locator_count"], 0)
        self.assertEqual(stats["frozen_locator_count"], 0)
        self.assertEqual(stats["frozen_material_count"], 0)
        with self.assertRaises(adapter.PreparedNumericAdapterError):
            port.execute_dense(first, np.zeros((1, 2), dtype=np.float64))

    def test_empty_reuse_and_changed_call_dispatch_fail_closed(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)

        empty_port = adapter.create_prepared_numeric_adapter(
            deadline=deadline
        )
        empty_dispatch = _dense_dispatch(empty_port)
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "DISPATCH_SLOT_MISMATCH",
        ):
            empty_dispatch(weight, max_abs, deadline=deadline)
        self.assertEqual(empty_port.stats()["state"], "POISONED")

        changed_call_port = adapter.create_prepared_numeric_adapter(
            deadline=_end()
        )
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "DISPATCH_CALL_MISMATCH",
        ):
            _dense_dispatch(changed_call_port)(weight)

        reuse_deadline = _end()
        reuse_port = adapter.create_prepared_numeric_adapter(
            deadline=reuse_deadline
        )
        reuse_port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=support,
        )
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "DISPATCH_SLOT_MISMATCH",
        ):
            _dense_dispatch(reuse_port)(
                weight, max_abs, deadline=reuse_deadline
            )
        self.assertEqual(reuse_port.stats()["locator_count"], 0)

    def test_conv_packet_shapes_and_kind_are_exact_one_shot_calls(self):
        deadline = _end()
        raw, _, plan = _conv_fixture(deadline)
        for changed_part in ("packet", "shapes"):
            with self.subTest(changed_part=changed_part):
                port = adapter.create_prepared_numeric_adapter(
                    deadline=deadline
                )
                cells = _closure_values(
                    port.admit_conv_prepared.__func__
                )
                binding = cells["_binding_copy"](
                    _binding(adapter.BRANCH_CONV_DENSE),
                    adapter.BRANCH_CONV_DENSE,
                    raw["layer_id"],
                )
                expected = cells["_conv_expected_call"](**raw)
                private_plan = cells["_copy_conv_plan"](plan)
                cells["_arm"](
                    kind=adapter.BRANCH_CONV_DENSE,
                    prepared=private_plan,
                    raw_binding=binding,
                    expected_call=expected,
                )
                packet, shapes = expected
                if changed_part == "packet":
                    packet = (
                        packet[0],
                        packet[1] + 1,
                        *packet[2:],
                    )
                else:
                    shapes = (
                        (shapes[0][0] + 1, *shapes[0][1:]),
                        *shapes[1:],
                    )
                with self.assertRaisesRegex(
                    adapter.PreparedNumericAdapterError,
                    "DISPATCH_CALL_MISMATCH",
                ):
                    _conv_dispatch(port)(
                        packet, shapes, deadline=deadline
                    )
                self.assertEqual(port.stats()["state"], "POISONED")
                self.assertEqual(
                    port.stats()["dispatch_consumes"], 0
                )

        weight, max_abs, _, support = _dense_fixture(deadline)
        kind_port = adapter.create_prepared_numeric_adapter(
            deadline=deadline
        )
        cells = _closure_values(
            kind_port.admit_dense_prepared.__func__
        )
        binding = cells["_binding_copy"](
            _binding(adapter.BRANCH_DENSE),
            adapter.BRANCH_DENSE,
            None,
        )
        cells["_arm"](
            kind=adapter.BRANCH_DENSE,
            prepared=cells["_copy_dense_support"](support),
            raw_binding=binding,
            expected_call=None,
        )
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "DISPATCH_SLOT_MISMATCH",
        ):
            _conv_dispatch(kind_port)((), (), deadline=deadline)
        self.assertEqual(kind_port.stats()["state"], "POISONED")
        del weight
        del max_abs


class PreparedAdapterExceptionIsolationTests(unittest.TestCase):
    def _assert_sanitized_exception(self, exc, forbidden):
        self.assertIsNone(exc.__cause__)
        self.assertIsNone(exc.__context__)
        self.assertTrue(exc.__suppress_context__)
        adapter_path = Path(adapter.__file__).resolve()
        adapter_frames = []
        traceback = exc.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if Path(frame.f_code.co_filename).resolve() == adapter_path:
                adapter_frames.append(frame)
                for value in frame.f_locals.values():
                    self.assertNotIsInstance(value, np.ndarray)
                    self.assertFalse(
                        type(value).__name__.startswith("_PreparedDense")
                    )
                    self.assertFalse(
                        type(value).__name__.startswith("_PreparedConv")
                    )
                    for forbidden_value in forbidden:
                        self.assertIsNot(value, forbidden_value)
            traceback = traceback.tb_next
        self.assertEqual(len(adapter_frames), 1)
        self.assertEqual(adapter_frames[0].f_code.co_freevars, ())
        self.assertIn(
            adapter_frames[0].f_code.co_name,
            {
                "admit_dense_prepared",
                "admit_conv_prepared",
                "execute_dense",
                "execute_conv",
                "stats",
                "close",
            },
        )

    def test_pre_and_post_consume_errors_have_scalar_public_graphs(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        raw_binding = _binding(adapter.BRANCH_DENSE)
        writable = np.ascontiguousarray(support.support_upper.copy())
        preconsume_wrong = _exact_clone(
            support, support_upper=writable
        )
        immutable_wrong_array = np.frombuffer(
            bytes(support.support_upper.nbytes),
            dtype=np.float64,
        )
        postconsume_wrong = _exact_clone(
            support, support_upper=immutable_wrong_array
        )
        cases = (
            ("INVALID_PREPARED_VALUE", preconsume_wrong),
            ("POST_CONSUME_FAILURE", postconsume_wrong),
        )
        for expected_code, wrong in cases:
            with self.subTest(code=expected_code):
                port = adapter.create_prepared_numeric_adapter(
                    deadline=_end()
                )
                caught = None
                try:
                    port.admit_dense_prepared(
                        weight=weight,
                        predecessor_max_abs=max_abs,
                        raw_binding=raw_binding,
                        prepared_support=wrong,
                    )
                except adapter.PreparedNumericAdapterError as exc:
                    caught = exc
                self.assertIsNotNone(caught)
                self.assertEqual(caught.code, expected_code)
                self._assert_sanitized_exception(
                    caught,
                    (
                        port,
                        weight,
                        max_abs,
                        raw_binding,
                        wrong,
                        writable,
                        immutable_wrong_array,
                    ),
                )

    def test_unexpected_execution_error_uses_type_name_without_str(self):
        deadline = _end()
        weight, max_abs, coefficients, support = _dense_fixture(
            deadline
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        locator = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=support,
        )
        str_calls = []

        class LoudFailure(Exception):
            def __str__(self):
                str_calls.append(True)
                raise AssertionError("unexpected __str__ call")

        def failing_execution(frozen_locator, values):
            del frozen_locator
            del values
            raise LoudFailure()

        _closure_cell(
            port.execute_dense.__func__, "_frozen_execute_dense"
        ).cell_contents = failing_execution
        caught = None
        try:
            port.execute_dense(locator, coefficients)
        except adapter.PreparedNumericAdapterError as exc:
            caught = exc
        self.assertIsNotNone(caught)
        self.assertEqual(caught.code, "FROZEN_EXECUTION_REJECTED")
        self.assertIn("LoudFailure", str(caught))
        self.assertEqual(str_calls, [])
        self._assert_sanitized_exception(
            caught, (port, locator, coefficients)
        )
        self.assertEqual(port.stats()["state"], "POISONED")

    def test_retained_public_exception_does_not_retain_materials(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        changed_array = np.frombuffer(
            bytes(support.support_upper.nbytes),
            dtype=np.float64,
        )
        wrong = _exact_clone(support, support_upper=changed_array)
        wrong_reference = weakref.ref(wrong)
        array_reference = weakref.ref(changed_array)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        retained = None
        try:
            port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=wrong,
            )
        except adapter.PreparedNumericAdapterError as exc:
            retained = exc
        self.assertIsNotNone(retained)
        self.assertEqual(retained.code, "POST_CONSUME_FAILURE")
        del wrong
        del changed_array
        for _ in range(4):
            gc.collect()
            if (
                wrong_reference() is None
                and array_reference() is None
            ):
                break
        self.assertIsNone(wrong_reference())
        self.assertIsNone(array_reference())
        self.assertIsNone(retained.__cause__)
        self.assertIsNone(retained.__context__)


class PreparedAdapterLifecycleTests(unittest.TestCase):
    def _admitted_dense(self):
        deadline = _end()
        weight, max_abs, coefficients, support = _dense_fixture(
            deadline
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        locator = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=support,
        )
        return port, locator, coefficients

    def test_locator_kind_generation_and_factory_mismatch_poison(self):
        first, locator, coefficients = self._admitted_dense()
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError, "LOCATOR_MISMATCH"
        ):
            first.execute_conv(locator, coefficients)
        self.assertEqual(first.stats()["state"], "POISONED")
        self.assertEqual(first.stats()["locator_count"], 0)

        second, second_locator, _ = self._admitted_dense()
        third, _, _ = self._admitted_dense()
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError, "LOCATOR_MISMATCH"
        ):
            third.execute_dense(second_locator, coefficients)
        self.assertEqual(third.stats()["state"], "POISONED")
        self.assertEqual(second.stats()["state"], "OPEN")

    def test_fixed_four_thread_contention_rejects_without_poison(self):
        port = adapter.create_prepared_numeric_adapter(deadline=_end())
        operation_lock = _adapter_operation_lock(port)
        self.assertTrue(operation_lock.acquire(False))
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(port.stats) for _ in range(4)]
                errors = []
                for future in futures:
                    with self.assertRaises(
                        adapter.PreparedNumericAdapterError
                    ) as raised:
                        future.result(timeout=5.0)
                    errors.append(raised.exception.code)
            self.assertEqual(
                errors, ["CONCURRENT_OPERATION"] * 4
            )
        finally:
            operation_lock.release()
        stats = port.stats()
        self.assertEqual(stats["state"], "OPEN")
        self.assertEqual(stats["rejected_operations"], 4)

    def test_real_execution_winner_survives_stats_and_close_losers(self):
        for loser_name in ("stats", "close"):
            with self.subTest(loser=loser_name):
                port, locator, coefficients = self._admitted_dense()
                execute_function = port.execute_dense.__func__
                read_cell = _closure_cell(
                    execute_function, "_read_frozen_snapshot"
                )
                original_read = read_cell.cell_contents
                entered = threading.Event()
                release = threading.Event()

                def blocking_read():
                    values = original_read()
                    entered.set()
                    if not release.wait(timeout=5.0):
                        raise AssertionError("winner barrier timed out")
                    return values

                read_cell.cell_contents = blocking_read
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        port.execute_dense, locator, coefficients
                    )
                    self.assertTrue(entered.wait(timeout=5.0))
                    loser = (
                        port.stats
                        if loser_name == "stats"
                        else port.close
                    )
                    with self.assertRaises(
                        adapter.PreparedNumericAdapterError
                    ) as raised:
                        loser()
                    self.assertEqual(
                        raised.exception.code, "CONCURRENT_OPERATION"
                    )
                    release.set()
                    result = future.result(timeout=5.0)
                read_cell.cell_contents = original_read
                self.assertEqual(
                    result[0], b"act.v51b.private.dense-result.v1"
                )
                stats = port.stats()
                self.assertEqual(stats["state"], "OPEN")
                self.assertEqual(stats["dense_executions"], 1)
                self.assertEqual(stats["rejected_operations"], 1)
                port.close()

    def test_close_rejects_stale_snapshot_and_keeps_live_counts_zero(self):
        port, locator, coefficients = self._admitted_dense()
        port.execute_dense(locator, coefficients)
        execute_cells = _closure_values(
            port.execute_dense.__func__
        )
        stale_values = execute_cells["_read_frozen_snapshot"]()
        commit_cells = _closure_values(
            execute_cells["_commit_execution"]
        )
        apply_snapshot = commit_cells[
            "_apply_frozen_snapshot_locked"
        ]
        lifecycle_lock = commit_cells["lifecycle_lock"]
        port.close()
        with lifecycle_lock:
            with self.assertRaisesRegex(
                adapter.PreparedNumericAdapterError,
                "FROZEN_STATS_MISMATCH",
            ):
                apply_snapshot(stale_values)
        first = port.stats()
        second = port.stats()
        self.assertEqual(first, second)
        self.assertEqual(first["state"], "CLOSED")
        for name in (
            "frozen_material_count",
            "frozen_locator_count",
            "frozen_dense_materials",
            "frozen_conv_materials",
        ):
            self.assertEqual(first[name], 0, name)
        # Admission/execution fields are historical counters, not live
        # object counts; retaining them after close is intentional.
        self.assertEqual(first["frozen_dense_admissions"], 1)
        self.assertEqual(first["frozen_dense_executions"], 1)

    def test_terminal_live_counts_stay_zero_if_frozen_close_raises(self):
        port, locator, coefficients = self._admitted_dense()
        port.execute_dense(locator, coefficients)
        close_cells = _closure_values(port.close.__func__)
        close_no_raise = close_cells["_close_frozen_no_raise"]
        frozen_close_ref = _closure_values(close_no_raise)[
            "frozen_close_ref"
        ]
        original_close = frozen_close_ref[0]

        def failing_close():
            raise RuntimeError("injected frozen close failure")

        frozen_close_ref[0] = failing_close
        try:
            port.close()
            first = port.stats()
            second = port.stats()
            self.assertEqual(first, second)
            self.assertEqual(first["state"], "CLOSED")
            self.assertEqual(first["locator_count"], 0)
            for name in (
                "frozen_material_count",
                "frozen_locator_count",
                "frozen_dense_materials",
                "frozen_conv_materials",
            ):
                self.assertEqual(first[name], 0, name)
        finally:
            frozen_close_ref[0] = original_close
            original_close()

    def test_publication_and_execution_deadlines_commit_nothing(self):
        deadline = _end()
        weight, max_abs, coefficients, support = _dense_fixture(
            deadline
        )
        admission_port = adapter.create_prepared_numeric_adapter(
            deadline=deadline
        )
        admit_function = admission_port.admit_dense_prepared.__func__
        mint_cell = _closure_cell(admit_function, "_mint_locator")
        original_mint = mint_cell.cell_contents
        mint_deadline_cell = _closure_cell(
            original_mint, "owner_deadline"
        )

        def expiring_mint(**kwargs):
            mint_deadline_cell.cell_contents = float(
                time.monotonic() - 1.0
            )
            return original_mint(**kwargs)

        mint_cell.cell_contents = expiring_mint
        with self.assertRaises(
            adapter.PreparedNumericAdapterTimeout
        ):
            admission_port.admit_dense_prepared(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
        admission_stats = admission_port.stats()
        self.assertEqual(admission_stats["state"], "POISONED")
        self.assertEqual(admission_stats["dense_admissions"], 0)
        self.assertEqual(admission_stats["locator_count"], 0)
        self.assertEqual(admission_stats["post_consume_failures"], 1)

        execution_port = adapter.create_prepared_numeric_adapter(
            deadline=_end()
        )
        locator = execution_port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(
                adapter.BRANCH_DENSE, physical="7"
            ),
            prepared_support=support,
        )
        execute_function = execution_port.execute_dense.__func__
        commit_cell = _closure_cell(
            execute_function, "_commit_execution"
        )
        original_commit = commit_cell.cell_contents
        commit_deadline_cell = _closure_cell(
            original_commit, "owner_deadline"
        )

        def expiring_commit(**kwargs):
            commit_deadline_cell.cell_contents = float(
                time.monotonic() - 1.0
            )
            return original_commit(**kwargs)

        commit_cell.cell_contents = expiring_commit
        with self.assertRaises(
            adapter.PreparedNumericAdapterTimeout
        ):
            execution_port.execute_dense(locator, coefficients)
        execution_stats = execution_port.stats()
        self.assertEqual(execution_stats["state"], "POISONED")
        self.assertEqual(execution_stats["dense_executions"], 0)
        self.assertEqual(execution_stats["locator_count"], 0)
        self.assertEqual(
            execution_stats["frozen_dense_executions"], 0
        )

    def test_consume_and_publication_epochs_survive_concurrent_poison(self):
        deadline = _end()
        weight, max_abs, _, support = _dense_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        admission_cells = _closure_values(
            port.admit_dense_prepared.__func__
        )
        binding = admission_cells["_binding_copy"](
            _binding(adapter.BRANCH_DENSE),
            adapter.BRANCH_DENSE,
            None,
        )
        private_support = admission_cells["_copy_dense_support"](support)
        wanted_epoch = admission_cells["_publication_epoch"]()
        wanted_generation = admission_cells["_arm"](
            kind=adapter.BRANCH_DENSE,
            prepared=private_support,
            raw_binding=binding,
            expected_call=None,
        )
        _dense_dispatch(port)(weight, max_abs, deadline=deadline)
        frozen_values = admission_cells["_read_frozen_snapshot"]()
        admission_cells["_poison"](
            rejected=True, post_consume=False
        )
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "POST_CONSUME_STATE_MISMATCH",
        ):
            admission_cells["_mint_locator"](
                frozen_locator=object(),
                kind=adapter.BRANCH_DENSE,
                raw_binding=binding,
                wanted_generation=wanted_generation,
                wanted_epoch=wanted_epoch,
                frozen_values=frozen_values,
            )
        with self.assertRaises(
            adapter.PreparedNumericAdapterError
        ):
            admission_cells["_admission_failure"](
                adapter.PreparedNumericAdapterError(
                    "SIMULATED_OWNER_FAILURE", "owner observed poison"
                ),
                wanted_generation,
            )
        stats = port.stats()
        self.assertEqual(stats["dispatch_consumes"], 1)
        self.assertEqual(stats["post_consume_failures"], 1)
        self.assertEqual(stats["dense_admissions"], 0)
        self.assertEqual(stats["locator_count"], 0)
        self.assertEqual(stats["state"], "POISONED")

        other = adapter.create_prepared_numeric_adapter(deadline=_end())
        execution_cells = _closure_values(other.execute_dense.__func__)
        wanted_epoch = execution_cells["_publication_epoch"]()
        frozen_values = execution_cells["_read_frozen_snapshot"]()
        failure_cells = _closure_values(
            execution_cells["_execution_failure"]
        )
        failure_cells["_poison"](
            rejected=True, post_consume=False
        )
        with self.assertRaisesRegex(
            adapter.PreparedNumericAdapterError,
            "CONCURRENT_POISON",
        ):
            execution_cells["_commit_execution"](
                kind=adapter.BRANCH_DENSE,
                wanted_epoch=wanted_epoch,
                frozen_values=frozen_values,
            )

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_pid_gate_precedes_an_inherited_locked_operation_lock(self):
        port = adapter.create_prepared_numeric_adapter(deadline=_end())
        operation_lock = _adapter_operation_lock(port)
        held = threading.Event()
        release = threading.Event()

        def holder():
            operation_lock.acquire()
            held.set()
            release.wait(timeout=10.0)
            operation_lock.release()

        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        self.assertTrue(held.wait(timeout=5.0))
        read_fd, write_fd = os.pipe()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This process .* is multi-threaded.*fork",
                )
                pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    port.stats()
                except adapter.PreparedNumericAdapterError as exc:
                    payload = exc.code.encode("ascii")
                except BaseException as exc:
                    payload = type(exc).__name__.encode("ascii")
                else:
                    payload = b"ACCEPTED"
                os.write(write_fd, payload)
                os.close(write_fd)
                os._exit(0)
            os.close(write_fd)
            write_fd = -1
            payload = os.read(read_fd, 128)
            waited, status = os.waitpid(pid, 0)
            self.assertEqual(waited, pid)
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(payload, b"FORKED_PROCESS")
        finally:
            if read_fd >= 0:
                os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            release.set()
            thread.join(timeout=5.0)

    def test_copy_pickle_gc_and_terminal_close(self):
        port, locator, coefficients = self._admitted_dense()
        result = port.execute_dense(locator, coefficients)
        for operation in (
            lambda: copy.copy(port),
            lambda: copy.deepcopy(port),
            lambda: pickle.dumps(port),
            lambda: copy.copy(locator),
            lambda: copy.deepcopy(locator),
            lambda: pickle.dumps(locator),
        ):
            with self.subTest(operation=repr(operation)):
                with self.assertRaises(
                    adapter.PreparedNumericAdapterError
                ):
                    operation()
        self.assertEqual(copy.copy(result), result)
        self.assertEqual(copy.deepcopy(result), result)
        self.assertEqual(pickle.loads(pickle.dumps(result)), result)

        locator_reference = weakref.ref(locator)
        del locator
        for _ in range(4):
            gc.collect()
            if locator_reference() is None:
                break
        self.assertIsNone(locator_reference())
        stats = port.stats()
        self.assertEqual(stats["locator_count"], 0)
        self.assertEqual(stats["frozen_locator_count"], 0)
        self.assertEqual(stats["frozen_material_count"], 0)
        port.close()
        port.close()
        self.assertEqual(port.stats()["state"], "CLOSED")

        disposable = adapter.create_prepared_numeric_adapter(
            deadline=_end()
        )
        disposable_reference = weakref.ref(disposable)
        del disposable
        for _ in range(4):
            gc.collect()
        self.assertIsNone(disposable_reference())

    def test_post_factory_changes_and_class_check_redirection_are_ignored(self):
        deadline = _end()
        weight, max_abs, coefficients, support = _dense_fixture(
            deadline
        )
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        captured_admit = port.admit_dense_prepared
        captured_execute = port.execute_dense
        calls = []

        def evil(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("changed dependency was called")

        with mock.patch.object(
            private, "create_private_numeric_kernel", evil
        ), mock.patch.object(
            dense_v51, "prepare_dense_support_v51", evil
        ), mock.patch.object(
            np, "frombuffer", evil
        ), mock.patch.object(
            time, "monotonic", evil
        ), mock.patch.object(
            dense_v51.DenseV51Support, "__getattribute__", evil
        ), mock.patch.object(
            type(port), "_check_port", evil, create=True
        ), mock.patch.object(
            type(port), "admit_dense_prepared", evil
        ):
            locator = captured_admit(
                weight=weight,
                predecessor_max_abs=max_abs,
                raw_binding=_binding(adapter.BRANCH_DENSE),
                prepared_support=support,
            )
            result = captured_execute(locator, coefficients)
        self.assertEqual(calls, [])
        self.assertEqual(
            result[0], b"act.v51b.private.dense-result.v1"
        )

    def test_stats_cross_consistency_and_no_material_values(self):
        deadline = _end()
        weight, max_abs, coefficients, support = _dense_fixture(
            deadline
        )
        raw, conv_coefficients, plan = _conv_fixture(deadline)
        port = adapter.create_prepared_numeric_adapter(deadline=deadline)
        dense_locator = port.admit_dense_prepared(
            weight=weight,
            predecessor_max_abs=max_abs,
            raw_binding=_binding(adapter.BRANCH_DENSE),
            prepared_support=support,
        )
        conv_locator = port.admit_conv_prepared(
            **raw,
            raw_binding=_binding(
                adapter.BRANCH_CONV_DENSE, physical="7"
            ),
            prepared_plan=plan,
        )
        port.execute_dense(dense_locator, coefficients)
        port.execute_conv(conv_locator, conv_coefficients)
        stats = port.stats()
        self.assertIs(type(stats), MappingProxyType)
        self.assertEqual(stats["slot_state"], "IDLE")
        self.assertEqual(stats["dispatch_arms"], 2)
        self.assertEqual(stats["dispatch_consumes"], 2)
        self.assertEqual(stats["locator_count"], 2)
        self.assertEqual(
            stats["locator_count"], stats["frozen_locator_count"]
        )
        self.assertEqual(
            stats["dense_admissions"],
            stats["frozen_dense_admissions"],
        )
        self.assertEqual(
            stats["conv_admissions"],
            stats["frozen_conv_admissions"],
        )
        self.assertEqual(
            stats["dense_executions"],
            stats["frozen_dense_executions"],
        )
        self.assertEqual(
            stats["conv_executions"],
            stats["frozen_conv_executions"],
        )
        self.assertFalse(stats["proof_authority"])
        for value in stats.values():
            self.assertNotIsInstance(value, np.ndarray)
            self.assertNotIsInstance(value, dense_v51.DenseV51Support)
            self.assertNotIsInstance(value, conv_v51.DenseConvV51Plan)
            self.assertNotIn("locator>", repr(value))


if __name__ == "__main__":
    unittest.main()
