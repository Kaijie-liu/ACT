#!/usr/bin/env python3
"""Focused Phase-B audits for the optional Operator constraint sink.

The production switch is deliberately internal and default-false.  These
tests enable it only under a process-local patch; no solver consumes the
sealed program and no real/large model is run.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import unittest
import weakref
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import operator_hz as operator
from act.back_end.solver import constraint_program as core


_DTYPE = torch.float64


class _Toy:
    def __init__(self, net: Any, facts: Mapping[int, Fact]) -> None:
        self.net = net
        self.facts = facts


class _DeterministicAllocator:
    def __init__(self, start: int = 1000) -> None:
        self.next = int(start)
        self.calls = []

    def __call__(self, count: int, device: Any = None) -> torch.Tensor:
        count = int(count)
        self.calls.append(count)
        start = self.next
        self.next += count
        return torch.arange(
            start, self.next, dtype=torch.long, device=device
        )


def _layer(
    layer_id: int,
    kind: str,
    width: int,
    params: Mapping[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[(int(layer_id), row) for row in range(int(width))],
    )


def _dense(
    layer_id: int,
    matrix: Sequence[Sequence[float]],
    bias: Sequence[float] | None = None,
) -> Any:
    weights = np.asarray(matrix, dtype=np.float64)
    if weights.ndim != 2:
        raise AssertionError("toy DENSE matrix must be rank two")
    values = (
        np.zeros(weights.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if values.size != weights.shape[0]:
        raise AssertionError("toy DENSE bias width changed")
    return _layer(
        layer_id,
        "DENSE",
        int(weights.shape[0]),
        {
            "weight": torch.tensor(weights, dtype=_DTYPE),
            "bias": torch.tensor(values, dtype=_DTYPE),
            "in_features": int(weights.shape[1]),
            "out_features": int(weights.shape[0]),
        },
    )


def _toy(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    *,
    input_lower: Sequence[float],
    input_upper: Sequence[float],
) -> _Toy:
    lower = np.asarray(input_lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(input_upper, dtype=np.float64).reshape(-1)
    if lower.shape != upper.shape or np.any(lower > upper):
        raise AssertionError("malformed toy input bounds")
    pred_map = {
        int(layer.id): tuple(int(value) for value in preds[int(layer.id)])
        for layer in layers
    }
    succs: Dict[int, list[int]] = {int(layer.id): [] for layer in layers}
    for child, parents in pred_map.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=list(layers),
        preds=pred_map,
        succs=succs,
        by_id={int(layer.id): layer for layer in layers},
    )
    facts: Dict[int, Fact] = {}
    for layer in layers:
        width = len(layer.out_vars)
        if str(layer.kind).upper() in {"INPUT", "INPUT_SPEC"}:
            lb = lower
            ub = upper
        else:
            lb = np.full(width, -1.0e6, dtype=np.float64)
            ub = np.full(width, 1.0e6, dtype=np.float64)
        facts[int(layer.id)] = Fact(
            Bounds(
                torch.tensor(lb.reshape(1, -1), dtype=_DTYPE),
                torch.tensor(ub.reshape(1, -1), dtype=_DTYPE),
            ),
            ConSet(),
        )
    return _Toy(net, facts)


def _input_layers(
    width: int,
    lower: Sequence[float],
    upper: Sequence[float],
) -> Tuple[Any, Any]:
    lb = torch.tensor(
        np.asarray(lower, dtype=np.float64).reshape(1, -1),
        dtype=_DTYPE,
    )
    ub = torch.tensor(
        np.asarray(upper, dtype=np.float64).reshape(1, -1),
        dtype=_DTYPE,
    )
    return (
        _layer(0, "INPUT", width, {"shape": (1, int(width))}),
        _layer(
            1,
            "INPUT_SPEC",
            width,
            {"kind": "BOX", "lb": lb, "ub": ub},
        ),
    )


def _identity_toy(
    lower: Sequence[float], upper: Sequence[float]
) -> _Toy:
    width = len(tuple(lower))
    input_layer, spec = _input_layers(width, lower, upper)
    layers = [
        input_layer,
        spec,
        _layer(2, "ASSERT", width, {"kind": "UNSAFE_LINEAR"}),
    ]
    return _toy(
        layers,
        {0: (), 1: (0,), 2: (1,)},
        input_lower=lower,
        input_upper=upper,
    )


def _add_toy(width: int, *, with_exact_relu: bool) -> _Toy:
    lower = [-1.0] * int(width)
    upper = [1.0] * int(width)
    input_layer, spec = _input_layers(width, lower, upper)
    identity = np.eye(width, dtype=np.float64)
    half = 0.5 * identity
    layers = [
        input_layer,
        spec,
        _dense(2, identity),
        _dense(3, half),
        _layer(4, "ADD", width),
    ]
    preds: Dict[int, Sequence[int]] = {
        0: (),
        1: (0,),
        2: (1,),
        3: (1,),
        4: (2, 3),
    }
    if with_exact_relu:
        layers.extend(
            [
                _layer(5, "RELU", width),
                _layer(6, "ASSERT", width, {"kind": "UNSAFE_LINEAR"}),
            ]
        )
        preds.update({5: (4,), 6: (5,)})
    else:
        layers.append(
            _layer(5, "ASSERT", width, {"kind": "UNSAFE_LINEAR"})
        )
        preds[5] = (4,)
    return _toy(
        layers,
        preds,
        input_lower=lower,
        input_upper=upper,
    )


def _exact_tag_toy() -> _Toy:
    lower = [1.0]
    upper = [2.0]
    input_layer, spec = _input_layers(1, lower, upper)
    layers = [
        input_layer,
        spec,
        _dense(2, [[1.0]]),
        _dense(3, [[1.0]]),
        _layer(4, "RELU", 1),
        _layer(5, "ASSERT", 1, {"kind": "UNSAFE_LINEAR"}),
    ]
    return _toy(
        layers,
        {0: (), 1: (0,), 2: (1,), 3: (2,), 4: (3,), 5: (4,)},
        input_lower=lower,
        input_upper=upper,
    )


def _property_tail_toy() -> _Toy:
    lower = [-1.0]
    upper = [1.0]
    input_layer, spec = _input_layers(1, lower, upper)
    layers = [
        input_layer,
        spec,
        _layer(2, "RELU", 1),
        _dense(3, [[1.0]]),
        _layer(4, "ASSERT", 1, {"kind": "UNSAFE_LINEAR"}),
    ]
    return _toy(
        layers,
        {0: (), 1: (0,), 2: (1,), 3: (2,), 4: (3,)},
        input_lower=lower,
        input_upper=upper,
    )


def _raw_ids(values: Iterable[Any]) -> Tuple[int, ...]:
    return tuple(int(value.raw_id) for value in values)


def _build(
    toy: _Toy,
    *,
    enabled: bool,
    allocator: _DeterministicAllocator,
    **kwargs: Any,
) -> operator.OperatorHZBuild:
    options = {"exact_budget": -1, "materialize_add": True}
    options.update(kwargs)
    with mock.patch.object(
        operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", enabled
    ), mock.patch.object(operator, "hz_fresh_col_ids", new=allocator):
        return operator.build_operator_hz(
            toy.net, toy.facts, toy.facts, **options
        )


def _bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and np.ascontiguousarray(a).tobytes()
        == np.ascontiguousarray(b).tobytes()
    )


def _csr_bytes_equal(left: sp.csr_matrix, right: sp.csr_matrix) -> bool:
    a = left.tocsr(copy=False)
    b = right.tocsr(copy=False)
    return (
        a.shape == b.shape
        and _bytes_equal(a.indptr, b.indptr)
        and _bytes_equal(a.indices, b.indices)
        and _bytes_equal(a.data, b.data)
    )


def _replay_legacy(
    program: Any,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, Tuple[str, ...]]:
    continuous = []
    binary = []
    upper = []
    tags = []
    for batch in program.iter_legacy_facet_batches(max_rows=2):
        continuous.append(batch.A_cont)
        binary.append(batch.A_bin)
        upper.append(batch.upper)
        tags.extend(str(value) for value in batch.row_tags)
    n_cont = len(program.continuous_ids)
    n_bin = len(program.binary_ids)
    return (
        sp.vstack(continuous, format="csr")
        if continuous
        else sp.csr_matrix((0, n_cont), dtype=np.float64),
        sp.vstack(binary, format="csr")
        if binary
        else sp.csr_matrix((0, n_bin), dtype=np.float64),
        np.concatenate(upper)
        if upper
        else np.zeros(0, dtype=np.float64),
        tuple(tags),
    )


class _BuildAbort(BaseException):
    pass


class OperatorConstraintProgramSinkTests(unittest.TestCase):
    def test_default_build_does_not_import_constraint_program(self) -> None:
        root = Path(__file__).resolve().parents[3]
        script = r'''
import sys
import numpy as np
import torch
from types import SimpleNamespace
from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import operator_hz as oh
name = "act.back_end.solver.constraint_program"
assert oh._EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK is False
assert name not in sys.modules
def layer(i, kind, params):
    return SimpleNamespace(id=i, kind=kind, params=params, in_vars=[], out_vars=[i])
zero = torch.tensor([[0.0]], dtype=torch.float64)
one = torch.tensor([[1.0]], dtype=torch.float64)
layers = [
    layer(0, "INPUT", {"shape": (1, 1)}),
    layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": zero, "ub": one}),
    layer(2, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
]
net = SimpleNamespace(
    layers=layers, preds={0: [], 1: [0], 2: [1]},
    succs={0: [1], 1: [2], 2: []}, by_id={x.id: x for x in layers},
)
facts = {i: Fact(Bounds(zero.clone(), one.clone()), ConSet()) for i in range(3)}
built = oh.build_operator_hz(net, facts, facts, exact_budget=-1)
assert built.constraint_program is None
assert name not in sys.modules
'''
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_all_unsupported_block_consumers_fail_before_owner(self) -> None:
        toy = _identity_toy([-1.0], [1.0])
        cases = (
            (toy, {"exact_budget": 0}),
            (
                toy,
                {
                    "exact_budget": -1,
                    "preactivation_lp_budget": 1,
                    "preactivation_lp_time_limit": 1.0,
                },
            ),
            (
                _property_tail_toy(),
                {
                    "exact_budget": -1,
                    "property_upper_C": np.ones((1, 1), dtype=np.float64),
                    "property_upper_thresholds": np.zeros(
                        1, dtype=np.float64
                    ),
                },
            ),
            (
                toy,
                {
                    "exact_budget": -1,
                    "property_micro_rlt_product_cap": 1,
                },
            ),
        )
        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            core, "ConstraintProgramOwner"
        ) as owner_constructor, mock.patch.object(
            operator, "hz_fresh_col_ids"
        ) as allocator:
            for case_toy, options in cases:
                with self.subTest(options=options):
                    with self.assertRaisesRegex(
                        operator.OperatorHZBuildError,
                        "constraint-program sink preflight rejected",
                    ):
                        operator.build_operator_hz(
                            case_toy.net,
                            case_toy.facts,
                            case_toy.facts,
                            **options,
                        )
            owner_constructor.assert_not_called()
            allocator.assert_not_called()

    def test_input_full_reservation_claims_only_live_factor_ids(self) -> None:
        mixed_allocator = _DeterministicAllocator(100)
        mixed = _build(
            _identity_toy([-1.0, 2.0, -3.0], [1.0, 2.0, -3.0]),
            enabled=True,
            allocator=mixed_allocator,
        )
        self.assertEqual(mixed_allocator.calls, [3])
        self.assertEqual(tuple(mixed.input_col_ids), (100, 101, 102))
        self.assertEqual(tuple(mixed.hz.col_ids), (100,))
        self.assertEqual(
            _raw_ids(mixed.constraint_program.continuous_ids), (100,)
        )
        self.assertEqual(mixed.constraint_program.source_rows, 0)

        point_allocator = _DeterministicAllocator(200)
        point = _build(
            _identity_toy([4.0, -2.0, 7.0], [4.0, -2.0, 7.0]),
            enabled=True,
            allocator=point_allocator,
        )
        self.assertEqual(point_allocator.calls, [3])
        self.assertEqual(tuple(point.input_col_ids), (200, 201, 202))
        self.assertEqual(point.hz.n_cont, 0)
        self.assertEqual(point.constraint_program.continuous_ids, ())

    def test_add_is_only_range_and_exact_tag_equalities_stay_le(self) -> None:
        ranged = _build(
            _add_toy(3, with_exact_relu=False),
            enabled=True,
            allocator=_DeterministicAllocator(),
        )
        program = ranged.constraint_program
        self.assertEqual(program.block_count, 1)
        self.assertEqual(program.source_rows, 3)
        self.assertEqual(program.virtual_facet_rows, 6)
        self.assertEqual(program.ranged_rows, 3)
        self.assertEqual(program.fallback_pairs, 0)
        self.assertEqual(program.virtual_facet_nnz, 2 * program.source_nnz)
        native_tags = tuple(
            str(tag)
            for batch in program.iter_native_batches(max_rows=2)
            for tag in batch.row_tags
        )
        self.assertEqual(native_tags, ("range:add_materialize:4",) * 3)
        replay_Ac, replay_Ab, replay_ub, replay_tags = _replay_legacy(
            program
        )
        self.assertEqual(
            replay_tags,
            ("add_materialize:4:forward",) * 3
            + ("add_materialize:4:reverse",) * 3,
        )
        self.assertTrue(_csr_bytes_equal(replay_Ac, ranged.hz.Auc))
        self.assertTrue(_csr_bytes_equal(replay_Ab, ranged.hz.Aub))
        self.assertTrue(_bytes_equal(replay_ub, ranged.hz.ub))
        self.assertEqual(ranged.hz.n_eq + ranged.hz.n_ub, 6)

        tagged = _build(
            _exact_tag_toy(),
            enabled=True,
            allocator=_DeterministicAllocator(500),
        )
        exact_program = tagged.constraint_program
        exact_tags = tuple(
            str(tag)
            for batch in exact_program.iter_native_batches(max_rows=2)
            for tag in batch.row_tags
        )
        self.assertEqual(
            exact_tags,
            (
                "affine_chain_cut:2:forward",
                "affine_chain_cut:2:reverse",
                "relu_active:4:forward",
                "relu_active:4:reverse",
            ),
        )
        self.assertEqual(exact_program.ranged_rows, 0)
        self.assertEqual(
            exact_program.source_rows, exact_program.virtual_facet_rows
        )

    def test_enabled_final_hz_tags_ids_and_metadata_are_bitwise_legacy(self) -> None:
        toy = _add_toy(2, with_exact_relu=True)
        with mock.patch.object(operator.time, "monotonic", return_value=123.0):
            legacy = _build(
                toy,
                enabled=False,
                allocator=_DeterministicAllocator(1000),
            )
            enabled = _build(
                toy,
                enabled=True,
                allocator=_DeterministicAllocator(1000),
            )
        self.assertIsNone(legacy.constraint_program)
        for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
            self.assertTrue(
                _bytes_equal(getattr(legacy.hz, name), getattr(enabled.hz, name)),
                msg=name,
            )
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            self.assertTrue(
                _csr_bytes_equal(
                    getattr(legacy.hz, name), getattr(enabled.hz, name)
                ),
                msg=name,
            )
        self.assertTrue(
            _bytes_equal(legacy.input_col_ids, enabled.input_col_ids)
        )
        self.assertEqual(legacy.metadata, enabled.metadata)
        self.assertEqual(
            legacy.hz.operator_hz_metadata,
            enabled.hz.operator_hz_metadata,
        )
        replay_Ac, replay_Ab, replay_ub, replay_tags = _replay_legacy(
            enabled.constraint_program
        )
        self.assertTrue(_csr_bytes_equal(replay_Ac, enabled.hz.Auc))
        self.assertTrue(_csr_bytes_equal(replay_Ab, enabled.hz.Aub))
        self.assertTrue(_bytes_equal(replay_ub, enabled.hz.ub))
        expected_tags = tuple(
            item["tag"]
            for item in enabled.metadata["constraint_tags_ub"]
            for _row in range(int(item["rows"]))
        )
        self.assertEqual(replay_tags, expected_tags)
        self.assertEqual(
            enabled.constraint_program.virtual_facet_rows,
            enabled.hz.n_eq + enabled.hz.n_ub,
        )
        self.assertLess(
            enabled.constraint_program.source_rows,
            enabled.constraint_program.virtual_facet_rows,
        )

    def test_adapter_initialize_public_return_keeps_recoverable_handle(
        self,
    ) -> None:
        toy = _identity_toy([-1.0], [1.0])
        abort = _BuildAbort("adapter initialize public return")
        captured: Dict[str, Any] = {}
        original_sink = operator._OperatorConstraintProgramSink.initialize
        original_initialize = core.ExternalFactorAllocatorAdapter.initialize

        def capture_sink(sink: Any) -> None:
            captured["sink"] = sink
            original_sink(sink)

        def interrupted(adapter: Any, *args: Any, **kwargs: Any) -> None:
            original_initialize(adapter, *args, **kwargs)
            captured["adapter"] = adapter
            raise abort

        gc.collect()
        adapters_before = set(core._ADAPTER_REGISTRY)
        bindings_before = set(core._ALLOCATOR_BINDINGS)
        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "initialize",
            new=capture_sink,
        ), mock.patch.object(
            core.ExternalFactorAllocatorAdapter,
            "initialize",
            new=interrupted,
        ):
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        sink = captured["sink"]
        adapter = captured["adapter"]
        self.assertIs(sink.adapter, adapter)
        self.assertEqual(sink.phase, "binding")
        self.assertIsNone(sink.owner)
        self.assertIsNotNone(adapter.namespace_identity)
        adapter_id = id(adapter)
        binding_id = id(sink.bridge)
        self.assertIn(adapter_id, core._ADAPTER_REGISTRY)
        self.assertIn(binding_id, core._ALLOCATOR_BINDINGS)

        reference = weakref.ref(adapter)
        raised.exception.__traceback__ = None
        captured.clear()
        del adapter
        del sink
        del raised
        del abort
        for _attempt in range(4):
            gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(adapter_id, core._ADAPTER_REGISTRY)
        self.assertNotIn(binding_id, core._ALLOCATOR_BINDINGS)
        self.assertEqual(set(core._ADAPTER_REGISTRY), adapters_before)
        self.assertEqual(set(core._ALLOCATOR_BINDINGS), bindings_before)

    def test_owner_initialize_public_return_recovers_then_discards(self) -> None:
        toy = _identity_toy([-1.0], [1.0])
        abort = _BuildAbort("owner initialize public return")
        captured: Dict[str, Any] = {}
        original_sink = operator._OperatorConstraintProgramSink.initialize
        original_initialize = core.ConstraintProgramOwner.initialize

        def capture_sink(sink: Any) -> None:
            captured["sink"] = sink
            original_sink(sink)

        def interrupted(owner: Any, adapter: Any) -> None:
            original_initialize(owner, adapter)
            captured["owner"] = owner
            raise abort

        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "initialize",
            new=capture_sink,
        ), mock.patch.object(
            core.ConstraintProgramOwner,
            "initialize",
            new=interrupted,
        ):
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        sink = captured["sink"]
        self.assertIs(sink.owner, captured["owner"])
        self.assertEqual(sink.phase, "discarded")
        self.assertTrue(sink.owner.discarded)
        self.assertTrue(sink.arena.discarded)
        self.assertIsNone(sink.program)

    def test_cleanup_secondary_never_replaces_primary_build_error(self) -> None:
        toy = _exact_tag_toy()
        primary = _BuildAbort("primary build failure")
        secondary = _BuildAbort("secondary cleanup failure")
        captured: Dict[str, Any] = {}
        original_append = operator._OperatorConstraintProgramSink.append_le
        original_cleanup = (
            operator._OperatorHZBuilder.
            _discard_open_constraint_program_sink
        )

        def fail_after_append(
            sink: Any, block: Any, *, layer_id: int
        ) -> int:
            captured["sink"] = sink
            original_append(sink, block, layer_id=layer_id)
            raise primary

        def cleanup_then_fail(builder: Any) -> Any:
            captured["cleanup_result"] = original_cleanup(builder)
            raise secondary

        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "append_le",
            new=fail_after_append,
        ), mock.patch.object(
            operator._OperatorHZBuilder,
            "_discard_open_constraint_program_sink",
            new=cleanup_then_fail,
        ):
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, primary)
        self.assertIsNone(captured["cleanup_result"])
        self.assertEqual(captured["sink"].phase, "discarded")
        self.assertTrue(captured["sink"].arena.discarded)
        notes = tuple(getattr(primary, "__notes__", ()))
        self.assertTrue(any("secondary cleanup failure" in note for note in notes))

    def test_replay_body_and_close_failures_preserve_primary_and_close(
        self,
    ) -> None:
        toy = _add_toy(129, with_exact_relu=False)
        primary = _BuildAbort("replay consumer body failure")
        secondary = _BuildAbort("replay cursor close failure")
        captured: Dict[str, Any] = {"close_calls": 0}
        original_sink = operator._OperatorConstraintProgramSink.initialize
        original_iter = core.ConstraintProgram.iter_legacy_facet_batches

        class BatchProxy:
            def __init__(self, batch: Any) -> None:
                self.batch = batch

            @property
            def row_offset(self) -> int:
                raise primary

            def __getattr__(self, name: str) -> Any:
                return getattr(self.batch, name)

        class CursorProxy:
            def __init__(self, cursor: Any) -> None:
                self.cursor = cursor
                self.first = True

            def __iter__(self) -> "CursorProxy":
                return self

            def __next__(self) -> Any:
                batch = next(self.cursor)
                if self.first:
                    self.first = False
                    return BatchProxy(batch)
                return batch

            def close(self) -> None:
                captured["close_calls"] += 1
                self.cursor.close()
                raise secondary

        def capture_sink(sink: Any) -> None:
            original_sink(sink)
            captured["sink"] = sink

        def proxy_iter(program: Any, *, max_rows: int) -> Any:
            cursor = original_iter(program, max_rows=max_rows)
            captured.update(program=program, cursor=cursor)
            return CursorProxy(cursor)

        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "initialize",
            new=capture_sink,
        ), mock.patch.object(
            core.ConstraintProgram,
            "iter_legacy_facet_batches",
            new=proxy_iter,
        ), mock.patch.object(
            operator, "_assemble_owned_operator_sparse_hz"
        ) as constructor:
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, primary)
        constructor.assert_not_called()
        self.assertEqual(captured["sink"].phase, "sealed")
        self.assertIs(captured["sink"].program, captured["program"])
        self.assertTrue(captured["program"].representation_authority)
        self.assertEqual(captured["program"].virtual_facet_rows, 258)
        self.assertEqual(captured["close_calls"], 4)
        self.assertTrue(captured["cursor"].closed)
        self.assertIsNone(core._iterator_state(captured["cursor"]).capture)
        notes = tuple(getattr(primary, "__notes__", ()))
        self.assertTrue(any("replay cursor close failure" in note for note in notes))

    def test_new_arena_public_return_abort_recovers_then_discards(self) -> None:
        toy = _identity_toy([-1.0], [1.0])
        abort = _BuildAbort("new-arena public return")
        original = core.ConstraintProgramOwner.new_arena
        captured: Dict[str, Any] = {}
        first = [True]

        def interrupted(owner: Any) -> Any:
            arena = original(owner)
            captured.update(owner=owner, arena=arena)
            if first[0]:
                first[0] = False
                raise abort
            return arena

        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            core.ConstraintProgramOwner, "new_arena", new=interrupted
        ), mock.patch.object(
            operator,
            "hz_fresh_col_ids",
            new=_DeterministicAllocator(),
        ):
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        self.assertTrue(captured["arena"].discarded)
        self.assertTrue(captured["owner"].discarded)

    def test_post_commit_preseal_baseexception_terminally_discards(self) -> None:
        toy = _exact_tag_toy()
        abort = _BuildAbort("after committed LE")
        captured: Dict[str, Any] = {}
        original = operator._OperatorConstraintProgramSink.append_le

        def interrupted(
            sink: Any, block: Any, *, layer_id: int
        ) -> int:
            captured["sink"] = sink
            original(sink, block, layer_id=layer_id)
            raise abort

        programs_before = set(core._PROGRAM_REGISTRY)
        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "append_le",
            new=interrupted,
        ), mock.patch.object(
            operator,
            "hz_fresh_col_ids",
            new=_DeterministicAllocator(),
        ), mock.patch.object(
            operator, "_assemble_owned_operator_sparse_hz"
        ) as constructor:
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        constructor.assert_not_called()
        sink = captured["sink"]
        self.assertEqual(sink.phase, "discarded")
        self.assertTrue(sink.arena.discarded)
        self.assertTrue(sink.owner.discarded)
        self.assertIsNone(sink.program)
        state = core._arena_state(sink.arena)
        self.assertEqual(state.blocks, {})
        self.assertEqual(state.pending, [])
        self.assertEqual(state.prepared, {})
        self.assertEqual(set(core._PROGRAM_REGISTRY), programs_before)
        continuous, binary = sink.bridge.snapshot()
        self.assertTrue(continuous)
        self.assertEqual(binary, ())

    def test_add_range_pair_mismatch_discards_instead_of_fallback(self) -> None:
        toy = _add_toy(2, with_exact_relu=False)
        captured: Dict[str, Any] = {}
        original_sink = (
            operator._OperatorConstraintProgramSink.
            append_add_materialize_range
        )
        original_core = core.ConstraintArena.append_guarded_band

        def mismatch(
            sink: Any,
            forward: Any,
            reverse: Any,
            *,
            layer_id: int,
        ) -> int:
            captured["sink"] = sink
            changed = reverse.Ac.copy().tocsr()
            if changed.nnz == 0:
                raise AssertionError("ADD toy unexpectedly has no coefficient")
            changed.data[0] = np.nextafter(changed.data[0], np.inf)
            changed_reverse = operator._ConstraintBlock(
                changed, reverse.Ab, reverse.rhs, reverse.tag
            )
            return original_sink(
                sink,
                forward,
                changed_reverse,
                layer_id=layer_id,
            )

        def capture_append(arena: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_core(arena, *args, **kwargs)
            captured["append"] = result
            return result

        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "append_add_materialize_range",
            new=mismatch,
        ), mock.patch.object(
            core.ConstraintArena,
            "append_guarded_band",
            new=capture_append,
        ), mock.patch.object(
            operator,
            "hz_fresh_col_ids",
            new=_DeterministicAllocator(),
        ), mock.patch.object(
            operator, "_assemble_owned_operator_sparse_hz"
        ) as constructor:
            with self.assertRaisesRegex(
                operator.OperatorHZBuildError,
                "did not commit as an all-RANGE block",
            ):
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        constructor.assert_not_called()
        self.assertEqual(captured["append"].fallback_pairs, 1)
        self.assertEqual(captured["sink"].phase, "discarded")
        self.assertTrue(captured["sink"].arena.discarded)
        self.assertTrue(captured["sink"].owner.discarded)
        self.assertIsNone(captured["sink"].program)

    def test_seal_public_return_abort_recovers_same_program(self) -> None:
        toy = _add_toy(2, with_exact_relu=False)
        abort = _BuildAbort("seal public return")
        captured: Dict[str, Any] = {}
        original_initialize = operator._OperatorConstraintProgramSink.initialize
        original_complete = core._complete_seal_publication
        first = [True]

        def capture_initialize(sink: Any) -> None:
            original_initialize(sink)
            captured["sink"] = sink

        def interrupted(*args: Any, **kwargs: Any) -> Any:
            program = original_complete(*args, **kwargs)
            captured["program"] = program
            if first[0]:
                first[0] = False
                raise abort
            return program

        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "initialize",
            new=capture_initialize,
        ), mock.patch.object(
            core, "_complete_seal_publication", new=interrupted
        ), mock.patch.object(
            operator,
            "hz_fresh_col_ids",
            new=_DeterministicAllocator(),
        ), mock.patch.object(
            operator, "_assemble_owned_operator_sparse_hz"
        ) as constructor:
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        constructor.assert_not_called()
        sink = captured["sink"]
        self.assertEqual(sink.phase, "sealed")
        self.assertIs(sink.program, captured["program"])
        self.assertTrue(sink.program.representation_authority)
        self.assertEqual(sink.program.virtual_facet_rows, 4)
        state = core._arena_state(sink.arena)
        self.assertTrue(state.sealed)
        self.assertTrue(state.owner_state.sealed)
        self.assertFalse(state.discarded)

    def test_seal_prepublication_abort_terminally_discards(self) -> None:
        toy = _add_toy(2, with_exact_relu=False)
        abort = _BuildAbort("seal staging")
        captured: Dict[str, Any] = {}
        original_initialize = operator._OperatorConstraintProgramSink.initialize

        def capture_initialize(sink: Any) -> None:
            original_initialize(sink)
            captured["sink"] = sink

        def fail_stage(*_args: Any, **_kwargs: Any) -> Any:
            raise abort

        programs_before = set(core._PROGRAM_REGISTRY)
        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "initialize",
            new=capture_initialize,
        ), mock.patch.object(
            core, "_stage_program", new=fail_stage
        ), mock.patch.object(
            operator,
            "hz_fresh_col_ids",
            new=_DeterministicAllocator(),
        ), mock.patch.object(
            operator, "_assemble_owned_operator_sparse_hz"
        ) as constructor:
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        constructor.assert_not_called()
        sink = captured["sink"]
        self.assertEqual(sink.phase, "discarded")
        self.assertTrue(sink.arena.discarded)
        self.assertTrue(sink.owner.discarded)
        self.assertIsNone(sink.program)
        self.assertEqual(set(core._PROGRAM_REGISTRY), programs_before)

    def test_postseal_replay_failure_keeps_program_then_gc_cleans_all(self) -> None:
        toy = _add_toy(2, with_exact_relu=True)
        abort = _BuildAbort("legacy replay")
        captured: Dict[str, Any] = {}
        original_initialize = operator._OperatorConstraintProgramSink.initialize

        def capture_initialize(sink: Any) -> None:
            original_initialize(sink)
            captured["sink"] = sink

        def fail_replay(program: Any, *, max_rows: int) -> Any:
            captured["program"] = program
            raise abort

        gc.collect()
        baselines = {
            "adapter": set(core._ADAPTER_REGISTRY),
            "binding": set(core._ALLOCATOR_BINDINGS),
            "owner": set(core._OWNER_REGISTRY),
            "arena": set(core._ARENA_REGISTRY),
            "sealed": set(core._SEALED_ARENA_REGISTRY),
            "program": set(core._PROGRAM_REGISTRY),
        }
        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "initialize",
            new=capture_initialize,
        ), mock.patch.object(
            core.ConstraintProgram,
            "iter_legacy_facet_batches",
            new=fail_replay,
        ), mock.patch.object(
            operator,
            "hz_fresh_col_ids",
            new=_DeterministicAllocator(),
        ), mock.patch.object(
            operator, "_assemble_owned_operator_sparse_hz"
        ) as constructor:
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        constructor.assert_not_called()
        sink = captured["sink"]
        program = captured["program"]
        self.assertEqual(sink.phase, "sealed")
        self.assertIs(sink.program, program)
        self.assertTrue(program.representation_authority)
        self.assertTrue(program.replay_authority)
        self.assertFalse(program.proof_authority)
        state = core._arena_state(sink.arena)
        self.assertTrue(state.sealed)
        self.assertTrue(state.owner_state.sealed)
        self.assertFalse(state.discarded)
        replay = _replay_legacy(program)
        self.assertEqual(replay[2].size, program.virtual_facet_rows)

        ids = {
            "adapter": id(sink.adapter),
            "binding": id(sink.bridge),
            "owner": id(sink.owner),
            "arena": id(sink.arena),
            "sealed": id(sink.arena),
            "program": id(program),
        }
        references = tuple(
            weakref.ref(value)
            for value in (
                sink.adapter,
                sink.owner,
                sink.arena,
                program,
            )
        )
        captured.clear()
        del program
        del sink
        del state
        del replay
        del raised
        for _attempt in range(4):
            gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        registries = {
            "adapter": core._ADAPTER_REGISTRY,
            "binding": core._ALLOCATOR_BINDINGS,
            "owner": core._OWNER_REGISTRY,
            "arena": core._ARENA_REGISTRY,
            "sealed": core._SEALED_ARENA_REGISTRY,
            "program": core._PROGRAM_REGISTRY,
        }
        for name, registry in registries.items():
            self.assertNotIn(ids[name], registry, msg=name)
            self.assertEqual(set(registry), baselines[name], msg=name)

    def test_postseal_final_assembly_failure_never_falls_back(self) -> None:
        toy = _add_toy(2, with_exact_relu=True)
        abort = _BuildAbort("SparseHZ assembly")
        captured: Dict[str, Any] = {}
        original = operator._OperatorConstraintProgramSink.seal_and_replay

        def capture_program(sink: Any, **kwargs: Any) -> Any:
            result = original(sink, **kwargs)
            captured.update(sink=sink, program=result[0])
            return result

        with mock.patch.object(
            operator, "_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK", True
        ), mock.patch.object(
            operator._OperatorConstraintProgramSink,
            "seal_and_replay",
            new=capture_program,
        ), mock.patch.object(
            operator,
            "hz_fresh_col_ids",
            new=_DeterministicAllocator(),
        ), mock.patch.object(
            operator,
            "_assemble_owned_operator_sparse_hz",
            side_effect=abort,
        ) as constructor:
            with self.assertRaises(_BuildAbort) as raised:
                operator.build_operator_hz(
                    toy.net, toy.facts, toy.facts, exact_budget=-1
                )
        self.assertIs(raised.exception, abort)
        constructor.assert_called_once()
        self.assertEqual(captured["sink"].phase, "sealed")
        self.assertIs(captured["sink"].program, captured["program"])
        self.assertTrue(captured["program"].representation_authority)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(
            kwargs["ub"].size,
            captured["program"].virtual_facet_rows,
        )
        self.assertEqual(
            kwargs["Auc"].shape[0],
            captured["program"].virtual_facet_rows,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
