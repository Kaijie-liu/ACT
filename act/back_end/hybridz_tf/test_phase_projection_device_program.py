#!/usr/bin/env python3
"""CPU/mock gates for the request-local phase-projection device program.

These tests use analytic tensors only.  They do not execute ONNX, sample an
input domain, run PGD, split, enumerate, propagate backward bounds, tighten a
dual, or call a solver.
"""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from act.back_end.hybridz_tf import (
    forward_exact_relu_live_row_stream_candidate as _live,
)
from act.back_end.hybridz_tf import phase_projection_device_program as _program


def _proxy(values):
    return MappingProxyType(dict(values))


def _frame(*, active, stream_rows):
    empty_i = np.empty(0, dtype=np.int64)
    return SimpleNamespace(
        active=np.asarray(active, dtype=np.int64),
        exact=empty_i,
        stream_rows=np.asarray(stream_rows, dtype=np.int64),
        stream_continuous_columns=empty_i,
        stream_half_widths=np.empty(0, dtype=np.float64),
    )


def _candidate_toy() -> _program.CandidateProgram:
    steps = (
        _program.LayerStep(0, "INPUT", (), 2),
        _program.LayerStep(1, "DENSE", (0,), 2),
        _program.LayerStep(2, "RELU", (1,), 2),
        _program.LayerStep(3, "ASSERT", (2,), 2),
    )
    matrix = _live._DeviceCSR(
        indptr=torch.tensor([0, 1, 2], dtype=torch.int64),
        indices=torch.tensor([0, 1], dtype=torch.int64),
        data=torch.tensor([2.0, 3.0], dtype=torch.float64),
        rows=2,
        columns=2,
    )
    return _program.CandidateProgram(
        steps=steps,
        affines=_proxy({}),
        pointwise_device=_proxy({}),
        matrices=_proxy({1: matrix}),
        live_rows=_proxy(
            {
                1: np.asarray([0, 1], dtype=np.int64),
                2: np.asarray([0, 1], dtype=np.int64),
            }
        ),
        device_rows=_proxy(
            {1: torch.tensor([0, 1], dtype=torch.int64)}
        ),
        input_batches=(
            _program.InputBatch(
                rows=torch.tensor([0, 1], dtype=torch.int64),
                columns=torch.tensor([0, 1], dtype=torch.int64),
                radii=torch.tensor([0.5, 0.25], dtype=torch.float64),
                start=0,
                stop=2,
            ),
        ),
        successor_uses=_proxy({0: 1, 1: 1, 2: 1, 3: 0}),
        n_cont=2,
        assert_layer_id=3,
        assert_width=2,
        device=torch.device("cpu"),
        deadline=None,
        build_seconds=0.0,
        cuda_bytes=0,
    )


def _candidate_toy_width(width: int) -> _program.CandidateProgram:
    rows = np.arange(width, dtype=np.int64)
    matrix = _live._DeviceCSR(
        indptr=torch.arange(width + 1, dtype=torch.int64),
        indices=torch.arange(width, dtype=torch.int64),
        data=torch.ones(width, dtype=torch.float64),
        rows=width,
        columns=width,
    )
    return _program.CandidateProgram(
        steps=(
            _program.LayerStep(0, "INPUT", (), width),
            _program.LayerStep(1, "DENSE", (0,), width),
            _program.LayerStep(2, "RELU", (1,), width),
            _program.LayerStep(3, "ASSERT", (2,), width),
        ),
        affines=_proxy({}),
        pointwise_device=_proxy({}),
        matrices=_proxy({1: matrix}),
        live_rows=_proxy({1: rows, 2: rows}),
        device_rows=_proxy({1: torch.arange(width, dtype=torch.int64)}),
        input_batches=(),
        successor_uses=_proxy({0: 1, 1: 1, 2: 1, 3: 0}),
        n_cont=width,
        assert_layer_id=3,
        assert_width=width,
        device=torch.device("cpu"),
        deadline=None,
        build_seconds=0.0,
        cuda_bytes=0,
    )


def _dense_affine_pair():
    weight = np.asarray([[2.0, -1.0], [1.0, 3.0]], dtype=np.float64)
    bias = np.asarray([0.125, -0.25], dtype=np.float64)
    snapshot = SimpleNamespace(
        kind="DENSE",
        weight=weight,
        bias=bias,
        input_size=2,
        output_size=2,
        topology=None,
    )
    candidate, terminal = _program._seal_affine_pair(
        snapshot,
        _program.LayerStep(1, "DENSE", (0,), 2),
        device=torch.device("cpu"),
        deadline=None,
    )
    return snapshot, candidate, terminal


def _dense_terminal_program(affine: _program.TerminalAffine):
    return _program.TerminalProgram(
        steps=(
            _program.LayerStep(0, "INPUT", (), 2),
            _program.LayerStep(1, "DENSE", (0,), 2),
            _program.LayerStep(2, "RELU", (1,), 2),
            _program.LayerStep(3, "ASSERT", (2,), 2),
        ),
        affines=_proxy({1: affine}),
        pointwise=_proxy({}),
        successor_uses=_proxy({0: 1, 1: 1, 2: 1, 3: 0}),
        output_layer_id=2,
        device=torch.device("cpu"),
        deadline=None,
    )


def _ordered_cpu(matrix: _live._DeviceCSR, dense: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(
        (matrix.rows, int(dense.shape[1])), dtype=torch.float64
    )
    for row in range(matrix.rows):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        for cursor in range(start, stop):
            result[row] += matrix.data[cursor] * dense[matrix.indices[cursor]]
    return result


class PhaseProjectionDeviceProgramStructureTests(unittest.TestCase):
    def test_terminal_api_and_state_are_candidate_blind(self) -> None:
        self.assertEqual(
            list(inspect.signature(_program.terminal_interval_forward).parameters),
            ["decoded", "program"],
        )
        terminal_fields = {item.name for item in fields(_program.TerminalProgram)}
        forbidden = {
            "candidate",
            "matrices",
            "live_rows",
            "device_rows",
            "input_batches",
            "phase",
            "margin",
            "factors",
            "lp",
        }
        self.assertTrue(terminal_fields.isdisjoint(forbidden))
        self.assertNotIn("snapshot", terminal_fields)
        candidate_fields = {item.name for item in fields(_program.CandidateProgram)}
        self.assertNotIn("terminal", candidate_fields)
        self.assertNotIn(
            "snapshot", {item.name for item in fields(_program.CandidateAffine)}
        )

    def test_graph_schedule_is_topological_and_counts_successors(self) -> None:
        layers = tuple(
            SimpleNamespace(id=index, kind=kind, out_vars=[0, 1])
            for index, kind in enumerate(
                ("INPUT", "INPUT_SPEC", "DENSE", "RELU", "ASSERT")
            )
        )
        net = SimpleNamespace(preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3]})
        steps = _program.seal_layer_steps(net, layers)
        self.assertEqual(tuple(step.layer_id for step in steps), tuple(range(5)))
        self.assertEqual(steps[3].predecessors, (2,))
        self.assertEqual(dict(_program._successor_uses(steps)), {0: 1, 1: 1, 2: 1, 3: 1, 4: 0})
        with self.assertRaises(_program.PhaseProjectionDeviceProgramError):
            _program.seal_layer_steps(net, layers[::-1])

    def test_stream_source_keeps_full_results_on_host(self) -> None:
        source = inspect.getsource(_program._stream_fixed_cell_generators_impl)
        self.assertIn("preactivation_dense =", source)
        self.assertIn("np.empty", source)
        self.assertNotIn("device_pre", source)
        delta_source = inspect.getsource(_program._stream_phase_deltas_impl)
        self.assertIn("delta_pre =", delta_source)
        self.assertNotIn("device_pre", delta_source)

    def test_failure_cleanup_and_last_use_are_structural(self) -> None:
        fixed = inspect.getsource(_program._stream_fixed_cell_generators_impl)
        delta = inspect.getsource(_program._stream_phase_deltas_impl)
        terminal = inspect.getsource(_program._terminal_interval_forward_impl)
        self.assertIn("finally:", fixed)
        self.assertIn("values.clear()", fixed)
        self.assertIn("finally:", delta)
        self.assertIn("values.clear()", delta)
        self.assertIn("del shadows[predecessor]", terminal)
        self.assertIn("shadows.clear()", terminal)
        self.assertIn('"device terminal sync"', terminal)
        self.assertIn('"device terminal return"', terminal)


class PhaseProjectionDeviceProgramArithmeticTests(unittest.TestCase):
    def test_affine_snapshot_and_candidate_cannot_mutate_terminal(self) -> None:
        snapshot, candidate, terminal = _dense_affine_pair()
        candidate_ptr = candidate.weight.untyped_storage().data_ptr()
        terminal_ptr = terminal.weight.untyped_storage().data_ptr()
        absolute_ptr = terminal.absolute_weight.untyped_storage().data_ptr()
        self.assertEqual(len({candidate_ptr, terminal_ptr, absolute_ptr}), 3)
        terminal_weight = terminal.weight.clone()
        terminal_absolute = terminal.absolute_weight.clone()
        terminal_version = terminal.weight._version
        absolute_version = terminal.absolute_weight._version

        snapshot.weight.fill(99.0)
        snapshot.bias.fill(77.0)
        snapshot.kind = "CONV2D"
        snapshot.topology = SimpleNamespace(input_shape=(9, 9, 9, 9))
        with torch.no_grad():
            candidate.weight.add_(5.0)
        self.assertEqual(terminal.weight._version, terminal_version)
        self.assertEqual(terminal.absolute_weight._version, absolute_version)
        self.assertTrue(torch.equal(terminal.weight, terminal_weight))
        self.assertTrue(torch.equal(terminal.absolute_weight, terminal_absolute))
        self.assertEqual(candidate.kind, "DENSE")
        self.assertEqual(terminal.kind, "DENSE")
        self.assertIsNone(candidate.topology)
        self.assertIsNone(terminal.topology)
        np.testing.assert_array_equal(
            terminal.bias.view(np.uint64),
            np.asarray([0.125, -0.25], dtype=np.float64).view(np.uint64),
        )
        with self.assertRaises(ValueError):
            terminal.bias.setflags(write=True)
        with self.assertRaises(ValueError):
            candidate.bias.setflags(write=True)

    def test_stored_input_breaks_aliases_and_is_the_terminal_input(self) -> None:
        _snapshot, _candidate, affine = _dense_affine_pair()
        terminal = _dense_terminal_program(affine)
        source = np.asarray([0.25, -0.5], dtype=np.float64)
        sealed = _program.seal_terminal_input(source)
        original_bits = sealed.values.view(np.uint64).copy()
        source[:] = 123.0
        np.testing.assert_array_equal(sealed.values.view(np.uint64), original_bits)
        with self.assertRaises(ValueError):
            sealed.values.setflags(write=True)

        observed = []
        original = _program._terminal_affine_shadow

        def inspect_source(shadow, affine_value, *, layer_id):
            observed.append(shadow.center is sealed.values)
            return original(shadow, affine_value, layer_id=layer_id)

        with mock.patch.object(
            _program, "_terminal_affine_shadow", side_effect=inspect_source
        ):
            _program.terminal_interval_forward(sealed, terminal)
        self.assertEqual(observed, [True])

    def test_dense_center_is_bitwise_equal_to_frozen_operation_order(self) -> None:
        weight_array = np.asarray(
            [[0.1, -0.7], [1.25, 0.3]], dtype=np.float64
        )
        bias = np.asarray([0.2, -0.4], dtype=np.float64)
        weight = torch.tensor(weight_array, dtype=torch.float64)
        source = _live._Shadow(
            np.asarray([0.25, -0.5], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.asarray([0.25, 0.5], dtype=np.float64),
        )
        frozen = (
            torch.matmul(
                torch.tensor(weight_array, dtype=torch.float64),
                torch.as_tensor(source.center, dtype=torch.float64),
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            + bias
        )
        actual = _program.candidate_affine_center(
            source,
            _program.CandidateAffine(
                kind="DENSE",
                weight=weight,
                bias=_program._readonly_array(
                    bias, dtype=np.float64, name="test_bias"
                ),
                output_size=2,
                topology=None,
            ),
            layer_id=7,
        )
        np.testing.assert_array_equal(
            actual.center.view(np.uint64), frozen.view(np.uint64)
        )

    def test_fixed_stream_reuses_batches_and_preserves_values(self) -> None:
        candidate = _candidate_toy()
        frames = {2: _frame(active=[0, 1], stream_rows=[0, 1])}
        schedule = _program.seal_fixed_phase_schedule(candidate, frames)
        with mock.patch.object(
            _program._live, "_ordered_csr_dense", side_effect=_ordered_cpu
        ):
            pre, output, _elapsed = _program.stream_fixed_cell_generators(
                candidate, schedule
            )
        expected = np.asarray([[1.0, 0.0], [0.0, 0.75]], dtype=np.float64)
        np.testing.assert_array_equal(pre[2], expected)
        np.testing.assert_array_equal(output, expected)

    def test_k_column_delta_is_batched_and_does_not_form_target_rows(self) -> None:
        candidate = _candidate_toy()
        original = {2: _frame(active=[0, 1], stream_rows=[0, 1])}
        target = {2: _frame(active=[0], stream_rows=[0, 1])}
        schedule = _program.seal_delta_schedule(
            candidate,
            original,
            target,
            ((2, 1, True, False),),
        )
        with mock.patch.object(
            _program._live, "_ordered_csr_dense", side_effect=_ordered_cpu
        ):
            delta_pre, delta_output, _elapsed = _program.stream_phase_deltas(
                candidate, schedule
            )
        np.testing.assert_array_equal(
            delta_pre[2], np.zeros((2, 1), dtype=np.float64)
        )
        np.testing.assert_array_equal(
            delta_output, np.asarray([[0.0], [1.0]], dtype=np.float64)
        )
        source = inspect.getsource(_program._stream_phase_deltas_impl)
        self.assertNotIn("target_pre", source)
        self.assertNotIn("target_output", source)
        self.assertIn("range(0, schedule.width, _live._FACTOR_BATCH)", source)

    def test_delta_zero_64_and_65_columns_have_one_fixed_batch_rule(self) -> None:
        zero_candidate = _candidate_toy()
        unchanged = {2: _frame(active=[0, 1], stream_rows=[0, 1])}
        zero = _program.seal_delta_schedule(
            zero_candidate, unchanged, unchanged, ()
        )
        self.assertEqual(zero.width, 0)
        pre, output, _elapsed = _program.stream_phase_deltas(
            zero_candidate, zero
        )
        self.assertEqual(pre[2].shape, (2, 0))
        self.assertEqual(output.shape, (2, 0))

        for width, expected_calls in ((64, 1), (65, 2)):
            candidate = _candidate_toy_width(width)
            rows = np.arange(width, dtype=np.int64)
            original = {2: _frame(active=rows, stream_rows=rows)}
            target = {2: _frame(active=[], stream_rows=rows)}
            changes = tuple((2, row, True, False) for row in range(width))
            schedule = _program.seal_delta_schedule(
                candidate, original, target, changes
            )
            with mock.patch.object(
                _program._live,
                "_ordered_csr_dense",
                side_effect=_ordered_cpu,
            ) as ordered:
                _pre, actual, _elapsed = _program.stream_phase_deltas(
                    candidate, schedule
                )
            self.assertEqual(ordered.call_count, expected_calls)
            np.testing.assert_array_equal(
                actual, np.eye(width, dtype=np.float64)
            )

    def test_delta_changes_must_be_exact_unique_topological_difference(self) -> None:
        candidate = _candidate_toy()
        original = {2: _frame(active=[0, 1], stream_rows=[0, 1])}
        target = {2: _frame(active=[], stream_rows=[0, 1])}
        valid = ((2, 0, True, False), (2, 1, True, False))
        self.assertEqual(
            _program.seal_delta_schedule(
                candidate, original, target, valid
            ).width,
            2,
        )
        for malformed in (valid[::-1], valid[:1], valid + valid[:1]):
            with self.assertRaises(
                _program.PhaseProjectionDeviceProgramError
            ):
                _program.seal_delta_schedule(
                    candidate, original, target, malformed
                )

    def test_host_outputs_are_rejected_before_oversize_allocation(self) -> None:
        candidate = _candidate_toy()
        fixed = _program.seal_fixed_phase_schedule(
            candidate, {2: _frame(active=[0, 1], stream_rows=[0, 1])}
        )
        too_wide = replace(
            candidate,
            n_cont=_program._MAX_HOST_OUTPUT_BYTES // 8 + 1,
        )
        with mock.patch.object(
            _program.np, "empty", side_effect=AssertionError("allocated")
        ):
            with self.assertRaises(
                _program.PhaseProjectionDeviceProgramError
            ):
                _program.stream_fixed_cell_generators(too_wide, fixed)

        oversize_delta = _program.DeltaSchedule(
            exact_rows=_proxy({2: torch.empty(0, dtype=torch.int64)}),
            active_rows=_proxy({2: torch.empty(0, dtype=torch.int64)}),
            changed_rows=_proxy({2: ()}),
            width=_program._MAX_HOST_OUTPUT_BYTES // 8 + 1,
        )
        with mock.patch.object(
            _program.np, "empty", side_effect=AssertionError("allocated")
        ):
            with self.assertRaises(
                _program.PhaseProjectionDeviceProgramError
            ):
                _program.stream_phase_deltas(candidate, oversize_delta)

    def test_deadlines_are_checked_after_batches_layers_sync_and_return(self) -> None:
        candidate = _candidate_toy()
        fixed = _program.seal_fixed_phase_schedule(
            candidate, {2: _frame(active=[0, 1], stream_rows=[0, 1])}
        )
        fixed_stages = []

        def audit_fixed(_deadline, stage):
            fixed_stages.append(stage)

        with mock.patch.object(_program, "_deadline", side_effect=audit_fixed), mock.patch.object(
            _program._live, "_ordered_csr_dense", side_effect=_ordered_cpu
        ):
            _program.stream_fixed_cell_generators(candidate, fixed)
        self.assertTrue(any(stage.endswith("layer 3 complete") for stage in fixed_stages))
        self.assertIn("device first stream sync", fixed_stages)
        self.assertIn("device first stream return", fixed_stages)

        _snapshot, _candidate, affine = _dense_affine_pair()
        terminal_stages = []
        with mock.patch.object(
            _program,
            "_deadline",
            side_effect=lambda _deadline, stage: terminal_stages.append(stage),
        ):
            _program.terminal_interval_forward(
                _program.seal_terminal_input(
                    np.asarray([0.25, -0.5], dtype=np.float64)
                ),
                _dense_terminal_program(affine),
            )
        self.assertTrue(
            any(stage.endswith("layer 3 complete") for stage in terminal_stages)
        )
        self.assertIn("device terminal sync", terminal_stages)
        self.assertIn("device terminal return", terminal_stages)

    def test_abs_weight_is_an_exact_structural_support_kernel(self) -> None:
        topology = _program.AffineTopology(
            input_shape=(1, 1, 2, 3),
            output_shape=(1, 1, 2, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
        weight = torch.tensor([[[[2.0, -0.5]]]], dtype=torch.float64)
        affine = _program.TerminalAffine(
            kind="CONV2D",
            weight=weight,
            absolute_weight=torch.abs(weight),
            bias=np.zeros(4, dtype=np.float64),
            output_size=4,
            topology=topology,
            fanin=np.full(4, 2.0, dtype=np.float64),
            gamma=np.ones(4, dtype=np.float64),
            absolute_bias=np.zeros(4, dtype=np.float64),
        )
        mass = np.asarray([False, True, False, False, False, False])
        error = np.asarray([False, False, False, False, False, True])
        mass_support, error_support = _program._terminal_support(
            affine, mass, error
        )
        np.testing.assert_array_equal(
            mass_support, np.asarray([True, True, False, False])
        )
        np.testing.assert_array_equal(
            error_support, np.asarray([False, False, False, True])
        )

    def test_terminal_accepts_only_decoded_and_encloses_dense_relu(self) -> None:
        steps = (
            _program.LayerStep(0, "INPUT", (), 2),
            _program.LayerStep(1, "DENSE", (0,), 2),
            _program.LayerStep(2, "RELU", (1,), 2),
            _program.LayerStep(3, "ASSERT", (2,), 2),
        )
        stored = np.asarray([[2.0, -1.0], [1.0, 3.0]], dtype=np.float64)
        bias = np.zeros(2, dtype=np.float64)
        weight = torch.tensor(stored, dtype=torch.float64)
        fanin = np.asarray([2.0, 2.0], dtype=np.float64)
        affine = _program.TerminalAffine(
            kind="DENSE",
            weight=weight,
            absolute_weight=torch.abs(weight),
            bias=_program._readonly_array(
                bias, dtype=np.float64, name="terminal_test_bias"
            ),
            output_size=2,
            topology=None,
            fanin=fanin,
            gamma=_program._oh._gamma_ops(
                2.0 * fanin + 2.0, name="device_program_test.gamma"
            ),
            absolute_bias=np.abs(bias),
        )
        terminal = _program.TerminalProgram(
            steps=steps,
            affines=_proxy({1: affine}),
            pointwise=_proxy({}),
            successor_uses=_proxy({0: 1, 1: 1, 2: 1, 3: 0}),
            output_layer_id=2,
            device=torch.device("cpu"),
            deadline=None,
        )
        decoded = np.asarray([0.25, -0.5], dtype=np.float64)
        sealed = _program.seal_terminal_input(decoded)
        lower, upper = _program.terminal_interval_forward(sealed, terminal)
        expected = np.asarray([1.0, 0.0], dtype=np.float64)
        self.assertTrue(np.all(lower <= expected))
        self.assertTrue(np.all(upper >= expected))
        with self.assertRaises(_program.PhaseProjectionDeviceProgramError):
            _program.seal_terminal_input(decoded.astype(np.float32))
        with self.assertRaises(_program.PhaseProjectionDeviceProgramError):
            _program.terminal_interval_forward(decoded, terminal)

    def test_baseexception_identity_survives_build_stream_delta_and_terminal(self) -> None:
        class Bomb(BaseException):
            pass

        candidate = _candidate_toy()
        fixed = _program.seal_fixed_phase_schedule(
            candidate, {2: _frame(active=[0, 1], stream_rows=[0, 1])}
        )
        fixed_bomb = Bomb("fixed")
        with mock.patch.object(
            _program._live, "_ordered_csr_dense", side_effect=fixed_bomb
        ):
            with self.assertRaises(Bomb) as caught:
                _program.stream_fixed_cell_generators(candidate, fixed)
        self.assertIs(caught.exception, fixed_bomb)

        delta = _program.seal_delta_schedule(
            candidate,
            {2: _frame(active=[0, 1], stream_rows=[0, 1])},
            {2: _frame(active=[0], stream_rows=[0, 1])},
            ((2, 1, True, False),),
        )
        delta_bomb = Bomb("delta")
        with mock.patch.object(
            _program._live, "_ordered_csr_dense", side_effect=delta_bomb
        ):
            with self.assertRaises(Bomb) as caught:
                _program.stream_phase_deltas(candidate, delta)
        self.assertIs(caught.exception, delta_bomb)

        snapshot, _candidate, affine = _dense_affine_pair()
        terminal_bomb = Bomb("terminal")
        with mock.patch.object(
            _program, "_terminal_affine_shadow", side_effect=terminal_bomb
        ):
            with self.assertRaises(Bomb) as caught:
                _program.terminal_interval_forward(
                    _program.seal_terminal_input(
                        np.asarray([0.25, -0.5], dtype=np.float64)
                    ),
                    _dense_terminal_program(affine),
                )
        self.assertIs(caught.exception, terminal_bomb)

        layers = (
            SimpleNamespace(id=0, kind="INPUT", out_vars=[0, 1]),
            SimpleNamespace(id=1, kind="DENSE", out_vars=[0, 1]),
            SimpleNamespace(id=2, kind="ASSERT", out_vars=[0, 1]),
        )
        net = SimpleNamespace(preds={0: [], 1: [0], 2: [1]})
        build_bomb = Bomb("build")
        cpu = torch.device("cpu")
        with mock.patch.object(
            _program.torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            _program.torch, "device", return_value=cpu
        ), mock.patch.object(
            _program, "_sync"
        ), mock.patch.object(
            _program.torch.cuda, "memory_allocated", return_value=0
        ), mock.patch.object(
            _program, "_seal_affine_pair", side_effect=build_bomb
        ):
            with self.assertRaises(Bomb) as caught:
                _program.build_request_local_programs(
                    net,
                    layers,
                    {1: snapshot},
                    {},
                    {},
                    {},
                    input_rows=np.asarray([0], dtype=np.int64),
                    input_radius=np.asarray([1.0, 1.0], dtype=np.float64),
                    assert_layer_id=2,
                    output_layer_id=1,
                    deadline=None,
                )
        self.assertIs(caught.exception, build_bomb)


if __name__ == "__main__":
    unittest.main()
