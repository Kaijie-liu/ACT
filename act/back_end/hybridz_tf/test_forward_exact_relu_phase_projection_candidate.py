#!/usr/bin/env python3
"""Focused gates for the disconnected one-update phase projection.

Only analytic toy graphs and stored binary64/Fraction checks are used.  The
tests do not execute ONNX, sample inputs, run PGD, BaB, backward propagation,
or dual tightening.
"""

from __future__ import annotations

from fractions import Fraction
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import forward_exact_relu_phase_projection_candidate as _projection
from act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate import (
    ExactReLUPhaseProjectionUnknown,
    build_forward_exact_relu_phase_projection_candidate,
)
from act.back_end.hybridz_tf.test_operator_add_fusion import (
    _assemble_width_toy,
    _dense,
    _dense_matrix,
    _input_layers,
    _layer,
    _wide_layer,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once
from act.util.stats import VerifyStatus


def _fact(lower: list[float], upper: list[float]) -> Fact:
    return Fact(
        Bounds(
            torch.tensor([lower], dtype=torch.float64, device="cuda"),
            torch.tensor([upper], dtype=torch.float64, device="cuda"),
        ),
        ConSet(),
    )


def _phase_projection_toy():
    input_layer, spec = _input_layers(-1, 1)
    layers = [
        input_layer,
        spec,
        _dense(2, 1, Fraction(-1, 4)),
        _layer(3, "RELU"),
        _dense_matrix(4, [[1], [-1]], [0, 0]),
        _wide_layer(5, "ASSERT", 2),
    ]
    layers[-1].params.update(
        {
            "C": torch.tensor([[1.0, -1.0]], dtype=torch.float64, device="cuda"),
            "thresholds": torch.tensor([0.0], dtype=torch.float64, device="cuda"),
            "M": torch.tensor([0], dtype=torch.int64, device="cuda"),
        }
    )
    toy = _assemble_width_toy(
        layers,
        {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        input_lb=-1,
        input_ub=1,
    )
    facts = dict(toy.facts)
    facts[0] = _fact([-1.0], [1.0])
    facts[1] = _fact([-1.0], [1.0])
    facts[3] = _fact([-1.25], [0.75])
    return toy, facts


class ExactReLUPhaseProjectionPureTests(unittest.TestCase):
    def test_optimal_negative_selector_is_one_frozen_strict_rule(self) -> None:
        selected, tight_count, negative_count = (
            _projection._select_optimal_negative_rows(
                row_value=np.asarray(
                    [1.0, 1.0 - 3.0e-9, 2.0, 3.0 + 5.0e-10]
                ),
                row_dual=np.asarray([-1.0, -2.0, 0.0, -3.0]),
                row_ids=np.asarray([4, 7, 11, 13], dtype=np.int64),
                loaded_upper=np.asarray([1.0, 1.0, 2.0, 3.0]),
                candidate_margin=-0.25,
            )
        )
        np.testing.assert_array_equal(
            selected, np.asarray([4, 13], dtype=np.int64)
        )
        self.assertEqual(tight_count, 3)
        self.assertEqual(negative_count, 3)
        with self.assertRaisesRegex(
            ExactReLUPhaseProjectionUnknown, "produced no phase rows"
        ):
            _projection._select_optimal_negative_rows(
                row_value=np.asarray([1.0]),
                row_dual=np.asarray([0.0]),
                row_ids=np.asarray([3], dtype=np.int64),
                loaded_upper=np.asarray([1.0]),
                candidate_margin=-1.0,
            )

    def test_infeasible_selector_uses_exact_nonzero_ray_support(self) -> None:
        tiny = float(np.nextafter(0.0, -1.0))
        selected = _projection._select_infeasible_ray_rows(
            row_ray=np.asarray([0.0, tiny, -2.0], dtype=np.float64),
            row_ids=np.asarray([2, 5, 9], dtype=np.int64),
            support_row_ids=(5, 9),
        )
        np.testing.assert_array_equal(selected, np.asarray([5, 9], dtype=np.int64))
        with self.assertRaisesRegex(
            ExactReLUPhaseProjectionUnknown, "support mapping drifted"
        ):
            _projection._select_infeasible_ray_rows(
                row_ray=np.asarray([0.0, -1.0]),
                row_ids=np.asarray([2, 5], dtype=np.int64),
                support_row_ids=(2,),
            )

    def test_terminal_box_and_forward_share_one_sealed_input(self) -> None:
        calls = {}

        def seal(values):
            raw = np.asarray(values, dtype=np.float64).reshape(-1)
            sealed_values = np.frombuffer(raw.tobytes(), dtype=np.float64)
            result = SimpleNamespace(values=sealed_values)
            calls["sealed"] = result
            return result

        def terminal(sealed, program):
            self.assertIs(sealed, calls["sealed"])
            self.assertIs(program, calls["program"])
            calls["terminal"] = sealed
            return np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])

        calls["program"] = object()
        device = SimpleNamespace(
            seal_terminal_input=seal,
            terminal_interval_forward=terminal,
        )
        decoded, margin, _seconds = _projection._terminal_candidate(
            device_module=device,
            terminal_program=calls["program"],
            factors=np.asarray([0.0]),
            input_rows=np.asarray([0], dtype=np.int64),
            input_center=np.asarray([0.5]),
            input_radius=np.asarray([0.5]),
            input_shape=(1, 1),
            raw_lower=np.asarray([0.0]),
            raw_upper=np.asarray([1.0]),
            property_row=np.asarray([1.0, -1.0]),
            threshold=0.0,
            output_width=2,
            deadline=None,
        )
        self.assertIs(calls["terminal"], calls["sealed"])
        self.assertTrue(np.shares_memory(decoded, calls["sealed"].values))
        self.assertEqual(margin, 1.0)

    def test_raw_box_rejects_the_sealed_values_before_terminal(self) -> None:
        terminal = mock.Mock()
        sealed = SimpleNamespace(
            values=np.frombuffer(np.asarray([2.0]).tobytes(), dtype=np.float64)
        )
        device = SimpleNamespace(
            seal_terminal_input=mock.Mock(return_value=sealed),
            terminal_interval_forward=terminal,
        )
        with self.assertRaisesRegex(
            ExactReLUPhaseProjectionUnknown, "outside raw BOX"
        ):
            _projection._terminal_candidate(
                device_module=device,
                terminal_program=object(),
                factors=np.asarray([0.0]),
                input_rows=np.asarray([0], dtype=np.int64),
                input_center=np.asarray([0.5]),
                input_radius=np.asarray([0.5]),
                input_shape=(1, 1),
                raw_lower=np.asarray([0.0]),
                raw_upper=np.asarray([1.0]),
                property_row=np.asarray([1.0, -1.0]),
                threshold=0.0,
                output_width=2,
                deadline=None,
            )
        terminal.assert_not_called()

    def test_production_source_has_one_owner_and_no_legacy_solver_or_terminal(self) -> None:
        source = inspect.getsource(_projection)
        self.assertNotIn("linprog", source)
        self.assertNotIn("_singleton_interval_forward", source)
        self.assertNotIn("clearModel", source)
        self.assertEqual(source.count("SafeHighsOwner("), 1)
        self.assertEqual(source.count("apply_incremental_update("), 1)
        public = source.index(
            "def build_forward_exact_relu_phase_projection_candidate("
        )
        prefix = source[:public]
        self.assertNotIn("phase_projection_device_program as device_module", prefix)
        self.assertNotIn("phase_projection_highs_owner as owner_module", prefix)
        self.assertNotIn("phase_projection_incremental_repair as repair_module", prefix)
        self.assertLess(
            source.index("owner_state_after_close = highs_owner.state"),
            source.index("decoded_input, exact_margin, singleton_seconds"),
        )

    def test_public_wrapper_maps_ordinary_exception_to_unknown(self) -> None:
        with mock.patch.object(
            _projection,
            "_build_forward_exact_relu_phase_projection_candidate_impl",
            side_effect=ValueError("ordinary"),
        ):
            with self.assertRaisesRegex(
                ExactReLUPhaseProjectionUnknown,
                "request-local transaction failed closed",
            ):
                build_forward_exact_relu_phase_projection_candidate(
                    object(), 0, {}, {}
                )

    def test_public_wrapper_preserves_baseexception_identity(self) -> None:
        marker = KeyboardInterrupt("identity")
        with mock.patch.object(
            _projection,
            "_build_forward_exact_relu_phase_projection_candidate_impl",
            side_effect=marker,
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                build_forward_exact_relu_phase_projection_candidate(
                    object(), 0, {}, {}
                )
        self.assertIs(caught.exception, marker)


@unittest.skipUnless(torch.cuda.is_available(), "phase projection requires CUDA")
class ExactReLUPhaseProjectionCandidateTests(unittest.TestCase):
    def test_inward_factor_bounds_are_exactly_inside_input_box(self) -> None:
        lower = np.asarray([-0.1, 2.0**-1022], dtype=np.float64)
        upper = np.asarray([0.7, 9.0 * 2.0**-1022], dtype=np.float64)
        center, radius = _projection._oh._enclosing_center_radius(
            lower, upper, name="phase_projection_test"
        )
        rows = np.asarray([0, 1], dtype=np.int64)
        bounds = _projection._inward_factor_bounds(
            lower, upper, center, radius, rows, 1.0e-9
        )
        for row, (lo, hi) in zip(rows, bounds):
            exact_center = Fraction.from_float(float(center[row]))
            exact_radius = Fraction.from_float(float(radius[row]))
            decoded_lower = exact_center + exact_radius * Fraction.from_float(lo)
            decoded_upper = exact_center + exact_radius * Fraction.from_float(hi)
            self.assertGreaterEqual(decoded_lower, Fraction.from_float(float(lower[row])))
            self.assertLessEqual(decoded_upper, Fraction.from_float(float(upper[row])))

    def test_gpu_selected_csr_matches_grouped_dilated_reference(self) -> None:
        topology = _projection._live.get_exact_conv_spatial_topology(
            input_shape=(2, 4, 6, 7),
            output_shape=(2, 4, 3, 9),
            kernel=(2, 3),
            stride=(2, 1),
            padding=(1, 2),
            dilation=(2, 1),
            groups=2,
        )
        weight = np.arange(1, 49, dtype=np.float64).reshape(4, 2, 2, 3)
        weight[1::2] *= -1.0
        snapshot = SimpleNamespace(
            kind="CONV2D",
            input_size=2 * 4 * 6 * 7,
            output_size=2 * 4 * 3 * 9,
            weight=weight,
            topology=topology,
        )
        selected = np.array([0, 8, 27, 55, 108, 161, 215], dtype=np.int64)
        possible = np.ones(snapshot.input_size, dtype=bool)
        possible[np.arange(0, snapshot.input_size, 11)] = False
        cpu = _projection._live._selected_affine_matrix(
            snapshot, selected, possible, name="phase_projection.gpu.cpu"
        )
        gpu = _projection._live._gpu_selected_affine_matrix(
            snapshot, selected, possible, name="phase_projection.gpu"
        )
        np.testing.assert_array_equal(gpu.indptr.cpu().numpy(), cpu.indptr)
        np.testing.assert_array_equal(gpu.indices.cpu().numpy(), cpu.indices)
        np.testing.assert_array_equal(
            gpu.data.cpu().numpy().view(np.uint64), cpu.data.view(np.uint64)
        )

    def test_zero_scale_is_outside_the_single_candidate_domain(self) -> None:
        layer = SimpleNamespace(
            id=4,
            kind="SCALE",
            params={"a": torch.tensor([1.0, 0.0], dtype=torch.float64)},
        )
        with self.assertRaisesRegex(
            ExactReLUPhaseProjectionUnknown, "all-nonzero"
        ):
            _projection._pointwise_snapshot(layer, width=2)

    def test_base_lp_and_device_terminal_replay(self) -> None:
        toy, facts = _phase_projection_toy()
        result = build_forward_exact_relu_phase_projection_candidate(
            toy.net, 0, facts, facts
        )
        receipt = result.receipt
        self.assertEqual(receipt.status, "singleton_verified")
        self.assertEqual(receipt.phase_updates, 1)
        self.assertEqual(receipt.owner_instances, 1)
        self.assertEqual(receipt.owner_solves, 1)
        self.assertEqual(receipt.repair_updates, 0)
        self.assertEqual(receipt.resolves_after_base, 0)
        self.assertEqual(receipt.dual_ray_requests, 0)
        self.assertFalse(receipt.same_owner_warm_update_used)
        self.assertFalse(receipt.second_solver_used)
        self.assertEqual(receipt.phase_retries, 0)
        self.assertEqual(receipt.property_rows_selected, 1)
        self.assertEqual(receipt.property_row_retries, 0)
        self.assertEqual(receipt.phase_rows, 1)
        self.assertEqual(receipt.initial_phase_changes, 1)
        self.assertGreater(receipt.candidate_margin, 0.0)
        self.assertGreater(receipt.singleton_margin_lower, 0.0)
        self.assertEqual(receipt.generator_streams, 1)
        self.assertEqual(
            receipt.generator_representation,
            "request_local_device_program_incremental_lowrank_v1",
        )
        self.assertFalse(receipt.candidate_outward_error_bands_used)
        self.assertFalse(receipt.intermediate_phase_or_margin_replay_used)
        self.assertTrue(receipt.all_unstable_exact)
        self.assertEqual(receipt.triangle_rows, 0)
        self.assertFalse(receipt.input_sampling_used)
        self.assertFalse(receipt.pgd_used)
        self.assertFalse(receipt.concrete_onnx_execution_used)
        self.assertFalse(receipt.bab_used)
        self.assertFalse(receipt.backward_used)
        self.assertFalse(receipt.dual_tightening_used)
        self.assertFalse(receipt.dual_ray_authority)
        self.assertFalse(receipt.dual_selector_authority)
        self.assertFalse(receipt.candidate_authority)
        self.assertFalse(receipt.proof_authority)
        self.assertFalse(receipt.verdict_authority)
        self.assertTrue(
            receipt.same_stored_binary64_input_for_box_and_terminal
        )
        self.assertFalse(receipt.updated_full_target_materialized)
        self.assertEqual(receipt.updated_lp_nnz, receipt.base_loaded_nnz)
        self.assertEqual(receipt.updated_logical_nnz, receipt.base_logical_nnz)
        self.assertFalse(result.decoded_input.flags.writeable)
        self.assertGreater(float(result.decoded_input.reshape(-1)[0]), 0.25)
        self.assertLessEqual(float(result.decoded_input.reshape(-1)[0]), 1.0)

        exact_x = Fraction.from_float(float(result.decoded_input.reshape(-1)[0]))
        exact_y = max(Fraction(0), exact_x - Fraction(1, 4))
        self.assertGreater(exact_y, 0)

    def test_device_terminal_failure_is_unknown_not_a_verdict(self) -> None:
        toy, facts = _phase_projection_toy()
        from act.back_end.hybridz_tf import phase_projection_device_program

        def zero_terminal(decoded, program):
            del decoded, program
            return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

        with mock.patch.object(
            phase_projection_device_program,
            "terminal_interval_forward",
            side_effect=zero_terminal,
        ):
            with self.assertRaisesRegex(
                ExactReLUPhaseProjectionUnknown, "did not prove"
            ):
                build_forward_exact_relu_phase_projection_candidate(
                    toy.net, 0, facts, facts
                )

    def test_non_top1_property_fails_before_solver(self) -> None:
        toy, facts = _phase_projection_toy()
        from act.back_end.hybridz_tf import phase_projection_highs_owner

        toy.net.by_id[5].params["C"] = torch.tensor(
            [[1.0, 0.5]], dtype=torch.float64, device="cuda"
        )
        with mock.patch.object(
            phase_projection_highs_owner, "SafeHighsOwner"
        ) as owner:
            with self.assertRaisesRegex(
                ExactReLUPhaseProjectionUnknown, "TOP1"
            ):
                build_forward_exact_relu_phase_projection_candidate(
                    toy.net, 0, facts, facts
                )
        owner.assert_not_called()


class ExactReLUPhaseProjectionConfigTests(unittest.TestCase):
    def test_enabled_mode_is_narrow_and_single_path(self) -> None:
        valid = HybridZConfig(
            engine="operator_hz_objbound",
            operator_exact_budget=-1,
            operator_phase_projection_time_limit=1.0,
        )
        self.assertEqual(valid.operator_phase_projection_time_limit, 1.0)

        invalid = (
            {
                "engine": "dense_hz_objbound",
                "operator_exact_budget": -1,
                "operator_phase_projection_time_limit": 1.0,
            },
            {
                "engine": "operator_hz_objbound",
                "operator_exact_budget": 0,
                "operator_phase_projection_time_limit": 1.0,
            },
            {
                "engine": "operator_hz_objbound",
                "operator_exact_budget": -1,
                "operator_phase_projection_time_limit": 1.0,
                "gpu_dual_steps": 1,
                "gpu_dual_time_limit": 1.0,
            },
            {
                "engine": "operator_hz_objbound",
                "operator_exact_budget": -1,
                "operator_phase_projection_time_limit": 1.0,
                "property_tail_upper": True,
            },
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    HybridZConfig(**kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "phase projection requires CUDA")
class ExactReLUPhaseProjectionVerifierIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")

    def tearDown(self) -> None:
        set_solver_mode(None)
        set_transfer_function_mode("interval")

    def test_verifier_owned_projection_falsifies_without_external_replay(
        self,
    ) -> None:
        toy, facts = _phase_projection_toy()
        toy.net.by_id[5].in_vars = list(toy.net.by_id[4].out_vars)
        toy.net.by_id[5].params["M"] = 1
        toy.net.by_id[5].params["thresholds"] = torch.tensor(
            [[0.0]], dtype=torch.float64, device="cuda"
        )
        config = BackendConfig(
            solver="hybridz",
            device="cuda",
            dtype="float64",
            timeout=6.0,
            hybridz=HybridZConfig(
                timeout=5.0,
                engine="operator_hz_objbound",
                operator_exact_budget=-1,
                operator_phase_projection_time_limit=4.0,
            ),
        )

        with (
            mock.patch(
                "act.back_end.analyze.analyze",
                return_value=(facts, facts, ConSet()),
            ),
            mock.patch(
                "act.back_end.solver.solver_hz.hz_objbound_decide",
                side_effect=AssertionError(
                    "verified projection must return before root MILP"
                ),
            ) as root_solver,
        ):
            results, collected = verify_once(
                toy.net,
                backend_cfg=config,
                collect_facts=True,
            )
            result = results[0]

        self.assertEqual(result.status, VerifyStatus.FALSIFIED)
        self.assertIsNotNone(result.counterexample)
        self.assertIs(collected, facts)
        root_solver.assert_not_called()
        receipt = result.metadata["operator_phase_projection"]
        self.assertEqual(receipt["status"], "FALSIFIED")
        self.assertTrue(receipt["verifier_owned_proof_authority"])
        self.assertFalse(receipt["input_sampling_used"])
        self.assertFalse(receipt["pgd_used"])
        self.assertFalse(receipt["concrete_onnx_execution_used"])
        self.assertFalse(receipt["bab_used"])
        self.assertFalse(receipt["backward_used"])
        self.assertFalse(receipt["dual_tightening_used"])
        self.assertIn("zero_width_forward_interval", receipt["proof_rule"])
        candidate = receipt["candidate_receipt"]
        self.assertFalse(candidate["proof_authority"])
        self.assertFalse(candidate["verdict_authority"])

    def test_projection_failure_is_immediate_unknown_without_root_fallback(
        self,
    ) -> None:
        toy, facts = _phase_projection_toy()
        toy.net.by_id[5].in_vars = list(toy.net.by_id[4].out_vars)
        toy.net.by_id[5].params["M"] = 1
        toy.net.by_id[5].params["thresholds"] = torch.tensor(
            [[0.0]], dtype=torch.float64, device="cuda"
        )
        config = BackendConfig(
            solver="hybridz",
            device="cuda",
            dtype="float64",
            timeout=6.0,
            hybridz=HybridZConfig(
                timeout=5.0,
                engine="operator_hz_objbound",
                operator_exact_budget=-1,
                operator_phase_projection_time_limit=4.0,
            ),
        )

        with (
            mock.patch(
                "act.back_end.analyze.analyze",
                return_value=(facts, facts, ConSet()),
            ),
            mock.patch.object(
                _projection,
                "build_forward_exact_relu_phase_projection_candidate",
                side_effect=ExactReLUPhaseProjectionUnknown("not applicable"),
            ),
            mock.patch(
                "act.back_end.solver.solver_hz.hz_objbound_decide",
                side_effect=AssertionError("single-path mode forbids fallback"),
            ) as root_solver,
        ):
            result = verify_once(toy.net, backend_cfg=config)[0]

        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertIsNone(result.counterexample)
        root_solver.assert_not_called()
        receipt = result.metadata["operator_phase_projection"]
        self.assertEqual(receipt["status"], "UNKNOWN")
        self.assertEqual(receipt["reason"], "not applicable")
        self.assertFalse(receipt["verifier_owned_proof_authority"])

    def test_malformed_controls_fail_closed(self) -> None:
        toy, facts = _phase_projection_toy()
        for kwargs in (
            {"entry_layer_id": True},
            {"entry_layer_id": 1},
            {"entry_layer_id": 0, "lp_time_limit": 0.0},
            {"entry_layer_id": 0, "lp_time_limit": 31.0},
            {"entry_layer_id": 0, "deadline": float("nan")},
        ):
            entry = kwargs.pop("entry_layer_id")
            with self.subTest(entry=entry, kwargs=kwargs):
                with self.assertRaises(ExactReLUPhaseProjectionUnknown):
                    build_forward_exact_relu_phase_projection_candidate(
                        toy.net, entry, facts, facts, **kwargs
                    )

    def test_raw_box_must_be_enclosed_by_forward_input_fact(self) -> None:
        toy, facts = _phase_projection_toy()
        from act.back_end.hybridz_tf import phase_projection_highs_owner

        narrowed = dict(facts)
        narrowed[0] = _fact([-0.5], [0.5])
        with mock.patch.object(
            phase_projection_highs_owner, "SafeHighsOwner"
        ) as owner:
            with self.assertRaisesRegex(
                ExactReLUPhaseProjectionUnknown, "does not enclose"
            ):
                build_forward_exact_relu_phase_projection_candidate(
                    toy.net, 0, facts, narrowed
                )
        owner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
