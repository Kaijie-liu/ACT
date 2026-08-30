#!/usr/bin/env python3
"""CPU/float64 gates for the bounded parent micro-RLT prefilter."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.hybridz_tf.test_binary_phase_split import (
    _verified_duplicate_relu_residual_net,
)
from act.back_end.hybridz_tf.test_operator_micro_rlt import _build
from act.back_end.solver.solver_hz import hz_objbound_decide
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once
from act.util.stats import VerifyStatus


def _phase_config(
    *,
    enabled: bool,
    parent_only: bool = False,
    product_cap: int = 64,
    packet_mode: str = "both",
    timeout: float = 5.0,
    gpu_steps: int = 0,
    gpu_seconds: float = 0.0,
) -> HybridZConfig:
    return HybridZConfig(
        timeout=timeout,
        engine="operator_hz_objbound",
        operator_exact_budget=2,
        property_residual_budget=2,
        property_residual_time_limit=1.0,
        property_residual_max_adjoint_cells=64,
        property_residual_pool_per_rival=4,
        property_tail_upper=True,
        property_tail_suffix_blocks=1,
        property_micro_rlt_product_cap=product_cap if enabled else 0,
        property_micro_rlt_packet_mode=packet_mode,
        property_micro_rlt_parent_prefilter_seconds=(
            1.0 if enabled else 0.0
        ),
        property_micro_rlt_parent_only_diagnostic=parent_only,
        gpu_dual_steps=gpu_steps,
        gpu_dual_time_limit=gpu_seconds,
        # The dedicated parent call overrides LP allocation.  GPU candidates
        # remain disabled except in the explicitly parent-only diagnostic.
        lp_prefilter_fraction=0.0,
        lp_prefilter_max_seconds=0.0,
    )


def _verify_with(config: HybridZConfig):
    return verify_once(
        _verified_duplicate_relu_residual_net(),
        backend_cfg=BackendConfig(
            solver="hybridz",
            device="cpu",
            dtype="float64",
            hybridz=config,
        ),
    )[0]


class PropertyMicroRLTConfigTests(unittest.TestCase):
    def test_valid_depth_two_contract_and_exported_fields(self) -> None:
        config = _phase_config(enabled=True)
        self.assertEqual(config.property_micro_rlt_product_cap, 64)
        self.assertEqual(config.property_micro_rlt_packet_mode, "both")
        self.assertEqual(
            config.property_micro_rlt_parent_prefilter_seconds, 1.0
        )
        self.assertFalse(
            config.property_micro_rlt_parent_only_diagnostic
        )
        for mode in ("first", "second"):
            self.assertEqual(
                _phase_config(
                    enabled=True, packet_mode=mode
                ).property_micro_rlt_packet_mode,
                mode,
            )

    def test_parent_only_diagnostic_is_strict_default_off_and_coupled(
        self,
    ) -> None:
        self.assertFalse(
            HybridZConfig().property_micro_rlt_parent_only_diagnostic
        )
        for value in (0, 1, None, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "parent_only_diagnostic must be a boolean",
                ):
                    HybridZConfig(
                        property_micro_rlt_parent_only_diagnostic=value,
                    )
        with self.assertRaisesRegex(
            ValueError,
            "parent_only_diagnostic requires property micro-RLT",
        ):
            HybridZConfig(
                property_micro_rlt_parent_only_diagnostic=True,
            )
        configured = _phase_config(
            enabled=True,
            parent_only=True,
        )
        self.assertTrue(
            configured.property_micro_rlt_parent_only_diagnostic
        )

    def test_strict_types_ranges_and_coupled_enablement(self) -> None:
        common = {
            "engine": "operator_hz_objbound",
            "operator_exact_budget": 2,
            "property_residual_budget": 2,
            "property_residual_time_limit": 1.0,
            "property_tail_upper": True,
            "property_tail_suffix_blocks": 1,
        }
        for value in (True, 1.0, "1", None):
            with self.subTest(product_cap=value):
                with self.assertRaisesRegex(
                    ValueError, "product_cap must be an integer"
                ):
                    HybridZConfig(
                        **common,
                        property_micro_rlt_product_cap=value,
                    )
        for value in (-1, 4097):
            with self.subTest(product_cap=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 4096\]"):
                    HybridZConfig(
                        **common,
                        property_micro_rlt_product_cap=value,
                    )
        for value in (None, "", "forward", 1, True):
            with self.subTest(packet_mode=value):
                with self.assertRaisesRegex(
                    ValueError, "packet_mode"
                ):
                    HybridZConfig(
                        **common,
                        property_micro_rlt_packet_mode=value,
                    )
        with self.assertRaisesRegex(
            ValueError, "first/second requires"
        ):
            HybridZConfig(
                **common,
                property_micro_rlt_packet_mode="first",
            )
        for value in (True, "1", None):
            with self.subTest(prefilter_seconds=value):
                with self.assertRaisesRegex(ValueError, "must be numeric"):
                    HybridZConfig(
                        **common,
                        property_micro_rlt_parent_prefilter_seconds=value,
                    )
        for value in (float("nan"), float("inf"), -0.1, 10.1):
            with self.subTest(prefilter_seconds=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 10\]"):
                    HybridZConfig(
                        **common,
                        property_micro_rlt_parent_prefilter_seconds=value,
                    )
        with self.assertRaisesRegex(ValueError, "enabled together"):
            HybridZConfig(
                **common,
                property_micro_rlt_product_cap=64,
            )
        with self.assertRaisesRegex(ValueError, "enabled together"):
            HybridZConfig(
                **common,
                property_micro_rlt_parent_prefilter_seconds=1.0,
            )

    def test_positive_mode_requires_operator_depth_two_property_tail(self):
        enabled = {
            "property_micro_rlt_product_cap": 64,
            "property_micro_rlt_parent_prefilter_seconds": 1.0,
        }
        with self.assertRaisesRegex(
            ValueError, "engine=operator_hz_objbound"
        ):
            HybridZConfig(
                engine="dense_hz_objbound",
                operator_exact_budget=2,
                property_residual_budget=2,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=1,
                **enabled,
            )
        with self.assertRaisesRegex(ValueError, "property_tail_upper=true"):
            HybridZConfig(
                engine="operator_hz_objbound",
                operator_exact_budget=2,
                property_residual_budget=2,
                property_residual_time_limit=1.0,
                property_tail_upper=False,
                property_tail_suffix_blocks=1,
                **enabled,
            )
        with self.assertRaisesRegex(ValueError, "depth-2"):
            HybridZConfig(
                engine="operator_hz_objbound",
                operator_exact_budget=1,
                property_residual_budget=1,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=1,
                **enabled,
            )


class PropertyMicroRLTRealLPCertificateTests(unittest.TestCase):
    def test_lift_is_the_causal_safe_only_lp_difference(self) -> None:
        C = np.asarray([[1.0]], dtype=np.float64)
        thresholds = np.asarray([0.1], dtype=np.float64)
        solver_kwargs = {
            "is_unsafe_linear": False,
            "time_limit": 2.0,
            "base_witness_precheck": False,
            "lp_prefilter_fraction": 1.0,
            "lp_prefilter_max_seconds": 1.5,
            "gpu_dual_steps": 0,
            "gpu_dual_time_limit": 0.0,
            "gpu_dual_row_topk": 0,
            "safe_row_groups": ((0,),),
            "expected_safe_group_count": 1,
            "safe_group_mixture_grid_bits": 0,
        }

        baseline = _build(0)
        baseline_verdict, baseline_witness = hz_objbound_decide(
            baseline.hz, C, thresholds, **solver_kwargs
        )
        baseline_stats = dict(baseline.hz._solver_objbound_stats)

        lifted = _build(64)
        lifted_verdict, lifted_witness = hz_objbound_decide(
            lifted.hz, C, thresholds, **solver_kwargs
        )
        lifted_stats = dict(lifted.hz._solver_objbound_stats)

        self.assertEqual(baseline_verdict, "UNKNOWN")
        self.assertIsNone(baseline_witness)
        self.assertEqual(baseline_stats["lp_certified_rows"], 0)
        self.assertGreater(baseline_stats["lp_cert_max_upper"], 0.0)
        self.assertFalse(baseline_stats["all_rivals_covered"])

        self.assertEqual(lifted_verdict, "SAFE")
        self.assertIsNone(lifted_witness)
        self.assertGreater(lifted_stats["lp_certified_rows"], 0)
        self.assertLess(lifted_stats["lp_cert_max_upper"], 0.0)
        self.assertTrue(lifted_stats["all_rivals_covered"])
        self.assertTrue(
            lifted_stats["lp_binary_relaxation_certificate_eligible"]
        )
        self.assertFalse(
            lifted_stats["lp_candidate_witness_eligible"]
        )
        # The cube cannot see constraint-only RLT rows, so both cases enter
        # the checked LP and the SAFE difference is attributable to the lift.
        self.assertGreater(baseline_stats["cube_max_upper"], 0.0)
        self.assertGreater(lifted_stats["cube_max_upper"], 0.0)


class ObjboundParentStageTimingTests(unittest.TestCase):
    def _solver_kwargs(self):
        return {
            "is_unsafe_linear": False,
            "time_limit": 0.5,
            "require_base_feasible": False,
            "base_witness_precheck": False,
            "lp_prefilter_fraction": 1.0,
            "lp_prefilter_max_seconds": 0.5,
            "gpu_dual_steps": 0,
            "gpu_dual_time_limit": 0.0,
            "gpu_dual_row_topk": 0,
        }

    def test_deadline_after_base_matrix_keeps_shape_and_exit_stats(
        self,
    ) -> None:
        from act.back_end.solver import solver_hz

        hz = _build(0).hz
        original_materialize = (
            solver_hz._base_milp_matrices_from_blocks
        )
        clock = {"matrix_complete": False}

        def controlled_clock():
            return 101.0 if clock["matrix_complete"] else 100.0

        def materialize_then_exhaust(*args, **kwargs):
            result = original_materialize(*args, **kwargs)
            clock["matrix_complete"] = True
            return result

        with patch.object(
            solver_hz.time,
            "monotonic",
            side_effect=controlled_clock,
        ), patch.object(
            solver_hz,
            "_base_milp_matrices_from_blocks",
            side_effect=materialize_then_exhaust,
        ), patch.object(
            solver_hz,
            "_hz_persistent_lp_filter",
        ) as persistent_lp:
            verdict, witness = solver_hz.hz_objbound_decide(
                hz,
                np.asarray([[1.0]], dtype=np.float64),
                np.asarray([0.1], dtype=np.float64),
                **self._solver_kwargs(),
            )

        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        persistent_lp.assert_not_called()
        stats = hz._solver_objbound_stats
        self.assertEqual(
            stats["parent_stage_timing_schema"],
            "hz_objbound_parent_stage_timing_v1",
        )
        self.assertTrue(
            stats["parent_stage_timings_diagnostic_only"]
        )
        self.assertFalse(
            stats["parent_stage_timings_proof_authority"]
        )
        self.assertEqual(
            stats["parent_exit_reason"],
            "deadline_after_base_matrix_materialization",
        )
        self.assertEqual(
            stats["parent_base_matrix_materialization_status"],
            "completed",
        )
        self.assertEqual(
            stats["parent_base_matrix_materialization_elapsed_s"],
            1.0,
        )
        self.assertGreater(stats["parent_base_matrix_rows"], 0)
        self.assertGreater(stats["parent_base_matrix_columns"], 0)
        self.assertGreater(stats["parent_base_matrix_nnz"], 0)
        self.assertEqual(
            stats["parent_persistent_lp_status"], "not_started"
        )

    def test_deadline_after_persistent_lp_keeps_inner_and_stage_stats(
        self,
    ) -> None:
        from act.back_end.solver import solver_hz

        hz = _build(0).hz
        clock = {"persistent_complete": False}

        def controlled_clock():
            return (
                201.0
                if clock["persistent_complete"]
                else 200.0
            )

        def persistent_then_exhaust(**kwargs):
            rows = np.asarray(
                kwargs["candidate_rows"], dtype=np.int64
            ).copy()
            clock["persistent_complete"] = True
            return (
                rows,
                {
                    "lp_status": "controlled_deadline",
                    "lp_coverage_ok": True,
                    "lp_certified_rows": 0,
                },
                None,
            )

        with patch.object(
            solver_hz.time,
            "monotonic",
            side_effect=controlled_clock,
        ), patch.object(
            solver_hz,
            "_hz_persistent_lp_filter",
            side_effect=persistent_then_exhaust,
        ) as persistent_lp:
            verdict, witness = solver_hz.hz_objbound_decide(
                hz,
                np.asarray([[1.0]], dtype=np.float64),
                np.asarray([0.1], dtype=np.float64),
                **self._solver_kwargs(),
            )

        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        persistent_lp.assert_called_once()
        stats = hz._solver_objbound_stats
        self.assertEqual(
            stats["parent_exit_reason"],
            "deadline_after_persistent_lp",
        )
        self.assertEqual(
            stats["parent_persistent_lp_status"],
            "completed_after_deadline",
        )
        self.assertEqual(
            stats["parent_persistent_lp_elapsed_s"], 1.0
        )
        self.assertEqual(
            stats["parent_persistent_lp_input_rows"], 1
        )
        self.assertEqual(
            stats["parent_persistent_lp_output_rows"], 1
        )
        self.assertEqual(
            stats["parent_persistent_lp_budget_s"], 0.5
        )
        # The inner filter's own receipt must be merged before the timeout
        # return, otherwise a large parent still looks as if LP never ran.
        self.assertEqual(stats["lp_status"], "controlled_deadline")
        self.assertGreater(stats["parent_base_matrix_rows"], 0)
        self.assertGreater(stats["parent_base_matrix_nnz"], 0)


class PropertyMicroRLTVerifierRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")

    def tearDown(self) -> None:
        set_solver_mode(None)
        set_transfer_function_mode("interval")

    def assert_parent_only_receipt(
        self,
        result,
        *,
        status: VerifyStatus = VerifyStatus.UNKNOWN,
    ) -> None:
        self.assertEqual(result.status, status)
        self.assertEqual(result.metadata["hz_verdict"], "UNKNOWN")
        self.assertFalse(result.metadata["hz_has_witness"])
        self.assertEqual(
            result.metadata["reason"],
            "property_micro_rlt_parent_only_diagnostic",
        )
        diagnostic = result.metadata[
            "property_micro_rlt_parent_only_diagnostic"
        ]
        self.assertEqual(
            diagnostic["schema"],
            (
                "verifier_property_micro_rlt_"
                "parent_only_diagnostic_v1"
            ),
        )
        self.assertTrue(diagnostic["diagnostic_only"])
        self.assertFalse(diagnostic["proof_authority"])
        self.assertTrue(diagnostic["verdict_forced_unknown"])
        self.assertFalse(diagnostic["phase_cover_attempted"])
        self.assertEqual(diagnostic["phase_children_created"], 0)
        self.assertFalse(diagnostic["baseline_solver_attempted"])
        self.assertIsInstance(diagnostic["receipt_sha256"], str)
        self.assertEqual(len(diagnostic["receipt_sha256"]), 64)
        phase = result.metadata["property_phase_split"]
        self.assertFalse(phase["proof_authority"])
        self.assertTrue(phase["diagnostic_only"])
        self.assertTrue(phase["phase_enumeration_skipped"])
        self.assertEqual(phase["actual_child_count"], 0)
        self.assertEqual(phase["children"], [])

    def test_default_off_never_calls_the_binary_parent(self) -> None:
        binary_depths = []

        def undecided_children(hz, *_args, **_kwargs):
            binary_depths.append(int(hz.n_bin))
            return "UNKNOWN", None

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=undecided_children,
        ):
            result = _verify_with(_phase_config(enabled=False))

        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(binary_depths, [0, 0, 0, 0])
        receipt = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(receipt["status"], "disabled")
        self.assertEqual(receipt["parent_call_count"], 0)
        self.assertEqual(
            result.metadata["cfg_property_micro_rlt_product_cap"], 0
        )
        self.assertEqual(
            result.metadata[
                "cfg_property_micro_rlt_parent_prefilter_seconds"
            ],
            0.0,
        )

    def test_parent_only_cap_no_op_stops_without_any_solver(self) -> None:
        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide"
        ) as decide, patch(
            "act.back_end.solver.solver_hz."
            "hz_enumerate_sparse_binary_phase_cover"
        ) as enumerate_cover:
            result = _verify_with(
                _phase_config(
                    enabled=True,
                    parent_only=True,
                    product_cap=1,
                )
            )

        self.assert_parent_only_receipt(result)
        decide.assert_not_called()
        enumerate_cover.assert_not_called()
        operator_receipt = result.metadata["operator_hz"][
            "property_micro_rlt"
        ]
        self.assertEqual(
            operator_receipt["status"],
            "no_op_cap_exceeded",
        )
        parent = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            parent["status"],
            "operator_receipt_ineligible_diagnostic_stop",
        )
        self.assertEqual(parent["parent_call_count"], 0)
        self.assertFalse(parent["proof_authority"])

    def test_parent_only_unknown_is_one_parent_call_and_no_children(
        self,
    ) -> None:
        binary_depths = []

        def parent_unknown(hz, *_args, **_kwargs):
            binary_depths.append(int(hz.n_bin))
            hz._solver_objbound_stats = {
                "all_rivals_covered": False,
            }
            return "UNKNOWN", None

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=parent_unknown,
        ), patch(
            "act.back_end.solver.solver_hz."
            "hz_enumerate_sparse_binary_phase_cover"
        ) as enumerate_cover:
            result = _verify_with(
                _phase_config(enabled=True, parent_only=True)
            )

        self.assert_parent_only_receipt(result)
        self.assertEqual(binary_depths, [2])
        enumerate_cover.assert_not_called()
        parent = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            parent["status"],
            "parent_unknown_diagnostic_stop",
        )
        self.assertEqual(parent["parent_call_count"], 1)
        self.assertFalse(parent["proof_authority"])

    def test_parent_only_can_enable_candidate_only_gpu_without_promotion(
        self,
    ) -> None:
        observed = {}

        def parent_unknown(hz, *_args, **kwargs):
            observed.update(kwargs)
            hz._solver_objbound_stats = {
                "all_rivals_covered": False,
                "gpu_dual_status": "diagnostic_sentinel",
                "gpu_dual_pc_cbde_status": "verified_replaced",
                "gpu_dual_pc_cbde_cone_rows": [7, 9],
                "gpu_dual_pc_cbde_full_nnz": 3,
                "gpu_dual_pc_cbde_checked_upper_full": -0.25,
                "gpu_dual_pc_cbde_strict_family_ablation": True,
                "gpu_dual_pc_cbde_support_improvement_tol": 1.0e-13,
                "gpu_dual_pc_cbde_proof_authority": False,
                "gpu_dual_pc_cbde_unlisted_poison": "must_be_filtered",
            }
            return "UNKNOWN", None

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=parent_unknown,
        ), patch(
            "act.back_end.solver.solver_hz."
            "hz_enumerate_sparse_binary_phase_cover"
        ) as enumerate_cover:
            result = _verify_with(
                _phase_config(
                    enabled=True,
                    parent_only=True,
                    gpu_steps=2,
                    gpu_seconds=0.25,
                )
            )

        self.assert_parent_only_receipt(result)
        enumerate_cover.assert_not_called()
        self.assertEqual(observed["gpu_dual_steps"], 2)
        self.assertEqual(observed["gpu_dual_time_limit"], 0.25)
        parent = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertTrue(parent["gpu_candidates_enabled"])
        self.assertEqual(
            parent["stats"]["gpu_dual_status"],
            "diagnostic_sentinel",
        )
        self.assertEqual(
            parent["stats"]["gpu_dual_pc_cbde_status"],
            "verified_replaced",
        )
        self.assertEqual(
            parent["stats"]["gpu_dual_pc_cbde_cone_rows"],
            [7, 9],
        )
        self.assertEqual(
            parent["stats"]["gpu_dual_pc_cbde_checked_upper_full"],
            -0.25,
        )
        self.assertTrue(
            parent["stats"][
                "gpu_dual_pc_cbde_strict_family_ablation"
            ]
        )
        self.assertEqual(
            parent["stats"][
                "gpu_dual_pc_cbde_support_improvement_tol"
            ],
            1.0e-13,
        )
        self.assertNotIn(
            "gpu_dual_pc_cbde_unlisted_poison",
            parent["stats"],
        )
        self.assertFalse(parent["proof_authority"])

    def test_parent_only_real_safe_is_observed_but_forced_unknown(
        self,
    ) -> None:
        binary_depths = []

        def real_parent(hz, C, thresholds, **kwargs):
            binary_depths.append(int(hz.n_bin))
            return hz_objbound_decide(hz, C, thresholds, **kwargs)

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=real_parent,
        ), patch(
            "act.back_end.solver.solver_hz."
            "hz_enumerate_sparse_binary_phase_cover"
        ) as enumerate_cover:
            result = _verify_with(
                _phase_config(enabled=True, parent_only=True)
            )

        self.assert_parent_only_receipt(result)
        self.assertEqual(binary_depths, [2])
        enumerate_cover.assert_not_called()
        parent = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            parent["status"],
            "parent_safe_observed_diagnostic_stop",
        )
        self.assertEqual(parent["parent_call_count"], 1)
        self.assertTrue(parent["safe_contract_valid"])
        self.assertIsNotNone(parent["safe_capability"])
        self.assertFalse(parent["proof_authority"])
        diagnostic = result.metadata[
            "property_micro_rlt_parent_only_diagnostic"
        ]
        self.assertEqual(
            diagnostic["parent_solver_verdict"],
            "SAFE",
        )
        self.assertTrue(
            diagnostic["parent_safe_contract_observed"]
        )

    def test_parent_only_exception_is_receipted_without_fallback(
        self,
    ) -> None:
        calls = []

        def parent_raises(hz, *_args, **_kwargs):
            calls.append(int(hz.n_bin))
            raise RuntimeError("controlled parent-only failure")

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=parent_raises,
        ), patch(
            "act.back_end.solver.solver_hz."
            "hz_enumerate_sparse_binary_phase_cover"
        ) as enumerate_cover:
            result = _verify_with(
                _phase_config(enabled=True, parent_only=True)
            )

        self.assert_parent_only_receipt(result)
        self.assertEqual(calls, [2])
        enumerate_cover.assert_not_called()
        parent = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            parent["status"],
            "parent_error_diagnostic_stop",
        )
        self.assertIn("controlled parent-only failure", parent["error"])
        self.assertFalse(parent["proof_authority"])

    def test_parent_only_shared_deadline_returns_timeout(self) -> None:
        def overrun_parent(_hz, *_args, **kwargs):
            time.sleep(float(kwargs["time_limit"]) + 0.02)
            return "UNKNOWN", None

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=overrun_parent,
        ), patch(
            "act.back_end.solver.solver_hz."
            "hz_enumerate_sparse_binary_phase_cover"
        ) as enumerate_cover:
            result = _verify_with(
                _phase_config(
                    enabled=True,
                    parent_only=True,
                    timeout=0.5,
                )
            )

        self.assert_parent_only_receipt(
            result,
            status=VerifyStatus.TIMEOUT,
        )
        enumerate_cover.assert_not_called()
        diagnostic = result.metadata[
            "property_micro_rlt_parent_only_diagnostic"
        ]
        self.assertTrue(diagnostic["shared_deadline_expired"])
        self.assertEqual(
            result.metadata["timeout_stage"],
            "property_micro_rlt_parent_only_diagnostic",
        )

    def test_parent_safe_is_one_call_and_skips_phase_enumeration(self):
        calls = []

        def safe_parent(hz, C, thresholds, **kwargs):
            calls.append((hz, C.copy(), thresholds.copy(), dict(kwargs)))
            return hz_objbound_decide(hz, C, thresholds, **kwargs)

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=safe_parent,
        ), patch(
            "act.back_end.solver.solver_hz."
            "hz_enumerate_sparse_binary_phase_cover"
        ) as enumerate_cover:
            result = _verify_with(_phase_config(enabled=True))

        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(
            result.metadata["reason"],
            "parent_binary_relaxation_safe",
        )
        self.assertEqual(len(calls), 1)
        hz, C, thresholds, kwargs = calls[0]
        self.assertEqual(hz.n_bin, 2)
        np.testing.assert_array_equal(C, np.eye(C.shape[0]))
        np.testing.assert_array_equal(
            thresholds, np.zeros_like(thresholds)
        )
        self.assertFalse(kwargs["base_witness_precheck"])
        self.assertEqual(kwargs["lp_prefilter_fraction"], 1.0)
        self.assertGreater(kwargs["lp_prefilter_max_seconds"], 0.0)
        self.assertLessEqual(kwargs["lp_prefilter_max_seconds"], 1.0)
        self.assertEqual(kwargs["gpu_dual_steps"], 0)
        self.assertEqual(kwargs["gpu_dual_time_limit"], 0.0)
        self.assertEqual(kwargs["safe_group_mixture_grid_bits"], 0)
        self.assertIsNotNone(kwargs["safe_row_groups"])
        enumerate_cover.assert_not_called()

        receipt = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            receipt["status"], "parent_binary_relaxation_safe"
        )
        self.assertTrue(receipt["proof_authority"])
        self.assertEqual(receipt["parent_call_count"], 1)
        self.assertEqual(receipt["phase_children_created"], 0)
        self.assertGreater(receipt["stats"]["lp_certified_rows"], 0)
        self.assertIsNotNone(receipt["safe_capability"])
        self.assertTrue(receipt["binary_relaxation_attributed"])
        phase = result.metadata["property_phase_split"]
        self.assertEqual(
            phase["status"], "parent_binary_relaxation_safe"
        )
        self.assertEqual(phase["actual_child_count"], 0)
        self.assertEqual(phase["children"], [])
        self.assertTrue(phase["phase_enumeration_skipped"])

    def test_bare_safe_and_forged_stats_have_no_promotion_authority(
        self,
    ) -> None:
        def forged_safe(hz, *_args, **_kwargs):
            hz._solver_objbound_stats = {
                "base_feasibility_status": "FEASIBLE",
                "safe_row_groups_enabled": True,
                "safe_row_group_count": 1,
                "safe_row_groups_resolved": 1,
                "safe_row_groups_unresolved": 0,
                "all_rivals_covered": True,
                "lp_safe_certificate_eligible": True,
                "lp_binary_relaxation_certificate_eligible": True,
                "lp_candidate_witness_eligible": False,
                "lp_proof_authority": True,
                "lp_coverage_ok": True,
            }
            return "SAFE", None

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=forged_safe,
        ):
            result = _verify_with(_phase_config(enabled=True))

        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        receipt = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            receipt["status"],
            "contract_mismatch_fallback_phase_cover",
        )
        self.assertFalse(receipt["proof_authority"])
        self.assertIsNone(receipt["safe_capability"])
        self.assertFalse(
            result.metadata["property_phase_split"]["proof_authority"]
        )

    def test_real_duplicate_relu_verifier_certifies_at_parent(self) -> None:
        result = _verify_with(_phase_config(enabled=True))

        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        receipt = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            receipt["status"], "parent_binary_relaxation_safe"
        )
        self.assertTrue(receipt["proof_authority"])
        self.assertEqual(receipt["parent_call_count"], 1)
        self.assertGreater(receipt["stats"]["lp_certified_rows"], 0)
        self.assertLess(receipt["stats"]["lp_cert_max_upper"], 0.0)
        self.assertTrue(
            receipt["stats"][
                "lp_binary_relaxation_certificate_eligible"
            ]
        )
        self.assertFalse(
            receipt["stats"]["lp_candidate_witness_eligible"]
        )
        self.assertTrue(receipt["stats"]["all_rivals_covered"])
        self.assertEqual(
            result.metadata["property_phase_split"]["actual_child_count"],
            0,
        )

    def test_directed_packets_preserve_parent_or_complete_phase_proof(
        self,
    ) -> None:
        first = _verify_with(
            _phase_config(enabled=True, packet_mode="first")
        )
        second = _verify_with(
            _phase_config(enabled=True, packet_mode="second")
        )
        self.assertEqual(first.status, VerifyStatus.CERTIFIED)
        self.assertEqual(second.status, VerifyStatus.UNKNOWN)
        self.assertEqual(
            first.metadata["operator_hz"]["property_micro_rlt"][
                "requested_packet_mode"
            ],
            "first",
        )
        self.assertEqual(
            second.metadata["operator_hz"]["property_micro_rlt"][
                "requested_packet_mode"
            ],
            "second",
        )

        first_parent = first.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(first_parent["solver_verdict"], "SAFE")
        self.assertTrue(first_parent["proof_authority"])
        self.assertEqual(
            first.metadata["property_phase_split"][
                "actual_child_count"
            ],
            0,
        )
        self.assertTrue(
            first.metadata["property_phase_split"]["proof_authority"]
        )

        second_parent = second.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(second_parent["solver_verdict"], "UNKNOWN")
        self.assertFalse(second_parent["proof_authority"])
        self.assertEqual(
            second.metadata["property_phase_split"][
                "actual_child_count"
            ],
            4,
        )
        self.assertFalse(
            second.metadata["property_phase_split"]["proof_authority"]
        )

    def test_parent_unknown_falls_back_to_the_complete_four_child_cover(
        self,
    ) -> None:
        from act.back_end.solver import solver_hz

        binary_depths = []
        lock = threading.Lock()

        def parent_unknown_children_safe(hz, *_args, **_kwargs):
            with lock:
                binary_depths.append(int(hz.n_bin))
            if hz.n_bin:
                hz._solver_objbound_stats = {
                    "all_rivals_covered": False,
                }
                return "UNKNOWN", None
            child_kwargs = dict(_kwargs)
            child_kwargs["lp_prefilter_fraction"] = 1.0
            child_kwargs["lp_prefilter_max_seconds"] = 2.0
            verdict, witness = hz_objbound_decide(
                hz, *_args, **child_kwargs
            )
            hz._solver_objbound_stats.update(
                {
                    "gpu_dual_pc_cbde_status": "child_diagnostic",
                    "gpu_dual_pc_cbde_cone_rows": [3],
                    "gpu_dual_pc_cbde_checked_upper_full": -0.5,
                    "gpu_dual_pc_cbde_support_improvement_tol": (
                        1.0e-13
                    ),
                    "gpu_dual_pc_cbde_proof_authority": False,
                    "gpu_dual_pc_cbde_unlisted_poison": (
                        "must_be_filtered"
                    ),
                }
            )
            return verdict, witness

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=parent_unknown_children_safe,
        ), patch(
            "act.back_end.solver.solver_hz."
            "hz_verify_sparse_binary_phase_child",
            wraps=solver_hz.hz_verify_sparse_binary_phase_child,
        ) as live_projection_audit:
            result = _verify_with(_phase_config(enabled=True))

        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(binary_depths.count(2), 1)
        self.assertEqual(binary_depths.count(0), 4)
        self.assertEqual(len(binary_depths), 5)
        receipt = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            receipt["status"], "unknown_fallback_phase_cover"
        )
        self.assertFalse(receipt["proof_authority"])
        phase = result.metadata["property_phase_split"]
        self.assertEqual(phase["status"], "all_children_safe")
        self.assertEqual(phase["actual_child_count"], 4)
        self.assertTrue(phase["all_assignments_enumerated"])
        self.assertTrue(
            phase["phase_cover_audit"][
                "all_children_live_projection_valid"
            ]
        )
        self.assertEqual(live_projection_audit.call_count, 4)
        self.assertTrue(
            all(
                child["stats"]["gpu_dual_pc_cbde_status"]
                == "child_diagnostic"
                and child["stats"]["gpu_dual_pc_cbde_cone_rows"] == [3]
                and child["stats"][
                    "gpu_dual_pc_cbde_checked_upper_full"
                ]
                == -0.5
                and child["stats"][
                    "gpu_dual_pc_cbde_support_improvement_tol"
                ]
                == 1.0e-13
                and "gpu_dual_pc_cbde_unlisted_poison"
                not in child["stats"]
                for child in phase["children"]
            )
        )

    def test_one_child_missing_private_safe_capability_blocks_parent(
        self,
    ) -> None:
        child_calls = 0
        lock = threading.Lock()

        def one_stripped_child(hz, *_args, **_kwargs):
            nonlocal child_calls
            if hz.n_bin:
                return "UNKNOWN", None
            child_kwargs = dict(_kwargs)
            child_kwargs["lp_prefilter_fraction"] = 1.0
            child_kwargs["lp_prefilter_max_seconds"] = 2.0
            verdict, witness = hz_objbound_decide(
                hz, *_args, **child_kwargs
            )
            with lock:
                child_calls += 1
                strip = child_calls == 1
            if strip and hasattr(
                hz, "_solver_objbound_safe_token"
            ):
                delattr(hz, "_solver_objbound_safe_token")
            return verdict, witness

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=one_stripped_child,
        ):
            result = _verify_with(_phase_config(enabled=True))

        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        phase = result.metadata["property_phase_split"]
        self.assertFalse(phase["proof_authority"])
        self.assertEqual(
            phase["status"], "focused_rival_unresolved"
        )
        self.assertEqual(child_calls, 4)
        self.assertEqual(
            sum(
                child["safe_contract_valid"]
                for child in phase["children"]
            ),
            3,
        )

    def test_parent_exception_is_receipted_and_falls_back(self) -> None:
        binary_depths = []
        lock = threading.Lock()

        def parent_raises_children_safe(hz, *_args, **_kwargs):
            with lock:
                binary_depths.append(int(hz.n_bin))
            if hz.n_bin:
                raise RuntimeError("controlled parent failure")
            child_kwargs = dict(_kwargs)
            child_kwargs["lp_prefilter_fraction"] = 1.0
            child_kwargs["lp_prefilter_max_seconds"] = 2.0
            return hz_objbound_decide(
                hz, *_args, **child_kwargs
            )

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=parent_raises_children_safe,
        ):
            result = _verify_with(_phase_config(enabled=True))

        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(binary_depths.count(2), 1)
        self.assertEqual(binary_depths.count(0), 4)
        receipt = result.metadata[
            "property_micro_rlt_parent_prefilter"
        ]
        self.assertEqual(
            receipt["status"], "error_fallback_phase_cover"
        )
        self.assertIn("controlled parent failure", receipt["error"])
        self.assertEqual(
            result.metadata["property_phase_split"]["actual_child_count"],
            4,
        )

    def test_phase_future_exception_is_receipted_as_unknown(self) -> None:
        def parent_unknown(hz, *_args, **_kwargs):
            if hz.n_bin:
                return "UNKNOWN", None
            return "SAFE", None

        with patch(
            "act.back_end.solver.solver_hz.hz_objbound_decide",
            side_effect=parent_unknown,
        ), patch(
            "concurrent.futures._base.Future.result",
            side_effect=RuntimeError("controlled future failure"),
        ):
            result = _verify_with(_phase_config(enabled=True))

        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        phase = result.metadata["property_phase_split"]
        self.assertFalse(phase["proof_authority"])
        self.assertEqual(len(phase["children"]), 4)
        self.assertTrue(
            all(
                child["verdict"] == "UNKNOWN"
                and child["safe_contract_valid"] is False
                and "controlled future failure" in child["error"]
                for child in phase["children"]
            )
        )


if __name__ == "__main__":
    unittest.main()
