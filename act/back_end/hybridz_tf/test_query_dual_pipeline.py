#!/usr/bin/env python3
"""Transactional and red-team tests for query-dual authority."""

from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace
import time
import unittest
from unittest import mock

import numpy as np
import torch

from act.back_end.hybridz_tf.query_dual_candidates import (
    _canonical_sha256 as _candidate_receipt_digest,
    generate_query_dual_candidates,
)
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
)
from act.back_end.hybridz_tf.query_dual_pipeline import (
    QueryDualPipelineError,
    QueryDualPipelineTimeout,
    _candidate_bounds_on_device,
    _replay_descriptor,
    _receipt,
    build_verified_query_dual_feedback,
    validate_verified_query_dual_feedback,
)
from act.util.device_manager import initialize_device


def _layer(layer_id, kind, width, params=None):
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=list(
            range(int(layer_id) * 100, int(layer_id) * 100 + int(width))
        ),
        cache={},
    )


def _pipeline_net():
    f64 = np.float64
    layers = [
        _layer(0, "INPUT", 2, {"shape": (1, 2), "dtype": "torch.float64"}),
        _layer(
            1,
            "INPUT_SPEC",
            2,
            {
                "kind": "BOX",
                "lb": torch.tensor([[-1.0, -0.75]], dtype=torch.float64),
                "ub": torch.tensor([[1.0, 1.25]], dtype=torch.float64),
            },
        ),
        _layer(
            2,
            "DENSE",
            2,
            {
                "weight": np.asarray([[2.0, 0.25], [-0.5, -1.5]], dtype=f64),
                "bias": np.asarray([-0.5, 0.25], dtype=f64),
                "in_features": 2,
                "out_features": 2,
            },
        ),
        _layer(3, "RELU", 2),
        _layer(
            4,
            "DENSE",
            2,
            {
                "weight": np.asarray([[-3.0, 0.5], [0.25, 2.0]], dtype=f64),
                "bias": np.asarray([1.0, -0.2], dtype=f64),
                "in_features": 2,
                "out_features": 2,
            },
        ),
        _layer(5, "RELU", 2),
        _layer(
            6,
            "DENSE",
            2,
            {
                "weight": np.asarray([[0.5, -0.25], [0.2, 0.75]], dtype=f64),
                "bias": np.asarray([-0.1, 0.05], dtype=f64),
                "in_features": 2,
                "out_features": 2,
            },
        ),
        _layer(7, "RELU", 2),
        _layer(
            8,
            "DENSE",
            2,
            {
                "weight": np.asarray([[1.25, -0.2], [-0.4, 0.8]], dtype=f64),
                "bias": np.asarray([0.2, -0.15], dtype=f64),
                "in_features": 2,
                "out_features": 2,
            },
        ),
        _layer(9, "ASSERT", 2, {"kind": "AUDIT"}),
    ]
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [2],
        4: [3],
        5: [4],
        6: [5],
        7: [6],
        8: [7],
        9: [8],
    }
    succs = {int(layer.id): [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    return SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={int(layer.id): layer for layer in layers},
    )


def _direct_relu_predecessor_net():
    layers = [
        _layer(0, "INPUT", 1, {"shape": (1, 1), "dtype": "torch.float64"}),
        _layer(
            1,
            "INPUT_SPEC",
            1,
            {
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=torch.float64),
                "ub": torch.tensor([[1.0]], dtype=torch.float64),
            },
        ),
        _layer(
            2,
            "DENSE",
            1,
            {
                "weight": torch.tensor([[2.0]], dtype=torch.float64),
                "bias": torch.tensor([-0.5], dtype=torch.float64),
                "in_features": 1,
                "out_features": 1,
            },
        ),
        _layer(3, "RELU", 1),
        _layer(4, "RELU", 1),
        _layer(
            5,
            "DENSE",
            1,
            {
                "weight": torch.tensor([[0.75]], dtype=torch.float64),
                "bias": torch.tensor([0.1], dtype=torch.float64),
                "in_features": 1,
                "out_features": 1,
            },
        ),
        _layer(6, "ASSERT", 1, {"kind": "AUDIT"}),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5]}
    succs = {int(layer.id): [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    return SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={int(layer.id): layer for layer in layers},
    )


class _IntervalCandidateSolver:
    """Proofless proposal mock; all authority still comes from real replay."""

    def compute_certified_bound(
        self,
        net,
        bounds_dict,
        c,
        *,
        M,
        optimize=False,
        alpha=None,
        start_lid=None,
        **kwargs,
    ):
        del kwargs, alpha
        if optimize:
            if start_lid is None:
                output = next(
                    int(net.preds[layer.id][0])
                    for layer in net.layers
                    if str(layer.kind).upper() == "ASSERT"
                )
            else:
                output = int(start_lid)
            ancestors = {output}
            stack = [output]
            while stack:
                for parent in net.preds[stack.pop()]:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        stack.append(parent)
            relus = [
                int(layer.id)
                for layer in net.layers
                if str(layer.kind).upper() == "RELU"
                and int(layer.id) in ancestors
            ]
            alpha_state = {
                lid: torch.full(
                    (1, int(M), int(bounds_dict[lid].lb.numel())),
                    0.5,
                    device=c.device,
                    dtype=c.dtype,
                )
                for lid in relus
            }
            return SimpleNamespace(
                margins=torch.zeros(int(M), device=c.device, dtype=c.dtype),
                alpha_state=alpha_state,
            )

        if start_lid is None:
            source_lid = next(
                int(net.preds[layer.id][0])
                for layer in net.layers
                if str(layer.kind).upper() == "ASSERT"
            )
        else:
            source_lid = int(start_lid)
        rows = c.detach().to(device="cpu", dtype=torch.float64).numpy()
        lower = (
            bounds_dict[source_lid]
            .lb.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )
        upper = (
            bounds_dict[source_lid]
            .ub.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )
        margins = (
            np.maximum(rows, 0.0) @ lower
            + np.minimum(rows, 0.0) @ upper
            + 0.01
        )
        return SimpleNamespace(
            margins=torch.as_tensor(margins, device=c.device, dtype=c.dtype),
            alpha_state=None,
        )


class _NestedAlphaCandidateSolver(_IntervalCandidateSolver):
    def compute_certified_bound(self, *args, optimize=False, **kwargs):
        result = super().compute_certified_bound(
            *args, optimize=optimize, **kwargs
        )
        if optimize:
            result.alpha_state = {
                lid: {"nested": value}
                for lid, value in result.alpha_state.items()
            }
        return result


class _DelayedCandidateSolver(_IntervalCandidateSolver):
    def compute_certified_bound(self, *args, optimize=False, **kwargs):
        if optimize:
            time.sleep(0.25)
        return super().compute_certified_bound(
            *args, optimize=optimize, **kwargs
        )


class _MissingAlphaCandidateSolver(_IntervalCandidateSolver):
    def compute_certified_bound(self, *args, optimize=False, **kwargs):
        result = super().compute_certified_bound(
            *args, optimize=optimize, **kwargs
        )
        if optimize:
            result.alpha_state = None
        return result


def build_live_feedback_toy(*, thresholds=None, solver_factory=None):
    """Reusable tiny live bundle for Operator-HZ integration tests."""

    net = _pipeline_net()
    rows = np.asarray([[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64)
    if thresholds is None:
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)
    bundle = build_verified_query_dual_feedback(
        net,
        rows,
        thresholds,
        target_relu_ids=(5, 7),
        steps=2,
        block_size=1,
        replay_chunk_size=2,
        candidate_device="cpu",
        solver_factory=solver_factory or _IntervalCandidateSolver,
    )
    return net, rows, thresholds, bundle


def _valid(net, rows, thresholds, bundle):
    return validate_verified_query_dual_feedback(
        bundle,
        net=net,
        property_rows=rows,
        thresholds=thresholds,
        expected_target_relu_ids=(5, 7),
    )


def _rehash_property_chain(bundle):
    """Rehash a deliberately modified property receipt all the way to top."""

    candidate = bundle.property_stage.candidate_receipt
    candidate_body = dict(candidate)
    candidate.clear()
    candidate.update(_receipt(candidate_body))

    stage = bundle.property_stage.receipt
    stage_body = dict(stage)
    stage_body["candidate_receipt_sha256"] = candidate["receipt_sha256"]
    stage.clear()
    stage.update(_receipt(stage_body))

    top = bundle.receipt
    top_body = dict(top)
    top_body["property_candidate_receipt_sha256"] = candidate[
        "receipt_sha256"
    ]
    top_body["property_receipt_sha256"] = stage["receipt_sha256"]
    top.clear()
    top.update(_receipt(top_body))


def _rehash_target_chain(bundle, index=0):
    """Rehash a deliberately modified target receipt all the way to top."""

    stage_object = bundle.stages[index]
    candidate = stage_object.candidate_receipt
    candidate_body = dict(candidate)
    candidate.clear()
    candidate.update(_receipt(candidate_body))

    stage = stage_object.receipt
    stage_body = dict(stage)
    stage_body["candidate_receipt_sha256"] = candidate["receipt_sha256"]
    stage.clear()
    stage.update(_receipt(stage_body))

    top = bundle.receipt
    top_body = dict(top)
    candidate_hashes = list(top_body["target_candidate_receipt_sha256"])
    candidate_hashes[index] = candidate["receipt_sha256"]
    top_body["target_candidate_receipt_sha256"] = candidate_hashes
    stage_hashes = list(top_body["stage_receipt_sha256"])
    stage_hashes[index] = stage["receipt_sha256"]
    top_body["stage_receipt_sha256"] = stage_hashes
    top.clear()
    top.update(_receipt(top_body))


class QueryDualPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prior_device = torch.get_default_device()
        cls._prior_dtype = torch.get_default_dtype()
        initialize_device(device="cpu", dtype="float64")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._prior_dtype)
        torch.set_default_device(cls._prior_device)

    def test_real_root_and_replay_blocking_sync_and_threshold_sign(self):
        net, rows, thresholds, bundle = build_live_feedback_toy()
        self.assertTrue(_valid(net, rows, thresholds, bundle))
        self.assertEqual(
            bundle.receipt["schema"],
            "act.verified_query_dual_feedback.v2",
        )
        self.assertEqual(
            bundle.receipt["candidate_schema"],
            "act.query_dual_candidates.v2",
        )
        self.assertEqual(
            bundle.receipt["candidate_protocol"],
            "descriptor_only_v2",
        )
        self.assertTrue(
            all(
                stage.receipt["schema"]
                == "act.verified_query_dual_stage.v2"
                and stage.receipt["candidate_schema"]
                == "act.query_dual_candidates.v2"
                and stage.receipt["candidate_protocol"]
                == "descriptor_only_v2"
                for stage in bundle.stages
            )
        )
        self.assertEqual(
            bundle.property_stage.receipt["schema"],
            "act.verified_query_dual_property.v2",
        )
        self.assertEqual(bundle.target_relu_ids, (5, 7))
        self.assertEqual([stage.target_relu_lid for stage in bundle.stages], [5, 7])
        self.assertGreaterEqual(len(bundle.stages[0].blocks), 2)
        self.assertGreaterEqual(len(bundle.property_stage.blocks), 2)
        for target in (5, 7):
            predecessor = int(net.preds[target][0])
            np.testing.assert_array_equal(
                bundle.certified_bounds[target].lb.numpy(),
                bundle.certified_bounds[predecessor].lb.numpy(),
            )
            np.testing.assert_array_equal(
                bundle.certified_bounds[target].ub.numpy(),
                bundle.certified_bounds[predecessor].ub.numpy(),
            )

        zero_net, zero_rows, zero_threshold, zero_bundle = (
            build_live_feedback_toy(thresholds=[0.0, 0.0])
        )
        self.assertTrue(
            _valid(zero_net, zero_rows, zero_threshold, zero_bundle)
        )
        # upper(Cy-t) = upper(Cy)-t.  These inequalities tolerate only the
        # final outward nextafter guard and catch either threshold sign error.
        self.assertGreater(
            bundle.property_upper[0],
            zero_bundle.property_upper[0] + 0.199999999,
        )
        self.assertLess(
            bundle.property_upper[1],
            zero_bundle.property_upper[1] - 0.349999999,
        )

    def test_v1_v2_same_alpha_replay_is_bit_identical(self):
        net = _pipeline_net()
        parent = certify_query_dual_boxes(net).bounds
        common = {
            "net": net,
            "bounds_dict": _candidate_bounds_on_device(
                parent, torch.device("cpu")
            ),
            "target_relu_lid": 5,
            "steps": 2,
            "block_size": 1024,
            "solver_factory": _IntervalCandidateSolver,
        }
        v1 = generate_query_dual_candidates(**common)
        v2 = generate_query_dual_candidates(
            **common,
            descriptor_only=True,
        )
        self.assertEqual(v1.status, "generated")
        self.assertEqual(v2.status, "descriptors_generated")
        self.assertEqual(len(v1.query_descriptors), 1)
        self.assertEqual(len(v2.query_descriptors), 1)
        self.assertEqual(
            v1.query_descriptors[0].alpha_sha256,
            v2.query_descriptors[0].alpha_sha256,
        )
        descriptor_v1 = v1.query_descriptors[0]
        descriptor_v2 = v2.query_descriptors[0]
        replay_kwargs = {
            "net": net,
            "parent_bounds": parent,
            "query_bias": np.zeros(descriptor_v1.M, dtype=np.float64),
            "expected_objectives": descriptor_v1.objectives,
            "expected_kind": "relu_unstable_plus_minus_one_hot",
            "expected_target": 5,
            "expected_start": 4,
            "expected_rows": tuple(descriptor_v1.row_ids),
            "chunk_size": 16,
            "max_workspace_bytes": 64 * 1024 * 1024,
            "deadline": None,
        }
        replay_v1 = _replay_descriptor(
            descriptor=descriptor_v1,
            alpha_tree=v1.alpha_trees[0],
            **replay_kwargs,
        )
        replay_v2 = _replay_descriptor(
            descriptor=descriptor_v2,
            alpha_tree=v2.alpha_trees[0],
            **replay_kwargs,
        )
        np.testing.assert_array_equal(
            replay_v1.lower_bounds,
            replay_v2.lower_bounds,
        )

    def test_cpu_replay_zero_improvement_has_blocks_but_no_improvement_status(
        self,
    ):
        net = _pipeline_net()
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        bundle = build_verified_query_dual_feedback(
            net,
            rows,
            thresholds,
            target_relu_ids=(7,),
            steps=2,
            block_size=1024,
            candidate_device="cpu",
            solver_factory=_IntervalCandidateSolver,
        )
        stage = bundle.stages[0]
        self.assertTrue(stage.blocks)
        self.assertEqual(stage.strict_improvements, 0)
        self.assertEqual(stage.status, "verified_no_improvement")
        self.assertEqual(
            stage.receipt["status"], "verified_no_improvement"
        )
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(7,),
            )
        )

    def test_live_tamper_matrix_and_restoration(self):
        net, rows, thresholds, bundle = build_live_feedback_toy()
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        target = bundle.target_relu_ids[-1]
        tensor = bundle.certified_bounds[target].lb
        old_tensor = tensor.clone()
        tensor.reshape(-1)[0] += 0.125
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        tensor.copy_(old_tensor)
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        record = bundle.stages[0].candidate_receipt["descriptor_records"][0]
        old_objective = record["objective_sha256"]
        record["objective_sha256"] = "0" * 64
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        record["objective_sha256"] = old_objective
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        old_alpha = record["alpha_sha256"]
        record["alpha_sha256"] = "1" * 64
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        record["alpha_sha256"] = old_alpha
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        stage_receipt = bundle.stages[0].receipt
        old_parent = stage_receipt["parent_boxes_sha256"]
        stage_receipt["parent_boxes_sha256"] = "2" * 64
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        stage_receipt["parent_boxes_sha256"] = old_parent
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        upper = bundle.property_upper
        old_upper = upper.copy()
        upper.setflags(write=True)
        upper[0] += 0.125
        upper.setflags(write=False)
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        upper.setflags(write=True)
        upper[:] = old_upper
        upper.setflags(write=False)
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        old_order = list(bundle.receipt["target_relu_ids"])
        bundle.receipt["target_relu_ids"] = list(reversed(old_order))
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        bundle.receipt["target_relu_ids"] = old_order
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        old_started = bundle.receipt["started_monotonic_hex"]
        bundle.receipt["started_monotonic_hex"] = float(
            float.fromhex(old_started) + 1.0
        ).hex()
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        bundle.receipt["started_monotonic_hex"] = old_started
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        replay_values = bundle.property_stage.blocks[0].lower_bounds
        old_replay = replay_values.copy()
        replay_values.setflags(write=True)
        replay_values[0] += 0.125
        replay_values.setflags(write=False)
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        replay_values.setflags(write=True)
        replay_values[:] = old_replay
        replay_values.setflags(write=False)
        self.assertTrue(_valid(net, rows, thresholds, bundle))

    def test_capability_rejects_copy_and_same_object_rehash(self):
        net, rows, thresholds, bundle = build_live_feedback_toy()
        self.assertTrue(_valid(net, rows, thresholds, bundle))

        copied = replace(bundle)
        self.assertFalse(_valid(net, rows, thresholds, copied))
        self.assertTrue(
            validate_verified_query_dual_feedback(
                copied,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        original_receipt = bundle.receipt
        forged_body = dict(original_receipt)
        forged_body["candidate_solver_factory"] = "forged.but_self_hashed"
        forged_receipt = _receipt(forged_body)
        object.__setattr__(bundle, "receipt", forged_receipt)
        # Content remains mathematically valid, but the registry froze the
        # original transaction SHA and therefore rejects this same-id rebuild.
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )
        self.assertFalse(_valid(net, rows, thresholds, bundle))
        object.__setattr__(bundle, "receipt", original_receipt)
        self.assertTrue(_valid(net, rows, thresholds, bundle))

    def test_receipt_only_property_chain_rejects_fully_rehashed_v2_claims(
        self,
    ):
        candidate_tampers = {
            "all_bounds_replayed_with_stored_alpha": True,
            "property_lower_dual_replayed": True,
            "return_optimized_required": False,
            "refresh_forward": True,
            "bounds_source": "caller_live_bounds",
            "alpha_storage": "gpu_mutable",
            "shared_absolute_deadline": True,
            "property_only": False,
            "caller_bounds_unchanged": False,
            "completed_blocks_discarded": 1,
            "non_authoritative_audit_fields": [],
            "property_upper_only": False,
            "property_rows": 999,
            "target_start_lid": 7,
            "planned_query_blocks": 999,
        }
        for field, value in candidate_tampers.items():
            with self.subTest(candidate_field=field):
                net, rows, thresholds, bundle = build_live_feedback_toy()
                bundle.property_stage.candidate_receipt[field] = value
                _rehash_property_chain(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

        net, rows, thresholds, bundle = build_live_feedback_toy()
        property_candidate = bundle.property_stage.candidate_receipt
        property_candidate["property_baseline_sha256"] = "3" * 64
        property_candidate["candidate_property_bounds_sha256"] = "3" * 64
        _rehash_property_chain(bundle)
        self.assertFalse(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        for field, value in {
            "M": 999,
            "objective_order": "wrong",
        }.items():
            with self.subTest(property_record_field=field):
                net, rows, thresholds, bundle = build_live_feedback_toy()
                candidate = bundle.property_stage.candidate_receipt
                candidate["descriptor_records"][0][field] = value
                records_hash = _candidate_receipt_digest(
                    candidate["descriptor_records"]
                )
                candidate["descriptor_records_sha256"] = records_hash
                candidate["descriptor_coverage_sha256"] = records_hash
                _rehash_property_chain(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

        property_tampers = {
            "status": "verified_no_improvement",
            "proof_authority": False,
            "direction": "LOWER",
            "quantity": "threshold_minus_C_y",
            "objective": "+C",
            "replay_query_bias": "-threshold",
            "upper_reconstruction": "+LB(C_y-threshold)",
            "candidate_bounds_sha256": "0" * 64,
            "property_rows": 999,
            "coverage_complete": False,
        }
        for field, value in property_tampers.items():
            with self.subTest(property_field=field):
                net, rows, thresholds, bundle = build_live_feedback_toy()
                bundle.property_stage.receipt[field] = value
                _rehash_property_chain(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

    def test_receipt_only_target_and_top_reject_fully_rehashed_claims(self):
        target_candidate_tampers = {
            "target_start_lid": 999,
            "property_upper_only": False,
            "unstable_target_neurons": 999,
            "planned_query_blocks": 999,
            "steps_requested": 999,
            "block_size": 999,
            "deadline_monotonic": 0.0,
        }
        for field, value in target_candidate_tampers.items():
            with self.subTest(target_candidate_field=field):
                net, rows, thresholds, bundle = build_live_feedback_toy()
                bundle.stages[0].candidate_receipt[field] = value
                _rehash_target_chain(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

        net, rows, thresholds, bundle = build_live_feedback_toy()
        target_candidate = bundle.stages[0].candidate_receipt
        target_candidate["target_bounds_sha256"] = "4" * 64
        target_candidate["candidate_target_bounds_sha256"] = "4" * 64
        _rehash_target_chain(bundle)
        self.assertFalse(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        for field, value in {
            "M": 999,
            "objective_order": "wrong",
        }.items():
            with self.subTest(target_record_field=field):
                net, rows, thresholds, bundle = build_live_feedback_toy()
                candidate = bundle.stages[0].candidate_receipt
                candidate["descriptor_records"][0][field] = value
                records_hash = _candidate_receipt_digest(
                    candidate["descriptor_records"]
                )
                candidate["descriptor_records_sha256"] = records_hash
                candidate["descriptor_coverage_sha256"] = records_hash
                _rehash_target_chain(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

        target_stage_tampers = {
            "proof_authority": False,
            "stage_index": 999,
            "target_relu_lid": 999,
            "predecessor_lid": 999,
            "predecessor_kind": "RELU",
            "predecessor_synchronised": False,
            "relu_key_semantics": "postactivation",
            "candidate_bounds_sha256": "a" * 64,
            "strict_improvements": 999,
            "commit": "partial",
        }
        for field, value in target_stage_tampers.items():
            with self.subTest(target_stage_field=field):
                net, rows, thresholds, bundle = build_live_feedback_toy()
                bundle.stages[0].receipt[field] = value
                _rehash_target_chain(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

        top_tampers = {
            "authority_source": "optimizer_margins",
            "transaction": "partial",
            "candidate_device_fallback": True,
            "root_certifier": "forged.root",
            "independent_replayer": "forged.replayer",
            "non_authoritative_audit_fields": [],
            "steps": 0,
            "block_size": 0,
            "replay_chunk_size": 17,
            "replay_max_workspace_bytes": 17,
            "conv_channel_chunk": 17,
        }
        for field, value in top_tampers.items():
            with self.subTest(top_field=field):
                net, rows, thresholds, bundle = build_live_feedback_toy()
                top_body = dict(bundle.receipt)
                top_body[field] = value
                bundle.receipt.clear()
                bundle.receipt.update(_receipt(top_body))
                self.assertFalse(
                    validate_verified_query_dual_feedback(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

    def test_fail_closed_deadline_nested_alpha_and_proof_injection(self):
        net = _pipeline_net()
        rows = np.asarray([[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64)
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        with self.assertRaises(QueryDualPipelineTimeout):
            build_verified_query_dual_feedback(
                net,
                rows,
                thresholds,
                target_relu_ids=(5, 7),
                deadline=time.monotonic() - 1.0,
                candidate_device="cpu",
                solver_factory=_IntervalCandidateSolver,
            )
        with self.assertRaisesRegex(QueryDualPipelineError, "INVALID_ALPHA"):
            build_verified_query_dual_feedback(
                _pipeline_net(),
                rows,
                thresholds,
                target_relu_ids=(5, 7),
                steps=2,
                block_size=1,
                candidate_device="cpu",
                solver_factory=_NestedAlphaCandidateSolver,
            )
        parameters = inspect.signature(
            build_verified_query_dual_feedback
        ).parameters
        self.assertNotIn("certifier", parameters)
        self.assertNotIn("replayer", parameters)
        with mock.patch(
            "act.back_end.hybridz_tf.query_dual_pipeline.get_default_device",
            return_value=torch.device("cuda:0"),
        ):
            with self.assertRaisesRegex(
                QueryDualPipelineError, "CANDIDATE_DEVICE_MISMATCH"
            ):
                build_verified_query_dual_feedback(
                    _pipeline_net(),
                    rows,
                    thresholds,
                    target_relu_ids=(5, 7),
                    candidate_device="cpu",
                    solver_factory=_IntervalCandidateSolver,
                )

    def test_v2_candidate_deadline_and_error_keep_distinct_classification(self):
        net = _pipeline_net()
        rows = np.asarray([[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64)
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        with self.assertRaises(QueryDualPipelineTimeout):
            build_verified_query_dual_feedback(
                net,
                rows,
                thresholds,
                target_relu_ids=(5,),
                steps=2,
                block_size=1024,
                candidate_device="cpu",
                timeout_s=0.2,
                solver_factory=_DelayedCandidateSolver,
            )
        with self.assertRaisesRegex(
            QueryDualPipelineError,
            "CANDIDATE_FAILURE.*error_fallback_frozen_bounds",
        ):
            build_verified_query_dual_feedback(
                _pipeline_net(),
                rows,
                thresholds,
                target_relu_ids=(5,),
                steps=2,
                block_size=1024,
                candidate_device="cpu",
                solver_factory=_MissingAlphaCandidateSolver,
            )

    def test_pipeline_rejects_legacy_v1_candidate_before_replay(self):
        net = _pipeline_net()
        rows = np.asarray([[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64)
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)

        def legacy_v1(**kwargs):
            kwargs.pop("descriptor_only")
            return generate_query_dual_candidates(**kwargs)

        with self.assertRaisesRegex(
            QueryDualPipelineError, "INVALID_CANDIDATE"
        ):
            build_verified_query_dual_feedback(
                net,
                rows,
                thresholds,
                target_relu_ids=(5,),
                steps=2,
                block_size=1024,
                candidate_device="cpu",
                solver_factory=_IntervalCandidateSolver,
                candidate_generator=legacy_v1,
            )

    def test_real_dual_solver_cpu_toy(self):
        net = _pipeline_net()
        for layer in net.layers:
            if str(layer.kind).upper() == "DENSE":
                layer.params["weight"] = torch.as_tensor(
                    layer.params["weight"], dtype=torch.float64
                )
                layer.params["bias"] = torch.as_tensor(
                    layer.params["bias"], dtype=torch.float64
                )
        rows = np.asarray([[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64)
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        started = time.monotonic()
        bundle = build_verified_query_dual_feedback(
            net,
            rows,
            thresholds,
            target_relu_ids=(5, 7),
            steps=2,
            block_size=1,
            replay_chunk_size=2,
            candidate_device="cpu",
            timeout_s=10.0,
        )
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertTrue(_valid(net, rows, thresholds, bundle))
        self.assertTrue(all(stage.blocks for stage in bundle.stages))

    def test_relu_predecessor_keeps_its_own_preactivation_semantics(self):
        net = _direct_relu_predecessor_net()
        rows = np.asarray([[1.0]], dtype=np.float64)
        thresholds = np.asarray([-0.25], dtype=np.float64)
        bundle = build_verified_query_dual_feedback(
            net,
            rows,
            thresholds,
            target_relu_ids=(4,),
            steps=2,
            block_size=1,
            candidate_device="cpu",
            solver_factory=_IntervalCandidateSolver,
        )
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(4,),
            )
        )
        root_a = bundle.root_certificate.bounds[3]
        final_a = bundle.certified_bounds[3]
        final_b = bundle.certified_bounds[4]
        np.testing.assert_array_equal(final_a.lb.numpy(), root_a.lb.numpy())
        np.testing.assert_array_equal(final_a.ub.numpy(), root_a.ub.numpy())
        self.assertLess(float(final_a.lb.reshape(-1)[0]), 0.0)
        self.assertGreaterEqual(float(final_b.lb.reshape(-1)[0]), 0.0)

        saved_lower = final_a.lb.clone()
        saved_upper = final_a.ub.clone()
        final_a.lb.copy_(final_b.lb)
        final_a.ub.copy_(final_b.ub)
        self.assertFalse(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(4,),
            )
        )
        final_a.lb.copy_(saved_lower)
        final_a.ub.copy_(saved_upper)

    def test_candidate_parent_mutation_aborts_and_property_rows_are_isolated(self):
        net = _pipeline_net()
        rows = np.asarray([[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64)
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)

        def mutate_candidate_bounds(**kwargs):
            result = generate_query_dual_candidates(**kwargs)
            first = sorted(kwargs["bounds_dict"])[0]
            kwargs["bounds_dict"][first].lb.reshape(-1)[0] += 0.5
            return result

        # Candidate bounds are now a private device copy.  Their mutation
        # cannot alter the CPU authority parent or the final replay snapshot.
        safe = build_verified_query_dual_feedback(
            net,
            rows,
            thresholds,
            target_relu_ids=(5, 7),
            steps=2,
            block_size=1,
            candidate_device="cpu",
            solver_factory=_IntervalCandidateSolver,
            candidate_generator=mutate_candidate_bounds,
        )
        self.assertTrue(_valid(net, rows, thresholds, safe))

        original_rows = rows.copy()

        def mutate_property_copy(**kwargs):
            result = generate_query_dual_candidates(**kwargs)
            value = kwargs.get("property_rows")
            if value is not None:
                value.reshape(-1)[0] += 123.0
            return result

        isolated_net = _pipeline_net()
        isolated = build_verified_query_dual_feedback(
            isolated_net,
            rows,
            thresholds,
            target_relu_ids=(5, 7),
            steps=2,
            block_size=1,
            candidate_device="cpu",
            solver_factory=_IntervalCandidateSolver,
            candidate_generator=mutate_property_copy,
        )
        np.testing.assert_array_equal(rows, original_rows)
        self.assertTrue(
            validate_verified_query_dual_feedback(
                isolated,
                net=isolated_net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
            )
        )


if __name__ == "__main__":
    unittest.main()
