#!/usr/bin/env python3
# ===- test_query_dual_candidates.py - frozen-alpha query toys -----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Candidate-only tests for blocked DualSolver alpha queries."""

from __future__ import annotations

import copy
from types import SimpleNamespace
import time
import unittest

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import query_dual_candidates as qdc
from act.back_end.hybridz_tf.query_dual_candidates import (
    generate_query_dual_candidates,
    validate_query_dual_candidates,
    verify_query_dual_candidates_receipt,
)


def _toy_net():
    layers = [
        SimpleNamespace(id=0, kind="INPUT"),
        SimpleNamespace(id=1, kind="DENSE"),
        SimpleNamespace(id=2, kind="RELU"),
        SimpleNamespace(id=3, kind="DENSE"),
        SimpleNamespace(id=4, kind="ASSERT"),
    ]
    return SimpleNamespace(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3]},
    )


def _toy_bounds(*, dtype=torch.float64):
    target_lb = torch.tensor(
        [[-1.0, 0.2, -2.0]], dtype=dtype, device="cpu"
    )
    target_ub = torch.tensor(
        [[1.0, 0.5, 2.0]], dtype=dtype, device="cpu"
    )
    return {
        0: Bounds(
            torch.tensor([[-1.0]], dtype=dtype, device="cpu"),
            torch.tensor([[1.0]], dtype=dtype, device="cpu"),
        ),
        1: Bounds(target_lb.clone(), target_ub.clone()),
        2: Bounds(target_lb.clone(), target_ub.clone()),
        3: Bounds(
            torch.tensor(
                [[-1.0, -2.0]], dtype=dtype, device="cpu"
            ),
            torch.tensor(
                [[1.0, 2.0]], dtype=dtype, device="cpu"
            ),
        ),
    }


def _snapshot(bounds):
    return {
        lid: (value.lb.detach().clone(), value.ub.detach().clone())
        for lid, value in bounds.items()
    }


def _assert_unchanged(test, bounds, snapshot):
    test.assertEqual(set(bounds), set(snapshot))
    for lid, value in bounds.items():
        test.assertTrue(
            torch.allclose(
                value.lb,
                snapshot[lid][0],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )
        test.assertTrue(
            torch.allclose(
                value.ub,
                snapshot[lid][1],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )


def _first_alpha_leaf(tree):
    if isinstance(tree, torch.Tensor):
        return tree
    if isinstance(tree, dict):
        for key in sorted(tree, key=str):
            leaf = _first_alpha_leaf(tree[key])
            if leaf is not None:
                return leaf
    if isinstance(tree, (list, tuple)):
        for value in tree:
            leaf = _first_alpha_leaf(value)
            if leaf is not None:
                return leaf
    return None


class _FakeDualSolver:
    """DualSolver-shaped mock whose replay margins depend on returned alpha."""

    def __init__(
        self,
        *,
        mode="improve",
        optimize_delay=0.0,
        target_lb=(-1.0, 0.2, -2.0),
        target_ub=(1.0, 0.5, 2.0),
        output_lb=(-1.0, -2.0),
        output_ub=(1.0, 2.0),
    ):
        self.mode = mode
        self.optimize_delay = float(optimize_delay)
        self.target_lb = np.asarray(target_lb, dtype=np.float64)
        self.target_ub = np.asarray(target_ub, dtype=np.float64)
        self.output_lb = np.asarray(output_lb, dtype=np.float64)
        self.output_ub = np.asarray(output_ub, dtype=np.float64)
        self.calls = []
        self.returned_alpha_values = []
        self.replayed_alpha_values = []

    def compute_certified_bound(
        self,
        net,
        bounds_dict,
        c,
        *,
        M,
        optimize=False,
        return_optimized=False,
        refresh_forward=True,
        alpha=None,
        start_lid=None,
        **kwargs,
    ):
        del net, bounds_dict, kwargs
        self.calls.append(
            {
                "optimize": bool(optimize),
                "return_optimized": bool(return_optimized),
                "refresh_forward": bool(refresh_forward),
                "alpha_present": alpha is not None,
                "start_lid": start_lid,
                "M": int(M),
                "objective": c.detach().cpu().to(torch.float64).clone(),
            }
        )
        if optimize:
            if self.optimize_delay:
                time.sleep(self.optimize_delay)
            value = 0.25 + 0.10 * len(self.returned_alpha_values)
            main = torch.full(
                (1, int(M), 3),
                value,
                dtype=torch.float32,
                device=c.device,
            )
            alpha_state = {
                7: {
                    "main": main,
                    "nested": [
                        torch.tensor(value / 2, device=c.device),
                        (torch.tensor(value / 4, device=c.device),),
                    ],
                }
            }
            if self.mode == "nonfinite_alpha":
                alpha_state[7]["main"][0, 0, 0] = float("nan")
            self.returned_alpha_values.append(value)
            # Deliberately unrelated to the alpha state.  Consuming these as
            # candidate bounds would make the binding tests fail loudly.
            margins = torch.full(
                (int(M),), 42.0, dtype=c.dtype, device=c.device
            )
            if self.mode == "nonfinite_optimizer_margin":
                margins[0] = float("nan")
            result = {
                "alpha_state": alpha_state,
                "eta_state": None,
            }
            if self.mode != "missing_optimizer_margin":
                result["margins"] = margins
            return SimpleNamespace(**result)

        if alpha is None:
            raise AssertionError("frozen replay omitted alpha")
        leaf = _first_alpha_leaf(alpha)
        if leaf is None:
            raise AssertionError("frozen replay alpha tree has no leaf")
        value = float(leaf.detach().reshape(-1)[0].cpu())
        self.replayed_alpha_values.append(value)
        objectives = c.detach().cpu().to(torch.float64).numpy()
        delta = 0.0 if self.mode == "no_improvement" else value
        if start_lid is not None:
            lower = []
            for row in objectives:
                neuron = int(np.flatnonzero(np.abs(row) > 0.0)[0])
                coefficient = float(row[neuron])
                endpoint = (
                    self.target_lb[neuron]
                    if coefficient > 0.0
                    else self.target_ub[neuron]
                )
                lower.append(coefficient * endpoint + delta)
        else:
            positive = np.maximum(objectives, 0.0)
            negative = np.minimum(objectives, 0.0)
            lower = (
                positive @ self.output_lb + negative @ self.output_ub
            )
            lower = lower + delta
        replay = np.asarray(lower, dtype=np.float64)
        if self.mode == "nonfinite_replay":
            replay[0] = np.nan
        return SimpleNamespace(
            margins=torch.as_tensor(
                replay, dtype=c.dtype, device=c.device
            ),
            alpha_state=None,
            eta_state=None,
        )


class QueryDualCandidateBindingTests(unittest.TestCase):
    def test_v3_constructs_only_selected_rows_and_keeps_property_full(self):
        source = _toy_bounds()
        snapshot = _snapshot(source)
        property_rows = np.asarray(
            [[1.0, 1.0], [-1.0, 0.5]], dtype=np.float64
        )
        solver = _FakeDualSolver(mode="nonfinite_optimizer_margin")
        result = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=source,
            target_relu_lid=2,
            property_rows=property_rows,
            property_upper_only=True,
            steps=8,
            block_size=1,
            descriptor_only=True,
            selected_target_rows=(2,),
            solver_factory=lambda: solver,
        )

        self.assertEqual(result.status, "descriptors_generated")
        self.assertEqual(
            result.receipt["schema"], "act.query_dual_candidates.v3"
        )
        self.assertEqual(
            result.receipt["protocol"],
            "property_sparse_descriptor_only_v3",
        )
        self.assertTrue(validate_query_dual_candidates(result))
        self.assertEqual(result.receipt["eligible_target_row_ids"], [0, 2])
        self.assertEqual(result.receipt["selected_target_row_ids"], [2])
        self.assertEqual(result.receipt["omitted_target_row_ids"], [0])
        self.assertTrue(result.receipt["target_partition_complete"])
        self.assertTrue(result.receipt["target_partition_disjoint"])
        self.assertTrue(result.receipt["selected_coverage_complete"])
        self.assertTrue(result.receipt["property_coverage_complete"])
        self.assertTrue(
            result.receipt["unselected_bounds_bit_identical_parent"]
        )
        self.assertIs(result.receipt["optimizer_margins_exported"], False)
        self.assertIs(result.receipt["gpu_frozen_alpha_replay"], False)

        target_descriptors = [
            descriptor
            for descriptor in result.query_descriptors
            if descriptor.query_kind
            == "relu_unstable_plus_minus_one_hot"
        ]
        self.assertEqual(len(target_descriptors), 1)
        self.assertEqual(target_descriptors[0].row_ids, (2,))
        np.testing.assert_array_equal(
            target_descriptors[0].objectives,
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        )
        property_descriptors = [
            descriptor
            for descriptor in result.query_descriptors
            if descriptor.query_kind
            == "final_property_negative_c_upper_only"
        ]
        self.assertEqual(
            [descriptor.row_ids for descriptor in property_descriptors],
            [(0,), (1,)],
        )
        np.testing.assert_array_equal(
            np.vstack(
                [descriptor.objectives for descriptor in property_descriptors]
            ),
            -property_rows,
        )
        # One selected target block plus every property row; no target row 0
        # objective was ever constructed or sent to the optimizer.
        self.assertEqual(len(solver.calls), 3)
        self.assertTrue(all(call["optimize"] for call in solver.calls))
        self.assertTrue(
            all(not call["alpha_present"] for call in solver.calls)
        )
        self.assertEqual([call["M"] for call in solver.calls], [2, 1, 1])
        np.testing.assert_array_equal(
            result.target_bounds.lb.numpy(), source[2].lb.numpy()
        )
        np.testing.assert_array_equal(
            result.target_bounds.ub.numpy(), source[2].ub.numpy()
        )
        _assert_unchanged(self, source, snapshot)

    def test_v3_rejects_duplicate_omit_reorder_and_shape_tampering(self):
        result = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=_toy_bounds(),
            target_relu_lid=2,
            steps=8,
            block_size=1,
            descriptor_only=True,
            selected_target_rows=(2, 0),
            solver_factory=lambda: _FakeDualSolver(),
        )
        self.assertTrue(validate_query_dual_candidates(result))
        self.assertEqual(
            [
                descriptor.row_ids
                for descriptor in result.query_descriptors
            ],
            [(2,), (0,)],
        )
        lower = result.target_bounds.lb.numpy().reshape(-1)
        upper = result.target_bounds.ub.numpy().reshape(-1)

        def bind_receipt(candidate, updates):
            payload = dict(candidate.receipt)
            payload.pop("receipt_sha256", None)
            payload.update(updates)
            rebound = qdc._receipt_with_sha256(payload)
            candidate.receipt.clear()
            candidate.receipt.update(rebound)

        duplicate = copy.deepcopy(result)
        bind_receipt(
            duplicate,
            {
                "selected_target_row_ids": [2, 2],
                "selected_target_rows_sha256":
                    qdc._ordered_rows_sha256((2, 2)),
                "selected_target_count": 2,
                "omitted_target_row_ids": [0],
                "omitted_target_rows_sha256":
                    qdc._ordered_rows_sha256((0,)),
                "omitted_target_count": 1,
                "selected_parent_target_bounds_sha256":
                    qdc._indexed_bounds_sha256(
                        lower, upper, (2, 2)
                    ),
                "unselected_parent_target_bounds_sha256":
                    qdc._indexed_bounds_sha256(lower, upper, (0,)),
                "unselected_candidate_target_bounds_sha256":
                    qdc._indexed_bounds_sha256(lower, upper, (0,)),
            },
        )
        self.assertTrue(
            verify_query_dual_candidates_receipt(duplicate.receipt)
        )
        self.assertFalse(validate_query_dual_candidates(duplicate))

        omitted = copy.deepcopy(result)
        bind_receipt(
            omitted,
            {
                "selected_target_row_ids": [2],
                "selected_target_rows_sha256":
                    qdc._ordered_rows_sha256((2,)),
                "selected_target_count": 1,
                "omitted_target_row_ids": [0],
                "omitted_target_rows_sha256":
                    qdc._ordered_rows_sha256((0,)),
                "omitted_target_count": 1,
                "selected_parent_target_bounds_sha256":
                    qdc._indexed_bounds_sha256(lower, upper, (2,)),
                "unselected_parent_target_bounds_sha256":
                    qdc._indexed_bounds_sha256(lower, upper, (0,)),
                "unselected_candidate_target_bounds_sha256":
                    qdc._indexed_bounds_sha256(lower, upper, (0,)),
            },
        )
        self.assertTrue(
            verify_query_dual_candidates_receipt(omitted.receipt)
        )
        self.assertFalse(validate_query_dual_candidates(omitted))

        reordered = copy.deepcopy(result)
        bind_receipt(
            reordered,
            {
                "selected_target_row_ids": [0, 2],
                "selected_target_rows_sha256":
                    qdc._ordered_rows_sha256((0, 2)),
                "selected_parent_target_bounds_sha256":
                    qdc._indexed_bounds_sha256(
                        lower, upper, (0, 2)
                    ),
            },
        )
        self.assertTrue(
            verify_query_dual_candidates_receipt(reordered.receipt)
        )
        self.assertFalse(validate_query_dual_candidates(reordered))

        malformed_shape = copy.deepcopy(result)
        descriptor = malformed_shape.query_descriptors[0]
        malformed_objective = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        malformed_hash = qdc._array_sha256(malformed_objective)
        object.__setattr__(
            descriptor, "objectives", malformed_objective
        )
        object.__setattr__(
            descriptor, "objective_sha256", malformed_hash
        )
        records = [
            dict(record)
            for record in malformed_shape.receipt[
                "descriptor_records"
            ]
        ]
        records[0]["objective_sha256"] = malformed_hash
        bind_receipt(
            malformed_shape,
            {
                "descriptor_records": records,
                "descriptor_records_sha256":
                    qdc._canonical_sha256(records),
                "descriptor_coverage_sha256":
                    qdc._canonical_sha256(records),
            },
        )
        self.assertTrue(
            verify_query_dual_candidates_receipt(
                malformed_shape.receipt
            )
        )
        self.assertFalse(validate_query_dual_candidates(malformed_shape))

    def test_v3_invalid_selection_fails_before_optimizer(self):
        solver = _FakeDualSolver()
        duplicate = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=_toy_bounds(),
            target_relu_lid=2,
            descriptor_only=True,
            selected_target_rows=(2, 2),
            solver_factory=lambda: solver,
        )
        self.assertEqual(
            duplicate.status, "error_fallback_frozen_bounds"
        )
        self.assertIn("duplicates", duplicate.receipt["error"])
        self.assertEqual(solver.calls, [])
        self.assertTrue(validate_query_dual_candidates(duplicate))

        stable_solver = _FakeDualSolver()
        stable = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=_toy_bounds(),
            target_relu_lid=2,
            descriptor_only=True,
            selected_target_rows=(1,),
            solver_factory=lambda: stable_solver,
        )
        self.assertEqual(stable.status, "error_fallback_frozen_bounds")
        self.assertIn("non-unstable", stable.receipt["error"])
        self.assertEqual(stable_solver.calls, [])
        self.assertTrue(validate_query_dual_candidates(stable))

    def test_v2_is_one_optimize_call_per_block_and_exports_no_margin(self):
        source = _toy_bounds()
        v1_solver = _FakeDualSolver()
        v2_solver = _FakeDualSolver()
        v1 = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1,
            solver_factory=lambda: v1_solver,
        )
        v2 = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1,
            descriptor_only=True,
            solver_factory=lambda: v2_solver,
        )

        self.assertEqual(v1.status, "generated")
        self.assertEqual(v2.status, "descriptors_generated")
        self.assertEqual(v1.receipt["schema"], "act.query_dual_candidates.v1")
        self.assertEqual(v2.receipt["schema"], "act.query_dual_candidates.v2")
        self.assertEqual(len(v1_solver.calls), 4)
        self.assertEqual(len(v2_solver.calls), 2)
        self.assertTrue(all(call["optimize"] for call in v2_solver.calls))
        self.assertTrue(
            all(call["return_optimized"] for call in v2_solver.calls)
        )
        self.assertTrue(validate_query_dual_candidates(v2))
        np.testing.assert_array_equal(
            v2.target_bounds.lb.numpy(), source[2].lb.numpy()
        )
        np.testing.assert_array_equal(
            v2.target_bounds.ub.numpy(), source[2].ub.numpy()
        )
        self.assertEqual(v2.improved_target_indices.size, 0)
        self.assertEqual(v2.improved_property_indices.size, 0)
        self.assertEqual(
            v2.receipt["candidate_bound_source"],
            "none_descriptor_only",
        )
        self.assertIs(v2.receipt["optimizer_margins_exported"], False)
        self.assertIs(v2.receipt["gpu_frozen_alpha_replay"], False)
        self.assertIs(v2.receipt["cpu_independent_replay_required"], True)
        self.assertTrue(
            all(
                "optimizer_margin_sha256" not in record
                and "replay_margin_sha256" not in record
                for record in v2.receipt["descriptor_records"]
            )
        )

    def test_v2_ignores_nan_optimizer_margin_and_rejects_material_tamper(self):
        clean = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=_toy_bounds(),
            target_relu_lid=2,
            steps=8,
            block_size=1,
            descriptor_only=True,
            solver_factory=lambda: _FakeDualSolver(),
        )
        poisoned = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=_toy_bounds(),
            target_relu_lid=2,
            steps=8,
            block_size=1,
            descriptor_only=True,
            solver_factory=lambda: _FakeDualSolver(
                mode="nonfinite_optimizer_margin"
            ),
        )
        self.assertEqual(poisoned.status, "descriptors_generated")
        self.assertTrue(validate_query_dual_candidates(poisoned))
        self.assertEqual(
            clean.receipt["descriptor_records_sha256"],
            poisoned.receipt["descriptor_records_sha256"],
        )
        self.assertEqual(
            clean.receipt["alpha_hashes_sha256"],
            poisoned.receipt["alpha_hashes_sha256"],
        )
        missing = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=_toy_bounds(),
            target_relu_lid=2,
            steps=8,
            block_size=1,
            descriptor_only=True,
            solver_factory=lambda: _FakeDualSolver(
                mode="missing_optimizer_margin"
            ),
        )
        self.assertEqual(missing.status, "descriptors_generated")
        self.assertTrue(validate_query_dual_candidates(missing))
        self.assertEqual(
            clean.receipt["descriptor_records_sha256"],
            missing.receipt["descriptor_records_sha256"],
        )

        objective_tamper = copy.deepcopy(clean)
        objective_tamper.query_descriptors[0].objectives[0, 0] += 1.0
        self.assertFalse(validate_query_dual_candidates(objective_tamper))
        alpha_tamper = copy.deepcopy(clean)
        leaf = _first_alpha_leaf(alpha_tamper.alpha_trees[0])
        leaf.reshape(-1)[0] += 0.125
        self.assertFalse(validate_query_dual_candidates(alpha_tamper))

    def test_complete_object_validator_rejects_in_memory_tampering(self):
        result = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=_toy_bounds(),
            target_relu_lid=2,
            steps=8,
            block_size=1,
            solver_factory=lambda: _FakeDualSolver(),
        )
        self.assertEqual(result.status, "generated")
        self.assertTrue(validate_query_dual_candidates(result))

        tampered_bound = copy.deepcopy(result)
        tampered_bound.target_bounds.lb[0, 0] += 0.125
        self.assertFalse(validate_query_dual_candidates(tampered_bound))

        tampered_objective = copy.deepcopy(result)
        tampered_objective.query_descriptors[0].objectives[0, 0] += 1.0
        self.assertFalse(validate_query_dual_candidates(tampered_objective))

        tampered_alpha = copy.deepcopy(result)
        alpha_leaf = _first_alpha_leaf(tampered_alpha.alpha_trees[0])
        self.assertIsNotNone(alpha_leaf)
        alpha_leaf.reshape(-1)[0] += 0.125
        self.assertFalse(validate_query_dual_candidates(tampered_alpha))

        tampered_indices = copy.deepcopy(result)
        tampered_indices.improved_target_indices[0] = 1
        self.assertFalse(validate_query_dual_candidates(tampered_indices))

        tampered_receipt = copy.deepcopy(result)
        tampered_receipt.receipt["steps_requested"] += 1
        self.assertFalse(validate_query_dual_candidates(tampered_receipt))

    def test_deterministic_blocking_and_frozen_alpha_replay_binding(self):
        net = _toy_net()
        source = _toy_bounds()
        snapshot = _snapshot(source)
        solvers = []

        def factory():
            solver = _FakeDualSolver()
            solvers.append(solver)
            return solver

        first = generate_query_dual_candidates(
            net=net,
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1,
            solver_factory=factory,
        )
        second = generate_query_dual_candidates(
            net=net,
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1,
            solver_factory=factory,
        )
        self.assertEqual(first.status, "generated")
        self.assertEqual(second.status, "generated")
        np.testing.assert_array_equal(
            first.improved_target_indices, [0, 2]
        )
        np.testing.assert_allclose(
            first.target_bounds.lb.numpy(),
            [[-0.75, 0.2, -1.65]],
        )
        np.testing.assert_allclose(
            first.target_bounds.ub.numpy(),
            [[0.75, 0.5, 1.65]],
        )
        self.assertEqual(len(first.query_descriptors), 2)
        self.assertEqual(len(first.alpha_trees), 2)
        for block, descriptor in enumerate(first.query_descriptors):
            self.assertEqual(
                descriptor.query_kind,
                "relu_unstable_plus_minus_one_hot",
            )
            self.assertEqual(descriptor.start_lid, 1)
            self.assertEqual(descriptor.target_relu_lid, 2)
            self.assertEqual(descriptor.M, 2)
            self.assertEqual(descriptor.alpha_tree_index, block)
            np.testing.assert_array_equal(
                descriptor.objectives[1],
                -descriptor.objectives[0],
            )
        for tree in first.alpha_trees:
            for leaf in (
                tree[7]["main"],
                tree[7]["nested"][0],
                tree[7]["nested"][1][0],
            ):
                self.assertEqual(leaf.device.type, "cpu")
                self.assertEqual(leaf.dtype, torch.float64)

        for solver in solvers:
            self.assertEqual(len(solver.calls), 4)
            for offset in range(0, len(solver.calls), 2):
                optimized = solver.calls[offset]
                replayed = solver.calls[offset + 1]
                self.assertTrue(optimized["optimize"])
                self.assertTrue(optimized["return_optimized"])
                self.assertFalse(optimized["refresh_forward"])
                self.assertFalse(optimized["alpha_present"])
                self.assertFalse(replayed["optimize"])
                self.assertFalse(replayed["return_optimized"])
                self.assertFalse(replayed["refresh_forward"])
                self.assertTrue(replayed["alpha_present"])
            np.testing.assert_allclose(
                solver.returned_alpha_values,
                solver.replayed_alpha_values,
                rtol=0.0,
                atol=1.0e-7,
            )
        # Optimizer margins were all 42; candidate values came only from the
        # second call and the alpha values 0.25/0.35.
        self.assertLess(float(first.target_bounds.ub.max()), 42.0)
        self.assertFalse(
            first.receipt["optimizer_best_margins_used_as_bounds"]
        )
        self.assertTrue(
            first.receipt["all_bounds_replayed_with_stored_alpha"]
        )
        self.assertEqual(first.receipt["steps_requested"], 8)
        self.assertEqual(len(first.timings), 2)
        self.assertTrue(
            verify_query_dual_candidates_receipt(first.receipt)
        )

        self.assertEqual(
            first.receipt["input_bounds_sha256"],
            second.receipt["input_bounds_sha256"],
        )
        self.assertEqual(
            first.receipt["descriptor_records_sha256"],
            second.receipt["descriptor_records_sha256"],
        )
        self.assertEqual(
            first.receipt["alpha_hashes_sha256"],
            second.receipt["alpha_hashes_sha256"],
        )
        np.testing.assert_array_equal(
            first.target_bounds.lb.numpy(),
            second.target_bounds.lb.numpy(),
        )
        np.testing.assert_array_equal(
            first.target_bounds.ub.numpy(),
            second.target_bounds.ub.numpy(),
        )
        _assert_unchanged(self, source, snapshot)

    def test_final_property_rows_are_queried_as_c_then_negative_c(self):
        source = _toy_bounds()
        snapshot = _snapshot(source)
        property_rows = np.asarray(
            [[1.0, 1.0], [-1.0, 0.5]], dtype=np.float64
        )
        solvers = []

        def factory():
            solver = _FakeDualSolver()
            solvers.append(solver)
            return solver

        result = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=source,
            target_relu_lid=2,
            property_rows=property_rows,
            property_upper_only=False,
            steps=8,
            block_size=2,
            solver_factory=factory,
        )
        self.assertEqual(result.status, "generated")
        property_descriptors = [
            descriptor
            for descriptor in result.query_descriptors
            if descriptor.query_kind == "final_property_c_minus_c"
        ]
        self.assertEqual(len(property_descriptors), 1)
        descriptor = property_descriptors[0]
        self.assertIsNone(descriptor.start_lid)
        self.assertIsNone(descriptor.target_relu_lid)
        self.assertEqual(descriptor.row_ids, (0, 1))
        self.assertEqual(descriptor.M, 4)
        np.testing.assert_array_equal(
            descriptor.objectives[:2], property_rows
        )
        np.testing.assert_array_equal(
            descriptor.objectives[2:], -property_rows
        )
        # ReLU block uses alpha=.25; property block uses alpha=.35.
        np.testing.assert_allclose(result.property_lower, [-2.65, -1.65])
        np.testing.assert_allclose(result.property_upper, [2.65, 1.65])
        np.testing.assert_array_equal(
            result.improved_property_indices, [0, 1]
        )
        self.assertEqual(
            [call["start_lid"] for call in solvers[0].calls],
            [1, 1, None, None],
        )
        _assert_unchanged(self, source, snapshot)

    def test_property_only_upper_query_after_target_feedback_uses_new_hash(self):
        net = _toy_net()
        source = _toy_bounds()
        source_snapshot = _snapshot(source)

        target_solver = _FakeDualSolver()
        target = generate_query_dual_candidates(
            net=net,
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1024,
            solver_factory=lambda: target_solver,
        )
        self.assertEqual(target.status, "generated")
        self.assertEqual(len(target.query_descriptors), 1)
        self.assertEqual(target.query_descriptors[0].M, 4)

        # Applying a candidate is intentionally a caller-side operation.  The
        # second, property-only stage must read this new frozen snapshot.
        updated = copy.deepcopy(source)
        updated[1] = Bounds(
            target.target_bounds.lb.clone(),
            target.target_bounds.ub.clone(),
        )
        updated[2] = Bounds(
            target.target_bounds.lb.clone(),
            target.target_bounds.ub.clone(),
        )
        updated_snapshot = _snapshot(updated)
        property_rows = np.asarray(
            [[1.0, 1.0], [-1.0, 0.5]], dtype=np.float64
        )
        property_solver = _FakeDualSolver()
        result = generate_query_dual_candidates(
            net=net,
            bounds_dict=updated,
            target_relu_lid=None,
            property_rows=property_rows,
            property_upper_only=True,
            steps=8,
            block_size=1024,
            solver_factory=lambda: property_solver,
        )

        self.assertEqual(result.status, "generated")
        self.assertTrue(result.receipt["property_only"])
        self.assertTrue(result.receipt["property_upper_only"])
        self.assertEqual(result.receipt["block_size"], 1024)
        self.assertEqual(result.receipt["unstable_target_neurons"], 0)
        self.assertEqual(
            result.receipt["property_lower_bound_source"],
            "frozen_interval_baseline_not_dual_replayed",
        )
        self.assertFalse(result.receipt["property_lower_dual_replayed"])
        self.assertFalse(
            result.receipt["all_bounds_replayed_with_stored_alpha"]
        )
        self.assertTrue(
            result.receipt[
                "all_candidate_updates_replayed_with_stored_alpha"
            ]
        )
        self.assertNotEqual(
            target.receipt["input_bounds_sha256"],
            result.receipt["input_bounds_sha256"],
        )
        self.assertEqual(tuple(result.target_bounds.lb.shape), (1, 0))
        self.assertEqual(tuple(result.target_bounds.ub.shape), (1, 0))
        self.assertEqual(len(result.query_descriptors), 1)
        descriptor = result.query_descriptors[0]
        self.assertEqual(
            descriptor.query_kind,
            "final_property_negative_c_upper_only",
        )
        self.assertEqual(
            descriptor.objective_order,
            "negated_rows_only_for_property_upper_bounds",
        )
        self.assertIsNone(descriptor.start_lid)
        self.assertIsNone(descriptor.target_relu_lid)
        self.assertEqual(descriptor.M, 2)
        np.testing.assert_array_equal(
            descriptor.objectives, -property_rows
        )
        np.testing.assert_array_equal(result.property_lower, [-3.0, -2.0])
        np.testing.assert_allclose(result.property_upper, [2.75, 1.75])
        np.testing.assert_array_equal(
            result.improved_property_indices, [0, 1]
        )
        self.assertEqual(len(property_solver.calls), 2)
        self.assertEqual(
            [call["start_lid"] for call in property_solver.calls],
            [None, None],
        )
        self.assertEqual(
            [call["M"] for call in property_solver.calls],
            [2, 2],
        )
        _assert_unchanged(self, source, source_snapshot)
        _assert_unchanged(self, updated, updated_snapshot)


class QueryDualCandidateFallbackTests(unittest.TestCase):
    def test_shared_deadline_discards_optimized_but_unreplayed_block(self):
        source = _toy_bounds()
        snapshot = _snapshot(source)
        solvers = []

        def factory():
            solver = _FakeDualSolver(optimize_delay=0.03)
            solvers.append(solver)
            return solver

        result = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1,
            deadline=time.monotonic() + 0.01,
            solver_factory=factory,
        )
        self.assertEqual(result.status, "deadline_fallback_frozen_bounds")
        self.assertFalse(result.receipt["whole_batch_complete"])
        self.assertEqual(result.query_descriptors, ())
        self.assertEqual(result.alpha_trees, ())
        self.assertEqual(len(solvers[0].calls), 1)
        self.assertTrue(solvers[0].calls[0]["return_optimized"])
        np.testing.assert_array_equal(
            result.target_bounds.lb.numpy(), source[2].lb.numpy()
        )
        np.testing.assert_array_equal(
            result.target_bounds.ub.numpy(), source[2].ub.numpy()
        )
        _assert_unchanged(self, source, snapshot)

        v2_solver = _FakeDualSolver(optimize_delay=0.03)
        v2 = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1,
            deadline=time.monotonic() + 0.01,
            descriptor_only=True,
            solver_factory=lambda: v2_solver,
        )
        self.assertEqual(v2.status, "deadline_fallback_frozen_bounds")
        self.assertFalse(v2.receipt["whole_batch_complete"])
        self.assertFalse(v2.receipt["descriptor_coverage_complete"])
        self.assertEqual(v2.query_descriptors, ())
        self.assertEqual(v2.alpha_trees, ())
        self.assertEqual(len(v2_solver.calls), 1)
        self.assertTrue(v2_solver.calls[0]["optimize"])
        self.assertTrue(validate_query_dual_candidates(v2))

    def test_nonfinite_and_malformed_inputs_fail_closed(self):
        cases = []
        nonfinite_bounds = _toy_bounds()
        nonfinite_bounds[2].lb[0, 0] = float("nan")
        cases.append(
            (
                nonfinite_bounds,
                None,
                lambda: _FakeDualSolver(),
                "non-finite",
            )
        )
        cases.append(
            (
                _toy_bounds(),
                None,
                lambda: _FakeDualSolver(mode="nonfinite_alpha"),
                "non-finite",
            )
        )
        cases.append(
            (
                _toy_bounds(),
                None,
                lambda: _FakeDualSolver(mode="nonfinite_replay"),
                "non-finite",
            )
        )
        cases.append(
            (
                _toy_bounds(),
                np.ones((1, 3), dtype=np.float64),
                lambda: _FakeDualSolver(),
                "width",
            )
        )
        for source, property_rows, factory, error_text in cases:
            with self.subTest(error=error_text, property=property_rows is not None):
                snapshot = _snapshot(source)
                result = generate_query_dual_candidates(
                    net=_toy_net(),
                    bounds_dict=source,
                    target_relu_lid=2,
                    property_rows=property_rows,
                    solver_factory=factory,
                )
                self.assertEqual(
                    result.status, "error_fallback_frozen_bounds"
                )
                self.assertFalse(result.receipt["whole_batch_complete"])
                self.assertIn(error_text, result.receipt["error"])
                self.assertEqual(result.query_descriptors, ())
                self.assertEqual(result.alpha_trees, ())
                _assert_unchanged(self, source, snapshot)

    def test_no_improvement_returns_original_bounds_and_no_candidates(self):
        source = _toy_bounds()
        snapshot = _snapshot(source)
        result = generate_query_dual_candidates(
            net=_toy_net(),
            bounds_dict=source,
            target_relu_lid=2,
            steps=8,
            block_size=1,
            solver_factory=lambda: _FakeDualSolver(
                mode="no_improvement"
            ),
        )
        self.assertEqual(result.status, "no_improvement_fallback")
        self.assertTrue(result.receipt["whole_batch_complete"])
        self.assertFalse(result.receipt["candidate_generated"])
        self.assertEqual(result.receipt["attempted_query_blocks"], 2)
        self.assertEqual(result.receipt["completed_blocks_discarded"], 2)
        self.assertEqual(result.query_descriptors, ())
        self.assertEqual(result.alpha_trees, ())
        np.testing.assert_array_equal(
            result.target_bounds.lb.numpy(), source[2].lb.numpy()
        )
        np.testing.assert_array_equal(
            result.target_bounds.ub.numpy(), source[2].ub.numpy()
        )
        self.assertTrue(
            verify_query_dual_candidates_receipt(result.receipt)
        )
        _assert_unchanged(self, source, snapshot)


if __name__ == "__main__":
    unittest.main()
