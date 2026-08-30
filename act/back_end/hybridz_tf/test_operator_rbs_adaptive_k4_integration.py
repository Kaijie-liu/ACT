#!/usr/bin/env python3
"""Integrated toy gates for RBS -> exact reservoir -> real Operator K4.

This suite deliberately joins the mechanisms which have separate component
tests elsewhere.  One property selector call produces a nested five-row
prefix.  Its first four rows are the primary exact schedule; the same-layer
suffix is only an exact-bit reservoir.  RBS proves the first primary stable,
the reservoir restores four live exact bits, and the ordinary raw-VNNLIB K4
pipeline must certify all six pair conflicts before materializing one fresh
cut and issuing its private one-use solver handoff.

The exact network geometry is also checked independently with ``Fraction``.
SciPy MILP and explicit binary enumeration then verify that the cut changes
only the LP relaxation, never the integer optimum.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import json
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_hz import build_operator_hz
from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
    OperatorPhaseCliquePipelineError,
    consume_operator_phase_clique_pipeline_solver_handoff,
    validate_consumed_operator_phase_clique_solver_build,
    verify_operator_phase_clique_pipeline_result,
)
from act.back_end.hybridz_tf.property_residual_targets import (
    plan_from_property_adjoints,
)
from act.back_end.hybridz_tf.test_operator_phase_clique_pipeline import (
    _enumerated_integer_upper,
    _live_assert,
    _property_sha256,
    _run,
    _write_raw,
)
from act.back_end.solver.solver_hz import hz_row_max


_DTYPE = torch.float64
_FOCUSED_OBJECTIVE = np.asarray([-1.0, 1.0, 0.0], dtype=np.float64)
_Q_DIRECTIONS = (
    (Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(-1)),
    (Fraction(-1), Fraction(1)),
    (Fraction(-1), Fraction(-1)),
)


def _layer(layer_id: int, kind: str, width: int, params=None):
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[
            (int(layer_id), row) for row in range(int(width))
        ],
    )


def _dense(layer_id: int, weight, bias):
    weight_tensor = torch.as_tensor(weight, dtype=_DTYPE)
    bias_tensor = torch.as_tensor(bias, dtype=_DTYPE).reshape(-1)
    if (
        weight_tensor.ndim != 2
        or int(weight_tensor.shape[0]) != int(bias_tensor.numel())
    ):
        raise AssertionError("malformed integrated RBS/K4 dense layer")
    return _layer(
        layer_id,
        "DENSE",
        int(weight_tensor.shape[0]),
        {
            "weight": weight_tensor,
            "bias": bias_tensor,
            "in_features": int(weight_tensor.shape[1]),
            "out_features": int(weight_tensor.shape[0]),
        },
    )


def _rbs_k4_net_and_facts(*, reserve_bias: float = -1.5):
    """Build the correlated ADD toy used by every integrated gate.

    The materialized ADD cube forgets that coordinates 0 and 2 are ``x`` and
    ``-x``.  Consequently primary row 0 appears unstable there, while its
    pre-materialization RBS shadow proves the exact constant ``+1/4``.

    Rows 1--4 are the four corner predicates.  With the ordinary bias -3/2,
    their positive phases are pairwise incompatible in ``[-1,1]^2``.  Setting
    only row 4's bias to -3 creates the controlled exhausted-reservoir case.
    """

    lower = torch.tensor([[-1.0, -1.0]], dtype=_DTYPE)
    upper = torch.tensor([[1.0, 1.0]], dtype=_DTYPE)
    layers = [
        _layer(0, "INPUT", 2, {"shape": (1, 2)}),
        _layer(
            1,
            "INPUT_SPEC",
            2,
            {"kind": "BOX", "lb": lower, "ub": upper},
        ),
        _dense(
            2,
            (
                (1.0, 0.0),
                (0.0, 1.0),
                (-1.0, 0.0),
                (0.0, -1.0),
            ),
            (0.0, 0.0, 0.0, 0.0),
        ),
        _dense(
            3,
            np.zeros((4, 2), dtype=np.float64),
            np.zeros(4, dtype=np.float64),
        ),
        _layer(4, "ADD", 4),
        _dense(
            5,
            (
                (1.0, 0.0, 1.0, 0.0),  # p0 = x + (-x) + 1/4
                (1.0, 1.0, 0.0, 0.0),  # q1 =  x + y - 3/2
                (1.0, 0.0, 0.0, 1.0),  # q2 =  x - y - 3/2
                (0.0, 1.0, 1.0, 0.0),  # q3 = -x + y - 3/2
                (0.0, 0.0, 1.0, 1.0),  # q4 = -x - y - 3/2
            ),
            (0.25, -1.5, -1.5, -1.5, float(reserve_bias)),
        ),
        _layer(6, "RELU", 5),
        _dense(
            7,
            (
                (0.0, 0.0, 0.0, 0.0, 0.0),
                (2.0, 1.0, 1.0, 1.0, 1.0),
                (0.0, 0.5, 0.5, 0.5, 0.5),
            ),
            (1.25, 0.0, 0.0),
        ),
        _layer(8, "ASSERT", 3),
    ]
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [1],
        4: [2, 3],
        5: [4],
        6: [5],
        7: [6],
        8: [7],
    }
    succs = {int(layer.id): [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[int(parent)].append(int(child))
    net = SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={int(layer.id): layer for layer in layers},
    )

    facts = {}
    for layer in layers:
        width = len(layer.out_vars)
        if int(layer.id) in {0, 1}:
            fact_lower = lower.clone()
            fact_upper = upper.clone()
        else:
            # Deliberately broad but finite.  The selector sees all five rows as
            # unstable, while Operator-HZ derives every Big-M bound from its
            # live expression rather than trusting these facts.
            fact_lower = torch.full((1, width), -4.0, dtype=_DTYPE)
            fact_upper = torch.full((1, width), 4.0, dtype=_DTYPE)
        facts[int(layer.id)] = Fact(
            Bounds(fact_lower, fact_upper), ConSet()
        )
    return net, facts


def _single_nested_selector_schedule(facts, property_sha256: str):
    """Run exactly one selector call and split its nested prefix afterward."""

    property_adjoints = {
        6: torch.tensor(
            [
                [2.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.5, 0.5, 0.5, 0.5],
            ],
            dtype=_DTYPE,
        )
    }
    plan = plan_from_property_adjoints(
        property_adjoints,
        facts,
        budget=5,
        rival_ids=(0, 1),
        rival_hardness=(2.0, 1.0),
        all_rivals_processed=True,
        property_sha256=property_sha256,
        pool_per_rival=5,
        phase_joint_focus_after_first=True,
    )
    if tuple((target.layer_id, target.row) for target in plan.targets) != (
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
    ):
        raise AssertionError("selector did not produce the controlled nested prefix")

    primary_targets = tuple(plan.targets[:4])
    primary_layers = {int(target.layer_id) for target in primary_targets}
    reserve_targets = tuple(
        (int(target.layer_id), int(target.row))
        for target in plan.targets[4:]
        if int(target.layer_id) in primary_layers
    )
    primary_builder_targets = tuple(
        target.builder_tuple() for target in primary_targets
    )
    schedule_payload = {
        "full_prefix": [
            [int(target.layer_id), int(target.row)]
            for target in plan.targets
        ],
        "primary": [
            [int(target.layer_id), int(target.row)]
            for target in primary_targets
        ],
        "same_layer_reserve": [list(value) for value in reserve_targets],
        "targets_sha256": str(plan.targets_sha256),
    }
    schedule_sha256 = hashlib.sha256(
        json.dumps(
            schedule_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    selector_receipt = dict(plan.receipt)
    selector_receipt.update(
        {
            "candidate_only": True,
            "proof_authority": False,
            "property_sha256": str(plan.property_sha256),
            "adaptive_exact_schedule": schedule_payload,
            "adaptive_exact_schedule_sha256": schedule_sha256,
        }
    )
    return (
        plan,
        primary_builder_targets,
        reserve_targets,
        selector_receipt,
    )


def _build_selected_rbs_k4(*, reserve_bias: float = -1.5):
    live_assert = _live_assert()
    property_sha256 = _property_sha256(live_assert)
    net, facts = _rbs_k4_net_and_facts(reserve_bias=reserve_bias)
    (
        plan,
        primary_targets,
        reserve_targets,
        selector_receipt,
    ) = _single_nested_selector_schedule(facts, property_sha256)
    # Keep the caller-owned list so the no-alias test can mutate it after the
    # synchronous builder has taken its strict builtin snapshot.
    reservoir_source = list(reserve_targets)
    build = build_operator_hz(
        net,
        facts,
        facts,
        exact_budget=4,
        materialize_add=True,
        residual_bound_screen=True,
        residual_targets=primary_targets,
        exact_target_reservoir=reservoir_source,
        export_verified_preactivation_frame=False,
        issue_constructive_nonempty_seal=True,
    )
    return (
        build,
        plan,
        selector_receipt,
        property_sha256,
        reservoir_source,
    )


def _relu_metadata(build):
    return next(
        item
        for item in build.metadata["layers"]
        if int(item["layer_id"]) == 6
    )


def _fraction_geometry_oracle():
    """Return exact six-pair contradictions and the network margin maximum."""

    pair_supports = []
    for left, right in itertools.combinations(_Q_DIRECTIONS, 2):
        summed_x = left[0] + right[0]
        summed_y = left[1] + right[1]
        # q_i >= 0 and q_j >= 0 would require the summed affine part
        # to be at least 3.  Its exact box support is at most 2.
        support = abs(summed_x) + abs(summed_y)
        pair_supports.append(support)

    q_max = Fraction(-10)
    for x_value, y_value in itertools.product(
        (Fraction(-1), Fraction(1)), repeat=2
    ):
        q_values = tuple(
            direction[0] * x_value
            + direction[1] * y_value
            - Fraction(3, 2)
            for direction in _Q_DIRECTIONS
        )
        q_sum = sum(
            (max(Fraction(0), value) for value in q_values),
            Fraction(0),
        )
        q_max = max(q_max, q_sum)

    # The ReLU sum is convex, so its maximum over a box is attained at a box
    # vertex.  p0 is exactly 1/4 by symbolic cancellation.
    p0 = Fraction(1, 4)
    margin = 2 * p0 + q_max - Fraction(5, 4)
    return tuple(pair_supports), q_max, margin


class OperatorRBSAdaptiveK4IntegrationTests(unittest.TestCase):
    def test_nested_selector_rbs_reserve_real_k4_and_private_handoff(self):
        (
            source,
            plan,
            selector_receipt,
            property_sha256,
            reservoir_source,
        ) = _build_selected_rbs_k4()

        self.assertEqual(
            tuple((target.layer_id, target.row) for target in plan.targets),
            ((6, 0), (6, 1), (6, 2), (6, 3), (6, 4)),
        )
        self.assertEqual(plan.receipt["targets_selected"], 5)
        self.assertEqual(
            plan.receipt["selection_policy"],
            "facility_first_then_same_rival_joint",
        )
        self.assertEqual(plan.receipt["joint_focus_rival_id"], 0)
        self.assertEqual(selector_receipt["property_sha256"], property_sha256)

        relu = _relu_metadata(source)
        reservoir = relu["exact_target_reservoir"]
        self.assertEqual(reservoir["rbs_newly_stabilized_primary"], [0])
        self.assertEqual(reservoir["non_rbs_stable_primary_not_replaced"], [])
        self.assertEqual(reservoir["selected_primary_rows"], [1, 2, 3])
        self.assertEqual(reservoir["selected_reserve_rows"], [4])
        self.assertEqual(reservoir["selected_rows"], [1, 2, 3, 4])
        self.assertEqual(
            reservoir["replacement_slots"],
            [{"stabilized_primary_row": 0, "selected_reserve_row": 4}],
        )
        self.assertEqual(reservoir["shortfall"], 0)
        self.assertEqual(relu["exact_index_preview"], [1, 2, 3, 4])
        self.assertEqual(relu["relu_residual_rows"], 0)
        self.assertEqual(source.hz.n_bin, 4)
        self.assertEqual(source.hz.n_cont, 14)
        self.assertEqual(source.hz.n_eq, 0)
        self.assertEqual(source.hz.n_ub, 22)
        self.assertEqual(source.hz.constraint_nnz, 58)
        self.assertIsNone(source.verified_preactivation_frame)

        # Mutating caller scheduling storage after construction cannot change
        # either the selected exact rows or the builder receipt.
        reservoir_source[0] = (6, 0)
        reservoir_source.append((6, 3))
        self.assertEqual(
            _relu_metadata(source)["exact_target_reservoir"]["reserve_rows"],
            [4],
        )
        self.assertEqual(
            _relu_metadata(source)["exact_target_reservoir"]["selected_rows"],
            [1, 2, 3, 4],
        )

        pair_supports, exact_q_max, exact_margin = (
            _fraction_geometry_oracle()
        )
        self.assertEqual(len(pair_supports), 6)
        self.assertTrue(
            all(support < Fraction(3) for support in pair_supports)
        )
        self.assertEqual(exact_q_max, Fraction(1, 2))
        self.assertEqual(exact_margin, Fraction(-1, 4))

        source_digest = sparse_hz_semantic_digest(source.hz)
        with tempfile.TemporaryDirectory() as directory:
            raw_path, raw_sha256 = _write_raw(directory)
            result = _run(
                source,
                raw_path,
                raw_sha256,
                residual_receipt=selector_receipt,
                residual_property_sha256=property_sha256,
            )

        self.assertEqual(
            result.status, "fresh_verified_k4_clique_materialized"
        )
        self.assertTrue(result.materialized)
        self.assertIsNot(result.build, source)
        self.assertEqual(result.receipt["ranked_literal_count"], 4)
        self.assertEqual(result.receipt["pair_count"], 6)
        self.assertEqual(result.receipt["certified_edge_count"], 6)
        self.assertEqual(result.receipt["clique_count"], 1)
        self.assertEqual(result.receipt["cut_row_count"], 1)
        self.assertEqual(result.build.hz.n_ub, source.hz.n_ub + 1)
        self.assertEqual(sparse_hz_semantic_digest(source.hz), source_digest)
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(
                source, result, deadline=time.monotonic() + 10.0
            )
        )

        before_lp = hz_row_max(
            source.hz, _FOCUSED_OBJECTIVE, integer=False, time_limit=5.0
        )
        after_lp = hz_row_max(
            result.build.hz,
            _FOCUSED_OBJECTIVE,
            integer=False,
            time_limit=5.0,
        )
        self.assertIsNotNone(before_lp)
        self.assertIsNotNone(after_lp)
        self.assertAlmostEqual(float(before_lp), 0.25, places=8)
        self.assertAlmostEqual(float(after_lp), -0.25, places=8)
        self.assertGreater(float(before_lp) - float(after_lp), 0.49)

        before_milp = hz_row_max(
            source.hz, _FOCUSED_OBJECTIVE, integer=True, time_limit=5.0
        )
        after_milp = hz_row_max(
            result.build.hz,
            _FOCUSED_OBJECTIVE,
            integer=True,
            time_limit=5.0,
        )
        before_enumerated = _enumerated_integer_upper(
            source.hz, _FOCUSED_OBJECTIVE
        )
        after_enumerated = _enumerated_integer_upper(
            result.build.hz, _FOCUSED_OBJECTIVE
        )
        self.assertIsNotNone(before_milp)
        self.assertIsNotNone(after_milp)
        for measured in (
            before_milp,
            after_milp,
            before_enumerated,
            after_enumerated,
        ):
            self.assertAlmostEqual(float(measured), -0.25, places=8)

        private = consume_operator_phase_clique_pipeline_solver_handoff(
            source, result, deadline=time.monotonic() + 10.0
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                result, private
            )
        )
        private_digest = sparse_hz_semantic_digest(private.hz)
        with self.assertRaises(ValueError):
            result.build.hz.c[0] += 777.0
        self.assertEqual(sparse_hz_semantic_digest(private.hz), private_digest)
        self.assertAlmostEqual(
            float(
                hz_row_max(
                    private.hz,
                    _FOCUSED_OBJECTIVE,
                    integer=True,
                    time_limit=5.0,
                )
            ),
            -0.25,
            places=8,
        )
        with self.assertRaises(OperatorPhaseCliquePipelineError):
            consume_operator_phase_clique_pipeline_solver_handoff(
                source, result, deadline=time.monotonic() + 10.0
            )

    def test_underfill_and_raw_binding_tamper_preserve_source(self):
        (
            valid_source,
            _plan,
            selector_receipt,
            property_sha256,
            _reservoir_source,
        ) = _build_selected_rbs_k4()
        valid_digest = sparse_hz_semantic_digest(valid_source.hz)

        with tempfile.TemporaryDirectory() as directory:
            raw_path, raw_sha256 = _write_raw(directory)
            raw_tamper = _run(
                valid_source,
                raw_path,
                "0" * 64,
                residual_receipt=selector_receipt,
                residual_property_sha256=property_sha256,
            )
        self.assertIs(raw_tamper.build, valid_source)
        self.assertFalse(raw_tamper.materialized)
        self.assertEqual(
            raw_tamper.receipt["failed_stage"], "raw_top1_issue_consume"
        )
        self.assertEqual(
            sparse_hz_semantic_digest(valid_source.hz), valid_digest
        )

        (
            underfilled,
            _underfill_plan,
            underfill_receipt,
            underfill_property_sha256,
            _underfill_source,
        ) = _build_selected_rbs_k4(reserve_bias=-3.0)
        underfill_meta = _relu_metadata(underfilled)[
            "exact_target_reservoir"
        ]
        self.assertEqual(underfilled.hz.n_bin, 3)
        self.assertEqual(underfill_meta["selected_reserve_rows"], [])
        self.assertEqual(underfill_meta["shortfall"], 1)
        underfill_digest = sparse_hz_semantic_digest(underfilled.hz)

        with tempfile.TemporaryDirectory() as directory:
            raw_path, raw_sha256 = _write_raw(directory)
            stopped = _run(
                underfilled,
                raw_path,
                raw_sha256,
                residual_receipt=underfill_receipt,
                residual_property_sha256=underfill_property_sha256,
            )
        self.assertIs(stopped.build, underfilled)
        self.assertFalse(stopped.materialized)
        self.assertEqual(stopped.receipt.get("cut_row_count", 0), 0)
        self.assertEqual(
            sparse_hz_semantic_digest(underfilled.hz), underfill_digest
        )
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(
                underfilled, stopped, deadline=time.monotonic() + 10.0
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
