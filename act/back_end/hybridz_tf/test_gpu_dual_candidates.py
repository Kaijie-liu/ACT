#!/usr/bin/env python3
# ===- test_gpu_dual_candidates.py - candidate mapping toy gate ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------====#
"""Controlled tests for original-frame GPU dual candidates.

Run from the repository root:

    python -m act.back_end.hybridz_tf.test_gpu_dual_candidates
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.gpu_dual_candidates import (
    CoordinateWavefrontCandidates,
    ConstraintRowTag,
    OriginalFrameLP,
    batched_original_frame_row_duals,
    original_frame_from_operator_hz,
    output_frame_objectives,
    property_conditioned_coordinate_wavefront_duals,
    rank_relu_property_sensitivities,
)
from act.back_end.solver.solver_hz import (
    _hz_candidate_support_attribution,
    _hz_constraint_generation_dual_candidate,
    _hz_gpu_dual_candidate_filter,
    _hz_independent_lp_lagrangian_upper,
    _hz_property_micro_rlt_bridge_candidate_rows,
    _hz_property_micro_rlt_focused_objective_schedule,
    _hz_property_micro_rlt_source_candidate_rows,
)


def _residual_relu_frame() -> OriginalFrameLP:
    """LP for r=x-ReLU(x), x in [-1,1], in normalized HZ coordinates.

    ``x = xi_x`` and ``y = 1/2 + (1/2)xi_y``.  The two triangle rows are:

    * x-y <= 0          -> [1, -1/2] v <= 1/2
    * y <= (x+1)/2     -> [-1/2, 1/2] v <= 0

    The residual objective is
    ``r = -1/2 + [1, -1/2]v``.  Its cube upper bound is one, while multiplier
    d=(1,0) proves the exact LP upper bound zero.
    """

    A = sp.csr_matrix(
        np.asarray(
            [
                [1.0, -0.5],
                [-0.5, 0.5],
            ],
            dtype=np.float64,
        )
    )
    return OriginalFrameLP(
        A=A,
        rl=np.full(2, -np.inf, dtype=np.float64),
        ru=np.asarray([0.5, 0.0], dtype=np.float64),
        lb=-np.ones(2, dtype=np.float64),
        ub=np.ones(2, dtype=np.float64),
        row_tags=(
            ConstraintRowTag(0, "ub", "relu_relaxed_lower:2", 0),
            ConstraintRowTag(
                1, "ub", "relu_relaxed_upper_fraction:2", 0
            ),
        ),
    )


def _shared_id_materialization_packet_toy(
    *,
    shared_id: bool,
    objective_sign: int,
):
    """Return one exact DAG-join frame and its stable column identities.

    ``y`` is the final materialized coordinate and ``x`` is the packet
    coordinate.  The positive/negative objective lanes respectively use

    ``maximize y - 1/10,  x <= 0``

    and

    ``maximize -y - 1/10, -x <= 0``.

    In the causal frame the materialization band joins ``y`` to the *same*
    stable coordinate ``x``.  The negative control has the identical box and
    row-family labels, but joins ``y`` to a fresh ``x_copy`` coordinate while
    the packet still constrains ``x``.  Stable identity, not a coincident
    interval or tag, is therefore the only path from the property to the
    packet.
    """

    sign = int(objective_sign)
    if sign not in {-1, 1}:
        raise ValueError("objective_sign must be -1 or +1")

    output_id = 101
    packet_id = 211
    copied_id = 307
    # Keep the negative-control frame bit-for-bit identical in shape, bounds,
    # stable-id order, and tags.  Only the bridge incidence chooses between
    # the shared packet id and the deliberately copied id.
    continuous_ids = (output_id, packet_id, copied_id)
    id_to_column = {
        stable_id: column
        for column, stable_id in enumerate(continuous_ids)
    }
    joined_id = packet_id if shared_id else copied_id
    n_cont = len(continuous_ids)
    # The final zero column is the live binary frame.  It deliberately remains
    # present because the production packet path is enabled only for a
    # signed-binary HZ, even though this exact toy does not need its value.
    A = np.zeros((3, n_cont + 1), dtype=np.float64)
    A[0, id_to_column[output_id]] = 1.0
    A[0, id_to_column[joined_id]] = -1.0
    A[1] = -A[0]
    A[2, id_to_column[packet_id]] = float(sign)

    Gc = sp.csr_matrix(
        (
            np.asarray([1.0], dtype=np.float64),
            (
                np.asarray([0], dtype=np.int64),
                np.asarray([id_to_column[output_id]], dtype=np.int64),
            ),
        ),
        shape=(1, n_cont),
        dtype=np.float64,
    )
    common = {
        "c": np.asarray([0.0], dtype=np.float64),
        "Gc": Gc,
        "Gb": sp.csr_matrix([[0.0]], dtype=np.float64),
        "C": np.asarray([[float(sign)]], dtype=np.float64),
        "t": np.asarray([0.1], dtype=np.float64),
        "candidate_rows": np.asarray([0], dtype=np.int64),
        "A": sp.csr_matrix(A),
        "rl": np.full(3, -np.inf, dtype=np.float64),
        "ru": np.zeros(3, dtype=np.float64),
        "lb": -np.ones(n_cont + 1, dtype=np.float64),
        "ub": np.ones(n_cont + 1, dtype=np.float64),
        "time_budget": 1.0,
        "steps": 8,
        "row_topk": 0,
        "learning_rate": 0.08,
        "tol": 1.0e-9,
        "constraint_row_tags": (
            "add_materialize:1:forward",
            "add_materialize:1:reverse",
            f"property_micro_rlt:generated:packet_sign_{sign}",
        ),
        "packet_core_seed_rows": np.zeros(0, dtype=np.int64),
    }
    return common, continuous_ids


def _fraction_materialization_packet_upper(
    *,
    shared_id: bool,
    objective_sign: int,
    include_materialization: bool,
    include_packet: bool,
) -> Fraction:
    """Exact vertex oracle for the small stable-ID join family."""

    sign = Fraction(int(objective_sign))
    endpoints = (Fraction(-1), Fraction(0), Fraction(1))
    best = None
    for y in endpoints:
        for x in endpoints:
            for x_copy in endpoints:
                if include_packet and sign * x > 0:
                    continue
                joined = x if shared_id else x_copy
                if include_materialization and y != joined:
                    continue
                value = sign * y - Fraction(1, 10)
                best = value if best is None else max(best, value)
    if best is None:
        raise AssertionError("exact join oracle found no feasible vertex")
    return best


def _lp_materialization_packet_upper(common, rows) -> float:
    """Independent HiGHS measurement for a selected exact toy family."""

    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    q = np.concatenate(
        (
            np.asarray(
                common["C"] @ common["Gc"].toarray(),
                dtype=np.float64,
            ).reshape(-1),
            np.asarray(
                common["C"] @ common["Gb"].toarray(),
                dtype=np.float64,
            ).reshape(-1),
        )
    )
    result = linprog(
        -q,
        A_ub=(
            common["A"][selected, :]
            if selected.size
            else None
        ),
        b_ub=(
            common["ru"][selected]
            if selected.size
            else None
        ),
        bounds=list(zip(common["lb"], common["ub"])),
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun - common["t"][0])


class OriginalFrameCandidateTests(unittest.TestCase):
    def test_support_attribution_partitions_columns_and_row_tags(self):
        receipt = _hz_candidate_support_attribution(
            q=np.asarray([3.0, -2.0, 1.0]),
            A=sp.csr_matrix([[1.0, 0.0, 1.0]]),
            rl=np.asarray([-np.inf]),
            ru=np.asarray([0.0]),
            lb=-np.ones(3),
            ub=np.ones(3),
            row_dual=np.asarray([-1.0]),
            column_layer_ids=np.asarray([3, 5, 3]),
            constraint_row_tags=("relu_relaxed_upper:3",),
        )
        self.assertEqual(receipt["status"], "computed")
        self.assertAlmostEqual(receipt["generator_box_support"], 4.0)
        self.assertAlmostEqual(receipt["constraint_row_support"], 0.0)
        self.assertAlmostEqual(receipt["candidate_support"], 4.0)
        by_layer = {
            item["layer_id"]: item["support"]
            for item in receipt["column_layer_support"]
        }
        self.assertAlmostEqual(by_layer[3], 2.0)
        self.assertAlmostEqual(by_layer[5], 2.0)
        self.assertEqual(
            receipt["constraint_tag_contribution"][0]["tag"],
            "relu_relaxed_upper:3",
        )
        self.assertAlmostEqual(
            receipt["constraint_tag_contribution"][0]["dual_l1"], 1.0
        )

    def test_support_attribution_scales_linearly_with_real_micro_rlt_tags(
        self,
    ):
        # The real CIFAR100 packet has 106,584 constraint rows and 8,208
        # individually named generated rows.  A mask-per-unique-tag
        # implementation performs roughly 8.7e8 string comparisons and can
        # outlive the worker hard deadline even though attribution is only
        # diagnostic.  Keep this exact tag cardinality as a regression gate.
        n_rows = 106_584
        n_generated = 8_208
        tags = (
            ("ordinary_base_row",) * (n_rows - n_generated)
            + tuple(
                f"property_micro_rlt:generated:{index}"
                for index in range(n_generated)
            )
        )
        row_dual = np.zeros(n_rows, dtype=np.float64)
        row_dual[-1] = -1.0
        started = time.monotonic()
        receipt = _hz_candidate_support_attribution(
            q=np.asarray([1.0, -1.0], dtype=np.float64),
            A=sp.csr_matrix((n_rows, 2), dtype=np.float64),
            rl=np.full(n_rows, -np.inf, dtype=np.float64),
            ru=np.zeros(n_rows, dtype=np.float64),
            lb=-np.ones(2, dtype=np.float64),
            ub=np.ones(2, dtype=np.float64),
            row_dual=row_dual,
            constraint_row_tags=tags,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(receipt["status"], "computed")
        self.assertEqual(receipt["tag_aggregation"], "single_pass")
        self.assertEqual(
            receipt["constraint_tag_group_count"], n_generated + 1
        )
        self.assertEqual(
            receipt["constraint_tag_contribution"][0]["tag"],
            f"property_micro_rlt:generated:{n_generated - 1}",
        )
        self.assertAlmostEqual(
            receipt["constraint_tag_contribution"][0]["dual_l1"], 1.0
        )
        self.assertLess(elapsed, 2.0)

    def test_support_attribution_obeys_an_expired_deadline(self):
        receipt = _hz_candidate_support_attribution(
            q=np.asarray([1.0], dtype=np.float64),
            A=sp.csr_matrix((1, 1), dtype=np.float64),
            rl=np.asarray([-np.inf], dtype=np.float64),
            ru=np.asarray([0.0], dtype=np.float64),
            lb=np.asarray([-1.0], dtype=np.float64),
            ub=np.asarray([1.0], dtype=np.float64),
            row_dual=np.asarray([0.0], dtype=np.float64),
            constraint_row_tags=("row",),
            deadline=time.monotonic() - 1.0,
        )
        self.assertEqual(receipt["status"], "skipped_deadline")
        self.assertFalse(receipt["proof_authority"])

    def test_micro_rlt_source_rows_map_to_live_base_lp_order(self):
        hz = SimpleNamespace(n_eq=1, n_ub=3)
        hz._property_micro_rlt_receipt = {
            "schema": "act.property_micro_rlt.v1",
            "base_n_eq": 1,
            "base_n_ub": 2,
            "result_n_ub": 3,
            "selection": [
                {
                    "binary_position": 0,
                    "source_upper_rows": [0],
                }
            ],
        }
        rows = _hz_property_micro_rlt_source_candidate_rows(
            hz,
            constraint_row_tags=(
                "base_eq",
                "source_upper_0",
                "source_upper_1",
                "property_micro_rlt:generated",
            ),
            matrix_row_count=4,
        )
        np.testing.assert_array_equal(rows, np.asarray([1]))

        # Malformed metadata is a heuristic no-op, never an exception or a
        # reason to trust a row outside the live base prefix.
        hz._property_micro_rlt_receipt["selection"][0][
            "source_upper_rows"
        ] = [2]
        self.assertEqual(
            _hz_property_micro_rlt_source_candidate_rows(
                hz,
                constraint_row_tags=(
                    "base_eq",
                    "source_upper_0",
                    "source_upper_1",
                    "property_micro_rlt:generated",
                ),
                matrix_row_count=4,
            ).size,
            0,
        )

    def test_micro_rlt_bridge_selects_bounded_live_constraint_cone(self):
        tags = (
            "add_materialize:20:forward",
            "add_materialize:20:reverse",
            "add_materialize:33:forward",
            "add_materialize:33:reverse",
            "add_materialize:37:forward",
            "add_materialize:37:reverse",
            "relu_relaxed_lower:40",
            "relu_relaxed_upper_fraction:40",
            "relu_exact_lower:40",
            "relu_exact_x_branch:40",
            "ordinary_tail_row",
            "ordinary_tail_row",
            "property_micro_rlt:generated:0",
            "property_micro_rlt:generated:1",
        )
        hz = SimpleNamespace(
            operator_hz_metadata={
                "property_micro_rlt": {
                    "schema": "operator_hz_property_micro_rlt_v1",
                    "status": "applied",
                    "exact_relu_records": [
                        {"layer_id": 40, "row": 8},
                        {"layer_id": 40, "row": 49},
                    ],
                    "base_counts": {
                        "n_eq": 0,
                        "n_ub": 12,
                    },
                    "result_counts": {
                        "n_eq": 0,
                        "n_ub": 14,
                    },
                }
            }
        )
        rows = _hz_property_micro_rlt_bridge_candidate_rows(
            hz,
            constraint_row_tags=tags,
            matrix_row_count=len(tags),
            deadline=time.monotonic() + 1.0,
        )
        # Keep all final-ReLU rows and the two closest complete ADD bands.
        # The farther layer-20 pair and generated rows stay outside the
        # candidate bridge.
        np.testing.assert_array_equal(
            rows,
            np.asarray([2, 3, 4, 5, 6, 7, 8, 9]),
        )
        self.assertTrue(np.all(rows < 12))

        malformed = list(tags)
        malformed[5] = "ordinary_tail_row"
        self.assertEqual(
            _hz_property_micro_rlt_bridge_candidate_rows(
                hz,
                constraint_row_tags=tuple(malformed),
                matrix_row_count=len(malformed),
                deadline=time.monotonic() + 1.0,
            ).size,
            0,
        )
        self.assertEqual(
            _hz_property_micro_rlt_bridge_candidate_rows(
                hz,
                constraint_row_tags=tags,
                matrix_row_count=len(tags),
                deadline=time.monotonic() - 1.0,
            ).size,
            0,
        )

    def test_micro_rlt_objective_schedule_uses_explicit_focus_group(self):
        hz = SimpleNamespace(
            operator_hz_metadata={
                "property_micro_rlt": {
                    "status": "applied",
                    "common_focused_rival_id": 1,
                },
                "property_tail_upper": {
                    "baseline_plane_count": 2,
                    "property_row_groups": [[0, 2], [1, 3]],
                },
            }
        )
        scheduled, deferred, focus, plane_kind = (
            _hz_property_micro_rlt_focused_objective_schedule(
                hz,
                safe_groups=((0, 2), (1, 3)),
                candidate_rows=np.asarray([2, 1, 0, 3]),
                cube_upper=np.asarray([10.0, 1.0, 8.0, 2.0]),
            )
        )
        # Generic hardness would visit group 0 / row 2 first.  The packet was
        # explicitly constructed for group 1, whose tighter plane is row 1.
        np.testing.assert_array_equal(scheduled, np.asarray([1]))
        np.testing.assert_array_equal(deferred, np.asarray([2, 0, 3]))
        self.assertEqual(focus, 1)
        self.assertEqual(plane_kind, "baseline_property_plane")

        hz.operator_hz_metadata["property_micro_rlt"][
            "common_focused_rival_id"
        ] = 7
        scheduled, deferred, focus, plane_kind = (
            _hz_property_micro_rlt_focused_objective_schedule(
                hz,
                safe_groups=((0, 2), (1, 3)),
                candidate_rows=np.asarray([2, 1, 0, 3]),
                cube_upper=np.asarray([10.0, 1.0, 8.0, 2.0]),
            )
        )
        self.assertIsNone(scheduled)
        np.testing.assert_array_equal(deferred, np.asarray([2, 1, 0, 3]))
        self.assertIsNone(focus)
        self.assertIsNone(plane_kind)

    def test_packet_source_row_seeds_generated_row_chain(self):
        # q=x.  The generated-only row w<=0 has zero KKT violation at the
        # cube maximizer and cannot start the wavefront.  Adding the recorded
        # source row x-w<=0 seeds the exact two-row chain x<=w<=0.
        common = {
            "c": np.asarray([-0.1], dtype=np.float64),
            "Gc": sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
            "Gb": sp.csr_matrix([[0.0]], dtype=np.float64),
            "C": np.asarray([[1.0]], dtype=np.float64),
            "t": np.asarray([0.0], dtype=np.float64),
            "candidate_rows": np.asarray([0], dtype=np.int64),
            "A": sp.csr_matrix(
                [
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "rl": np.full(2, -np.inf, dtype=np.float64),
            "ru": np.zeros(2, dtype=np.float64),
            "lb": np.asarray([-1.0, -1.0, 0.0], dtype=np.float64),
            "ub": np.ones(3, dtype=np.float64),
            "time_budget": 1.0,
            "steps": 2,
            "row_topk": 0,
            "learning_rate": 0.08,
            "tol": 1e-9,
            "constraint_row_tags": (
                "source_upper",
                "property_micro_rlt:generated:w_upper",
            ),
        }
        generated_only, generated_stats = _hz_gpu_dual_candidate_filter(
            **common,
            packet_core_seed_rows=np.zeros(0, dtype=np.int64),
            deadline=time.monotonic() + 2.0,
        )
        np.testing.assert_array_equal(generated_only, np.asarray([0]))
        self.assertEqual(generated_stats["gpu_dual_wavefront_updates"], 0)
        self.assertEqual(
            generated_stats["gpu_dual_candidate_constraint_scope"],
            "property_micro_rlt_generated_rows",
        )

        source_seeded, seeded_stats = _hz_gpu_dual_candidate_filter(
            **common,
            packet_core_seed_rows=np.asarray([0], dtype=np.int64),
            deadline=time.monotonic() + 2.0,
        )
        self.assertEqual(source_seeded.size, 0)
        self.assertGreaterEqual(
            seeded_stats["gpu_dual_wavefront_updates"], 2
        )
        self.assertEqual(seeded_stats["gpu_dual_certified_rows"], 1)
        self.assertEqual(
            seeded_stats["gpu_dual_candidate_constraint_scope"],
            "property_micro_rlt_generated_plus_source_rows",
        )
        self.assertEqual(
            seeded_stats["gpu_dual_packet_generated_rows_selected"], 1
        )
        self.assertEqual(
            seeded_stats["gpu_dual_packet_source_rows_selected"], 1
        )
        self.assertAlmostEqual(
            seeded_stats["gpu_dual_support_best_improvement"], 1.0
        )
        self.assertEqual(
            seeded_stats["gpu_dual_checked_generated_nnz_max"], 1
        )
        self.assertEqual(
            seeded_stats["gpu_dual_checked_source_nnz_max"], 1
        )
        self.assertEqual(
            seeded_stats["gpu_dual_checked_other_nnz_max"], 0
        )
        self.assertLess(seeded_stats["gpu_dual_checked_upper_max"], 0.0)
        self.assertTrue(seeded_stats["gpu_dual_coverage_ok"])

    def test_materialization_bridge_and_packet_are_jointly_causal(self):
        # Exact rational oracle:
        #   y,w in [-1,1], y=w, w<=0, maximize y-1/10.
        # The optimum is -1/10.  The packet row w<=0 cannot touch q=y.
        # The materialization band alone merely moves one unit of residual
        # from y to w at zero support gain.  Their joint constraint cone
        # reaches d_forward=d_packet=1 and the independent checker closes the
        # property.  Deleting either component must retain UNKNOWN.
        common = {
            "c": np.asarray([-0.1], dtype=np.float64),
            "Gc": sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
            "Gb": sp.csr_matrix([[0.0]], dtype=np.float64),
            "C": np.asarray([[1.0]], dtype=np.float64),
            "t": np.asarray([0.0], dtype=np.float64),
            "candidate_rows": np.asarray([0], dtype=np.int64),
            "A": sp.csr_matrix(
                [
                    [1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "rl": np.full(3, -np.inf, dtype=np.float64),
            "ru": np.zeros(3, dtype=np.float64),
            "lb": np.asarray([-1.0, -1.0, 0.0], dtype=np.float64),
            "ub": np.ones(3, dtype=np.float64),
            "time_budget": 1.0,
            "steps": 8,
            "row_topk": 0,
            "learning_rate": 0.08,
            "tol": 1e-9,
            "constraint_row_tags": (
                "add_materialize:1:forward",
                "add_materialize:1:reverse",
                "property_micro_rlt:generated:w_upper",
            ),
            "packet_core_seed_rows": np.zeros(0, dtype=np.int64),
        }
        packet_only, packet_stats = _hz_gpu_dual_candidate_filter(
            **common,
            packet_core_bridge_rows=np.zeros(0, dtype=np.int64),
            deadline=time.monotonic() + 2.0,
        )
        np.testing.assert_array_equal(packet_only, np.asarray([0]))
        self.assertEqual(packet_stats["gpu_dual_wavefront_updates"], 0)
        self.assertAlmostEqual(
            packet_stats["gpu_dual_support_best_improvement"], 0.0
        )

        combined, combined_stats = _hz_gpu_dual_candidate_filter(
            **common,
            packet_core_bridge_rows=np.asarray([0, 1], dtype=np.int64),
            deadline=time.monotonic() + 2.0,
        )
        self.assertEqual(combined.size, 0)
        self.assertEqual(
            combined_stats["gpu_dual_device"], "cpu_packet_bridge"
        )
        self.assertEqual(
            combined_stats["gpu_dual_candidate_constraint_scope"],
            "property_micro_rlt_plus_constraint_cone_bridge",
        )
        self.assertEqual(
            combined_stats["gpu_dual_packet_bridge_rows_selected"], 2
        )
        self.assertGreater(
            combined_stats["gpu_dual_checked_bridge_nnz_max"], 0
        )
        self.assertGreater(
            combined_stats["gpu_dual_checked_generated_nnz_max"], 0
        )
        self.assertEqual(
            combined_stats["gpu_dual_checked_other_nnz_max"], 0
        )
        self.assertAlmostEqual(
            combined_stats["gpu_dual_support_best_improvement"], 1.0
        )
        self.assertAlmostEqual(
            combined_stats[
                "gpu_dual_bridge_combined_support_improvement"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            combined_stats["gpu_dual_checked_upper_max"], -0.1
        )
        self.assertTrue(combined_stats["gpu_dual_coverage_ok"])

        # Bridge-only uses no generated tag, hence no restricted packet path;
        # the exact box upper remains +0.9.  Check the oracle directly with a
        # legal bridge multiplier: its zero gain cannot prove the property.
        bridge_only_upper, receipt = _hz_independent_lp_lagrangian_upper(
            c=np.asarray([-0.1], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0, 0.0, 0.0]]),
            C_row=np.asarray([1.0], dtype=np.float64),
            threshold=0.0,
            A=common["A"][:2],
            rl=common["rl"][:2],
            ru=common["ru"][:2],
            lb=common["lb"],
            ub=common["ub"],
            row_dual=np.asarray([-1.0, 0.0], dtype=np.float64),
        )
        self.assertEqual(receipt["status"], "verified_upper")
        self.assertAlmostEqual(float(bridge_only_upper), 0.9)

    def test_pc_cbde_filter_auto_selects_checks_and_replaces_packet_warm(self):
        """The live filter may replace only after four full-A checks.

        The old packet warm ``d_G=1`` is support-flat for ``w in [0,1]``:

          G: w <= 0
          M: y - w <= 0
          maximize y - 1/10.

        The static bridge input is empty.  The CSR selector must discover M,
        the block exchange reaches ``d_G=d_M=1``, and the ordinary outer
        checker must independently certify the replacement.
        """

        common = {
            "c": np.asarray([-0.1], dtype=np.float64),
            "Gc": sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
            "Gb": sp.csr_matrix([[0.0]], dtype=np.float64),
            "C": np.asarray([[1.0]], dtype=np.float64),
            "t": np.asarray([0.0], dtype=np.float64),
            "candidate_rows": np.asarray([0], dtype=np.int64),
            "A": sp.csr_matrix(
                [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0]],
                dtype=np.float64,
            ),
            "rl": np.full(2, -np.inf, dtype=np.float64),
            "ru": np.zeros(2, dtype=np.float64),
            "lb": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
            "ub": np.ones(3, dtype=np.float64),
            "time_budget": 3.0,
            "steps": 2,
            "row_topk": 0,
            "learning_rate": 0.08,
            "tol": 1.0e-9,
            "constraint_row_tags": (
                "property_micro_rlt:generated:w_upper",
                "ordinary_bridge:y_to_w",
            ),
            "packet_core_seed_rows": np.zeros(0, dtype=np.int64),
            "packet_core_bridge_rows": np.zeros(0, dtype=np.int64),
        }

        def old_packet_warm(frame, q, **_kwargs):
            self.assertEqual(frame.n_rows, 1)
            np.testing.assert_array_equal(q, [[1.0, 0.0, 0.0]])
            return CoordinateWavefrontCandidates(
                d=np.asarray([[1.0]], dtype=np.float64),
                initial_support=np.asarray([1.0], dtype=np.float64),
                candidate_support=np.asarray([1.0], dtype=np.float64),
                updates=1,
                selected_constraint_count=1,
                elapsed_seconds=0.0,
                deadline_reached=False,
                method="test_old_packet_warm",
            )

        # Independent primal oracle: full=-1/10; deleting either family=9/10.
        for rows, expected in (
            ((0, 1), Fraction(-1, 10)),
            ((0,), Fraction(9, 10)),
            ((1,), Fraction(9, 10)),
            ((), Fraction(9, 10)),
        ):
            selected = np.asarray(rows, dtype=np.int64)
            primal = linprog(
                np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
                A_ub=(
                    common["A"][selected, :]
                    if selected.size
                    else None
                ),
                b_ub=(
                    common["ru"][selected]
                    if selected.size
                    else None
                ),
                bounds=list(zip(common["lb"], common["ub"])),
                method="highs",
            )
            self.assertTrue(primal.success, primal.message)
            self.assertAlmostEqual(
                float(-primal.fun - 0.1),
                float(expected),
                places=12,
            )

        with patch(
            "act.back_end.hybridz_tf.gpu_dual_candidates."
            "property_conditioned_coordinate_wavefront_duals",
            side_effect=old_packet_warm,
        ):
            survivors, stats = _hz_gpu_dual_candidate_filter(
                **common,
                deadline=time.monotonic() + 4.0,
            )

        self.assertEqual(survivors.size, 0)
        self.assertEqual(stats["gpu_dual_certified_rows"], 1)
        self.assertEqual(
            stats["gpu_dual_pc_cbde_status"],
            "verified_replaced",
        )
        np.testing.assert_array_equal(
            stats["gpu_dual_pc_cbde_cone_rows"],
            [1],
        )
        self.assertEqual(stats["gpu_dual_pc_cbde_full_nnz"], 2)
        self.assertEqual(stats["gpu_dual_pc_cbde_bridge_row_count"], 1)
        self.assertGreaterEqual(stats["gpu_dual_pc_cbde_updates"], 1)
        self.assertTrue(
            stats["gpu_dual_pc_cbde_all_ablations_verified"]
        )
        self.assertTrue(
            stats["gpu_dual_pc_cbde_strict_family_ablation"]
        )
        self.assertAlmostEqual(
            stats["gpu_dual_pc_cbde_strict_family_ablation_tol"],
            512.0 * np.finfo(np.float64).eps,
            places=27,
        )
        self.assertTrue(
            stats["gpu_dual_pc_cbde_replaced_old_candidate"]
        )
        self.assertFalse(stats["gpu_dual_pc_cbde_proof_authority"])
        self.assertAlmostEqual(
            stats["gpu_dual_pc_cbde_checked_upper_full"],
            -0.1,
            places=12,
        )
        self.assertAlmostEqual(
            stats[
                "gpu_dual_pc_cbde_checked_upper_without_generated"
            ],
            0.9,
            places=12,
        )
        self.assertAlmostEqual(
            stats["gpu_dual_pc_cbde_checked_upper_without_bridge"],
            0.9,
            places=12,
        )
        self.assertAlmostEqual(
            stats["gpu_dual_pc_cbde_checked_upper_without_both"],
            0.9,
            places=12,
        )
        self.assertAlmostEqual(
            stats["gpu_dual_pc_cbde_support_improvement"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            stats["gpu_dual_pc_cbde_support_improvement_tol"],
            512.0 * np.finfo(np.float64).eps,
            places=27,
        )
        # The regular outer checker still ran and supplied verdict authority.
        self.assertEqual(
            stats["gpu_dual_certificate_attempted_rows"],
            1,
        )
        self.assertEqual(stats["gpu_dual_checked_bridge_nnz_max"], 1)
        self.assertEqual(stats["gpu_dual_checked_other_nnz_max"], 0)
        self.assertAlmostEqual(
            stats["gpu_dual_checked_upper_max"],
            -0.1,
            places=12,
        )

    def test_pc_cbde_error_and_row_topk_retain_old_packet_candidate(self):
        common = {
            "c": np.asarray([-0.1], dtype=np.float64),
            "Gc": sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
            "Gb": sp.csr_matrix([[0.0]], dtype=np.float64),
            "C": np.asarray([[1.0]], dtype=np.float64),
            "t": np.asarray([0.0], dtype=np.float64),
            "candidate_rows": np.asarray([0], dtype=np.int64),
            "A": sp.csr_matrix(
                [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0]],
                dtype=np.float64,
            ),
            "rl": np.full(2, -np.inf, dtype=np.float64),
            "ru": np.zeros(2, dtype=np.float64),
            "lb": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
            "ub": np.ones(3, dtype=np.float64),
            "time_budget": 3.0,
            "steps": 2,
            "learning_rate": 0.08,
            "tol": 1.0e-9,
            "constraint_row_tags": (
                "property_micro_rlt:generated:w_upper",
                "ordinary_bridge:y_to_w",
            ),
            "packet_core_seed_rows": np.zeros(0, dtype=np.int64),
            "packet_core_bridge_rows": np.zeros(0, dtype=np.int64),
        }

        def old_packet_warm(*_args, **_kwargs):
            return CoordinateWavefrontCandidates(
                d=np.asarray([[1.0]], dtype=np.float64),
                initial_support=np.asarray([1.0], dtype=np.float64),
                candidate_support=np.asarray([1.0], dtype=np.float64),
                updates=1,
                selected_constraint_count=1,
                elapsed_seconds=0.0,
                deadline_reached=False,
                method="test_old_packet_warm",
            )

        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "property_conditioned_coordinate_wavefront_duals",
                side_effect=old_packet_warm,
            ),
            patch(
                "act.back_end.hybridz_tf."
                "property_causal_block_integration."
                "property_causal_block_integration",
                side_effect=RuntimeError("controlled PC failure"),
            ),
        ):
            error_rows, error_stats = _hz_gpu_dual_candidate_filter(
                **common,
                row_topk=0,
                deadline=time.monotonic() + 4.0,
            )
        np.testing.assert_array_equal(error_rows, [0])
        self.assertEqual(
            error_stats["gpu_dual_pc_cbde_status"],
            "error:RuntimeError",
        )
        self.assertFalse(
            error_stats["gpu_dual_pc_cbde_replaced_old_candidate"]
        )
        self.assertEqual(error_stats["gpu_dual_errors"], 0)
        self.assertEqual(
            error_stats["gpu_dual_certificate_attempted_rows"],
            1,
        )
        self.assertAlmostEqual(
            error_stats["gpu_dual_checked_upper_max"],
            0.9,
            places=12,
        )

        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "property_conditioned_coordinate_wavefront_duals",
                side_effect=old_packet_warm,
            ),
            patch(
                "act.back_end.hybridz_tf."
                "property_causal_block_integration."
                "property_causal_block_integration",
                return_value=SimpleNamespace(
                    success=False,
                    status="deadline",
                    deadline_reached=True,
                ),
            ),
        ):
            timeout_rows, timeout_stats = _hz_gpu_dual_candidate_filter(
                **common,
                row_topk=0,
                deadline=time.monotonic() + 4.0,
            )
        np.testing.assert_array_equal(timeout_rows, [0])
        self.assertEqual(
            timeout_stats["gpu_dual_pc_cbde_status"],
            "integration_deadline",
        )
        self.assertTrue(
            timeout_stats["gpu_dual_pc_cbde_deadline_reached"]
        )
        self.assertFalse(
            timeout_stats["gpu_dual_pc_cbde_replaced_old_candidate"]
        )
        self.assertFalse(
            timeout_stats["gpu_dual_pc_cbde_proof_authority"]
        )
        self.assertEqual(
            timeout_stats["gpu_dual_certificate_attempted_rows"],
            1,
        )
        self.assertEqual(
            timeout_stats["gpu_dual_checked_bridge_nnz_max"],
            0,
        )
        self.assertAlmostEqual(
            timeout_stats["gpu_dual_checked_upper_max"],
            0.9,
            places=12,
        )

        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "property_conditioned_coordinate_wavefront_duals",
                side_effect=old_packet_warm,
            ),
            patch(
                "act.back_end.hybridz_tf."
                "property_causal_block_integration."
                "property_causal_block_integration",
                side_effect=AssertionError(
                    "row_topk gate must skip PC integration"
                ),
            ),
        ):
            topk_rows, topk_stats = _hz_gpu_dual_candidate_filter(
                **common,
                row_topk=1,
                deadline=time.monotonic() + 4.0,
            )
        np.testing.assert_array_equal(topk_rows, [0])
        self.assertEqual(
            topk_stats["gpu_dual_pc_cbde_status"],
            "skipped_row_topk",
        )
        self.assertFalse(
            topk_stats["gpu_dual_pc_cbde_replaced_old_candidate"]
        )
        self.assertEqual(
            topk_stats["gpu_dual_certificate_attempted_rows"],
            1,
        )
        self.assertAlmostEqual(
            topk_stats["gpu_dual_checked_upper_max"],
            0.9,
            places=12,
        )

    def test_pc_cbde_truncates_generated_warm_before_family_ablation(self):
        generated_count = 66
        dense = np.zeros((generated_count + 1, 3), dtype=np.float64)
        dense[:generated_count, 1] = 1.0
        dense[generated_count, :2] = (1.0, -1.0)
        tags = tuple(
            f"property_micro_rlt:generated:w_upper:{row}"
            for row in range(generated_count)
        ) + ("ordinary_bridge:y_to_w",)

        def old_packet_warm(frame, _q, **_kwargs):
            self.assertEqual(frame.n_rows, generated_count)
            return CoordinateWavefrontCandidates(
                d=np.ones((1, generated_count), dtype=np.float64),
                initial_support=np.asarray([1.0], dtype=np.float64),
                candidate_support=np.asarray([1.0], dtype=np.float64),
                updates=generated_count,
                selected_constraint_count=generated_count,
                elapsed_seconds=0.0,
                deadline_reached=False,
                method="test_large_old_packet_warm",
            )

        with patch(
            "act.back_end.hybridz_tf.gpu_dual_candidates."
            "property_conditioned_coordinate_wavefront_duals",
            side_effect=old_packet_warm,
        ):
            survivors, stats = _hz_gpu_dual_candidate_filter(
                c=np.asarray([-0.1], dtype=np.float64),
                Gc=sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
                Gb=sp.csr_matrix([[0.0]], dtype=np.float64),
                C=np.asarray([[1.0]], dtype=np.float64),
                t=np.asarray([0.0], dtype=np.float64),
                candidate_rows=np.asarray([0], dtype=np.int64),
                A=sp.csr_matrix(dense),
                rl=np.full(generated_count + 1, -np.inf),
                ru=np.zeros(generated_count + 1),
                lb=np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
                ub=np.ones(3, dtype=np.float64),
                deadline=time.monotonic() + 4.0,
                time_budget=3.0,
                steps=2,
                row_topk=0,
                learning_rate=0.08,
                tol=1.0e-9,
                constraint_row_tags=tags,
                packet_core_seed_rows=np.zeros(0, dtype=np.int64),
                packet_core_bridge_rows=np.zeros(0, dtype=np.int64),
            )

        self.assertEqual(survivors.size, 0)
        self.assertEqual(
            stats["gpu_dual_pc_cbde_status"],
            "verified_replaced",
        )
        self.assertEqual(
            stats["gpu_dual_pc_cbde_generated_warm_nonzero_count"],
            66,
        )
        self.assertEqual(
            stats["gpu_dual_pc_cbde_generated_warm_truncated_count"],
            2,
        )
        self.assertEqual(
            stats["gpu_dual_pc_cbde_generated_row_count"],
            64,
        )
        self.assertEqual(stats["gpu_dual_pc_cbde_full_nnz"], 65)
        self.assertTrue(
            stats["gpu_dual_pc_cbde_strict_family_ablation"]
        )
        self.assertAlmostEqual(
            stats[
                "gpu_dual_pc_cbde_checked_upper_without_generated"
            ],
            0.9,
            places=12,
        )
        self.assertEqual(
            stats["gpu_dual_checked_generated_nnz_max"],
            64,
        )
        self.assertEqual(stats["gpu_dual_checked_bridge_nnz_max"], 1)
        self.assertEqual(stats["gpu_dual_checked_other_nnz_max"], 0)

    def test_shared_stable_id_join_and_wrong_copy_family_ablation(self):
        """Only a live shared column may connect the property to its packet.

        The two objective signs cover both directions of the stored
        materialization equality band.  For each sign, deleting the *whole*
        materialization family or the packet family leaves exact upper
        ``9/10``.  The complete shared-ID family has exact upper ``-1/10``.
        Replacing the join source by an interval-identical fresh stable column
        also leaves exact upper ``9/10`` and must not be certified.
        """

        for sign in (1, -1):
            with self.subTest(objective_sign=sign):
                shared, shared_ids = (
                    _shared_id_materialization_packet_toy(
                        shared_id=True,
                        objective_sign=sign,
                    )
                )
                copied, copied_ids = (
                    _shared_id_materialization_packet_toy(
                        shared_id=False,
                        objective_sign=sign,
                    )
                )
                self.assertEqual(shared_ids, copied_ids)
                self.assertEqual(len(set(shared_ids)), len(shared_ids))
                np.testing.assert_array_equal(
                    shared["lb"], copied["lb"]
                )
                np.testing.assert_array_equal(
                    shared["ub"], copied["ub"]
                )
                self.assertEqual(
                    shared["constraint_row_tags"],
                    copied["constraint_row_tags"],
                )
                # The packet row is identical.  Only the bridge source column
                # changes from stable x to the fresh x_copy.
                np.testing.assert_array_equal(
                    shared["A"].getrow(2).toarray(),
                    copied["A"].getrow(2).toarray(),
                )
                self.assertFalse(
                    np.array_equal(
                        shared["A"].getrow(0).toarray(),
                        copied["A"].getrow(0).toarray(),
                    )
                )

                family_rows = {
                    "complete": (0, 1, 2),
                    "materialization_deleted": (2,),
                    "packet_deleted": (0, 1),
                    "both_deleted": (),
                }
                expected = {
                    "complete": Fraction(-1, 10),
                    "materialization_deleted": Fraction(9, 10),
                    "packet_deleted": Fraction(9, 10),
                    "both_deleted": Fraction(9, 10),
                }
                for family, rows in family_rows.items():
                    include_materialization = (
                        family not in {
                            "materialization_deleted",
                            "both_deleted",
                        }
                    )
                    include_packet = family not in {
                        "packet_deleted",
                        "both_deleted",
                    }
                    exact = _fraction_materialization_packet_upper(
                        shared_id=True,
                        objective_sign=sign,
                        include_materialization=include_materialization,
                        include_packet=include_packet,
                    )
                    self.assertEqual(exact, expected[family], family)
                    self.assertAlmostEqual(
                        _lp_materialization_packet_upper(shared, rows),
                        float(exact),
                        places=10,
                        msg=family,
                    )

                complete_rows, complete_stats = (
                    _hz_gpu_dual_candidate_filter(
                        **shared,
                        packet_core_bridge_rows=np.asarray(
                            [0, 1], dtype=np.int64
                        ),
                        deadline=time.monotonic() + 2.0,
                    )
                )
                self.assertEqual(complete_rows.size, 0)
                self.assertEqual(
                    complete_stats["gpu_dual_certified_rows"], 1
                )
                self.assertGreater(
                    complete_stats[
                        "gpu_dual_checked_bridge_nnz_max"
                    ],
                    0,
                )
                self.assertGreater(
                    complete_stats[
                        "gpu_dual_checked_generated_nnz_max"
                    ],
                    0,
                )
                self.assertAlmostEqual(
                    complete_stats["gpu_dual_checked_upper_max"],
                    -0.1,
                    places=12,
                )
                self.assertAlmostEqual(
                    complete_stats[
                        "gpu_dual_support_best_improvement"
                    ],
                    1.0,
                    places=12,
                )
                self.assertTrue(
                    complete_stats["gpu_dual_coverage_ok"]
                )

                no_bridge_rows, no_bridge_stats = (
                    _hz_gpu_dual_candidate_filter(
                        **shared,
                        packet_core_bridge_rows=np.zeros(
                            0, dtype=np.int64
                        ),
                        deadline=time.monotonic() + 2.0,
                    )
                )
                np.testing.assert_array_equal(
                    no_bridge_rows, np.asarray([0], dtype=np.int64)
                )
                self.assertEqual(
                    no_bridge_stats["gpu_dual_certified_rows"], 0
                )
                self.assertAlmostEqual(
                    no_bridge_stats["gpu_dual_checked_upper_max"],
                    0.9,
                    places=12,
                )

                copied_exact = (
                    _fraction_materialization_packet_upper(
                        shared_id=False,
                        objective_sign=sign,
                        include_materialization=True,
                        include_packet=True,
                    )
                )
                self.assertEqual(copied_exact, Fraction(9, 10))
                self.assertAlmostEqual(
                    _lp_materialization_packet_upper(
                        copied, (0, 1, 2)
                    ),
                    float(copied_exact),
                    places=10,
                )
                copied_rows, copied_stats = (
                    _hz_gpu_dual_candidate_filter(
                        **copied,
                        packet_core_bridge_rows=np.asarray(
                            [0, 1], dtype=np.int64
                        ),
                        deadline=time.monotonic() + 2.0,
                    )
                )
                np.testing.assert_array_equal(
                    copied_rows, np.asarray([0], dtype=np.int64)
                )
                self.assertEqual(
                    copied_stats["gpu_dual_certified_rows"], 0
                )
                self.assertEqual(
                    copied_stats[
                        "gpu_dual_checked_generated_nnz_max"
                    ],
                    0,
                )
                self.assertAlmostEqual(
                    copied_stats["gpu_dual_support_best_improvement"],
                    0.0,
                    places=12,
                )
                self.assertGreaterEqual(
                    copied_stats["gpu_dual_checked_upper_max"],
                    0.899999999,
                )
                self.assertTrue(copied_stats["gpu_dual_coverage_ok"])

    def test_coordinate_wavefront_matches_one_row_exact_oracle(self) -> None:
        frame = _residual_relu_frame()
        q = np.asarray([[1.0, -0.5]], dtype=np.float64)
        candidates = property_conditioned_coordinate_wavefront_duals(
            frame,
            q,
            max_updates=8,
            frontier_topk=2,
            refresh_batch=1,
        )
        self.assertFalse(candidates.proof_authority)
        self.assertGreaterEqual(candidates.updates, 1)
        self.assertEqual(candidates.selected_constraint_count, 1)
        self.assertAlmostEqual(candidates.initial_support[0], 1.5)
        self.assertAlmostEqual(candidates.candidate_support[0], 0.5)
        self.assertAlmostEqual(candidates.d[0, 0], 1.0)
        self.assertAlmostEqual(candidates.d[0, 1], 0.0)

    def test_coordinate_wavefront_crosses_zero_gain_chain(self) -> None:
        # x0 <= x1 <= x2 <= 0.  The first two dual-coordinate moves are
        # support-neutral, so a strictly improving greedy rule cannot reach
        # the final useful row.  The wavefront deliberately moves to the far
        # edge of each flat segment and recovers d=(1,1,1).
        frame = OriginalFrameLP(
            A=sp.csr_matrix(
                np.asarray(
                    [
                        [1.0, -1.0, 0.0],
                        [0.0, 1.0, -1.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            ),
            rl=np.full(3, -np.inf, dtype=np.float64),
            ru=np.zeros(3, dtype=np.float64),
            lb=-np.ones(3, dtype=np.float64),
            ub=np.ones(3, dtype=np.float64),
            row_tags=tuple(
                ConstraintRowTag(row, "ub", f"chain:{row}", 0)
                for row in range(3)
            ),
        )
        candidates = property_conditioned_coordinate_wavefront_duals(
            frame,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
            max_updates=8,
            frontier_topk=3,
            refresh_batch=1,
        )
        self.assertEqual(candidates.updates, 3)
        np.testing.assert_allclose(candidates.d[0], np.ones(3), atol=1e-12)
        self.assertAlmostEqual(candidates.initial_support[0], 1.0)
        self.assertAlmostEqual(candidates.candidate_support[0], 0.0)

    def test_constraint_generation_adds_violated_chain_rows_and_rechecks(
        self,
    ) -> None:
        frame = OriginalFrameLP(
            A=sp.csr_matrix(
                np.asarray(
                    [
                        [1.0, -1.0, 0.0],
                        [0.0, 1.0, -1.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            ),
            rl=np.full(3, -np.inf, dtype=np.float64),
            ru=np.zeros(3, dtype=np.float64),
            lb=-np.ones(3, dtype=np.float64),
            ub=np.ones(3, dtype=np.float64),
            row_tags=tuple(
                ConstraintRowTag(row, "ub", f"chain:{row}", 0)
                for row in range(3)
            ),
        )
        # Seed only the first row.  The reduced primal must expose row 1 and
        # then row 2 as violated cuts before the exact support reaches zero.
        row_dual, stats = _hz_constraint_generation_dual_candidate(
            q=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            A=frame.A,
            rl=frame.rl,
            ru=frame.ru,
            lb=frame.lb,
            ub=frame.ub,
            seed_row_dual=np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
            deadline=time.monotonic() + 3.0,
            max_rounds=6,
            add_batch=1,
            max_selected_rows=3,
        )
        self.assertIsNotNone(row_dual, stats)
        self.assertGreaterEqual(stats["rounds_completed"], 3)
        self.assertEqual(stats["rows_added_by_violation"], 2)
        self.assertEqual(stats["rows_selected"], 3)
        self.assertAlmostEqual(stats["best_support"], 0.0, places=9)
        upper, receipt = _hz_independent_lp_lagrangian_upper(
            c=np.asarray([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0, 0.0, 0.0]], dtype=np.float64),
            C_row=np.asarray([1.0], dtype=np.float64),
            threshold=1.0e-10,
            A=frame.A,
            rl=frame.rl,
            ru=frame.ru,
            lb=frame.lb,
            ub=frame.ub,
            row_dual=row_dual,
        )
        self.assertEqual(receipt["status"], "verified_upper")
        self.assertLess(upper, 0.0)

    def test_tiny_residual_exact_row_mapping_and_checker_authority(self) -> None:
        frame = _residual_relu_frame()
        # Manually mapped d=(1,0); checker input is row_dual=-d.
        upper, receipt = _hz_independent_lp_lagrangian_upper(
            c=np.asarray([-0.5], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0, -0.5]], dtype=np.float64),
            C_row=np.asarray([1.0], dtype=np.float64),
            threshold=1.0e-10,
            A=frame.A,
            rl=frame.rl,
            ru=frame.ru,
            lb=frame.lb,
            ub=frame.ub,
            row_dual=np.asarray([-1.0, 0.0], dtype=np.float64),
        )
        self.assertIsNotNone(upper, receipt)
        self.assertEqual(receipt["status"], "verified_upper")
        self.assertLess(upper, 0.0)
        self.assertEqual(frame.row_tags[0].block_tag, "relu_relaxed_lower:2")

    def test_batched_candidate_improves_residual_cube(self) -> None:
        frame = _residual_relu_frame()
        # First rival is r=x-y; second is -r.  This also checks that rival
        # tensor rows remain independent.
        q = np.asarray(
            [
                [1.0, -0.5],
                [-1.0, 0.5],
            ],
            dtype=np.float64,
        )
        candidates = batched_original_frame_row_duals(
            frame,
            q,
            device="cpu",
            steps=160,
            learning_rate=0.08,
        )
        self.assertFalse(candidates.proof_authority)
        self.assertEqual(candidates.row_dual.shape, (2, 2))
        self.assertTrue(
            np.all(candidates.candidate_support <= candidates.initial_support)
        )
        # r<=0 uses the first lower-envelope row with d≈1.  The candidate
        # support excludes kappa=-1/2, so its optimum here is 1/2.
        self.assertLess(candidates.candidate_support[0], 0.55)
        self.assertLess(candidates.row_dual[0, 0], -0.9)
        self.assertEqual(candidates.steps_completed, 160)
        self.assertFalse(candidates.deadline_reached)

    def test_selected_generated_rows_eliminate_zero_objective_auxiliary(
        self,
    ) -> None:
        # Full frame columns are (x, w, z).  The first row is an unrelated
        # base constraint.  The two selected generated rows imply x<=w<=0,
        # even though the objective q=x has zero coefficient on the product
        # auxiliary w.  Candidate row selection has no authority: the sparse
        # multiplier is expanded back to all three rows and checked on full A.
        frame = OriginalFrameLP(
            A=sp.csr_matrix(
                np.asarray(
                    [
                        [0.0, 0.0, 1.0],
                        [1.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float64,
                )
            ),
            rl=np.full(3, -np.inf, dtype=np.float64),
            ru=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            lb=np.asarray([-1.0, -1.0, 0.0], dtype=np.float64),
            ub=np.ones(3, dtype=np.float64),
            row_tags=tuple(
                ConstraintRowTag(row, "ub", f"row:{row}", 0)
                for row in range(3)
            ),
        )
        candidates = batched_original_frame_row_duals(
            frame,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
            device="cpu",
            steps=2,
            learning_rate=0.08,
            candidate_rows=np.asarray([1, 2], dtype=np.int64),
        )
        self.assertEqual(candidates.row_dual.shape, (1, 3))
        self.assertEqual(candidates.row_dual[0, 0], 0.0)
        self.assertLess(candidates.candidate_support[0], 1e-9)
        upper, receipt = _hz_independent_lp_lagrangian_upper(
            c=np.asarray([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0, 0.0, 0.0]]),
            C_row=np.asarray([1.0], dtype=np.float64),
            threshold=0.1,
            A=frame.A,
            rl=frame.rl,
            ru=frame.ru,
            lb=frame.lb,
            ub=frame.ub,
            row_dual=candidates.row_dual[0],
        )
        self.assertEqual(receipt["status"], "verified_upper")
        self.assertLess(upper, 0.0)

    def test_candidate_explicitly_enables_grad_inside_verifier_no_grad(self) -> None:
        frame = _residual_relu_frame()
        with torch.no_grad():
            candidates = batched_original_frame_row_duals(
                frame,
                np.asarray([[1.0, -0.5]], dtype=np.float64),
                device="cpu",
                steps=4,
                learning_rate=0.08,
            )
        self.assertEqual(candidates.steps_completed, 4)
        self.assertTrue(
            np.all(candidates.candidate_support <= candidates.initial_support)
        )
        self.assertTrue(np.all(np.isfinite(candidates.row_dual)))

    def test_expired_deadline_returns_zero_candidate_fail_closed(self) -> None:
        frame = _residual_relu_frame()
        candidates = batched_original_frame_row_duals(
            frame,
            np.asarray([[1.0, -0.5]], dtype=np.float64),
            device="cpu",
            steps=160,
            deadline=time.monotonic() - 1.0,
        )
        self.assertTrue(candidates.deadline_reached)
        self.assertEqual(candidates.steps_completed, 0)
        np.testing.assert_array_equal(candidates.row_dual, np.zeros((1, 2)))
        np.testing.assert_array_equal(
            candidates.candidate_support,
            candidates.initial_support,
        )

    def test_operator_export_order_and_output_objective_mapping(self) -> None:
        hz = SimpleNamespace(
            c=np.asarray([2.0], dtype=np.float64),
            Gc=sp.csr_matrix([[3.0, 0.0]], dtype=np.float64),
            Gb=sp.csr_matrix([[4.0]], dtype=np.float64),
            Ac=sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
            Ab=sp.csr_matrix([[0.0]], dtype=np.float64),
            b=np.asarray([0.25], dtype=np.float64),
            Auc=sp.csr_matrix([[0.0, 2.0]], dtype=np.float64),
            Aub=sp.csr_matrix([[-1.0]], dtype=np.float64),
            ub=np.asarray([0.5], dtype=np.float64),
            operator_hz_metadata={
                "constraint_tags_eq": [{"tag": "eq_band", "rows": 1}],
                "constraint_tags_ub": [{"tag": "relu_upper", "rows": 1}],
            },
        )
        frame = original_frame_from_operator_hz(hz)
        self.assertEqual(frame.A.shape, (2, 3))
        np.testing.assert_array_equal(
            frame.A.toarray(),
            np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]]),
        )
        self.assertEqual(
            [(tag.global_row, tag.sense, tag.block_tag) for tag in frame.row_tags],
            [(0, "eq", "eq_band"), (1, "ub", "relu_upper")],
        )
        kappa, q = output_frame_objectives(
            hz,
            np.asarray([[2.0], [-1.0]], dtype=np.float64),
            thresholds=np.asarray([1.0, 0.5], dtype=np.float64),
        )
        np.testing.assert_array_equal(kappa, np.asarray([3.0, -2.5]))
        np.testing.assert_array_equal(
            q,
            np.asarray([[6.0, 0.0, 8.0], [-3.0, 0.0, -4.0]]),
        )

    def test_dualsolver_lane_mapping_is_heuristic_only(self) -> None:
        # M=3 rival rows. Neuron 0 is most sensitive to rival 1; neuron 1 is
        # most sensitive to rival 2. Stable neuron 2 must never be scheduled.
        nu = torch.tensor(
            [
                [1.0, 0.0, 100.0],
                [5.0, 1.0, 100.0],
                [2.0, 7.0, 100.0],
            ],
            dtype=torch.float64,
        )
        bounds = Bounds(
            lb=torch.tensor([[-1.0, -2.0, 0.2]], dtype=torch.float64),
            ub=torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float64),
        )
        ranked = rank_relu_property_sensitivities(
            {7: nu},
            {7: bounds},
            M=3,
            top_k=2,
        )
        self.assertEqual(
            [(item.layer_id, item.neuron, item.rival) for item in ranked],
            [(7, 1, 2), (7, 0, 1)],
        )
        self.assertAlmostEqual(ranked[0].score, 7.0)
        self.assertAlmostEqual(ranked[1].score, 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
