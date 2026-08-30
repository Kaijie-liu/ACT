#!/usr/bin/env python3
# ===- test_property_causal_block_dual.py - PC-CBDE toy gates -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Controlled exact-LP gates for the proof-neutral PC-CBDE primitive."""

from __future__ import annotations

from fractions import Fraction
import time
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from act.back_end.hybridz_tf.gpu_dual_candidates import (
    ConstraintRowTag,
    OriginalFrameLP,
    property_conditioned_coordinate_wavefront_duals,
)
from act.back_end.hybridz_tf.property_causal_block_dual import (
    CausalDirectionUnion,
    CausalDualBlock,
    property_causal_block_duals,
)


def _stall_problem(
    *,
    row_order: tuple[int, int] = (0, 1),
    wrong_copy: bool = False,
) -> tuple[
    OriginalFrameLP,
    np.ndarray,
    np.ndarray,
    tuple[CausalDualBlock, ...],
    tuple[CausalDirectionUnion, ...],
]:
    """Return the coordinate-stationary two-family exchange LP.

    Shared frame:

      maximize x0
      x0 + x1/2 <= 0       (materialization)
      x0 - x1/2 <= 0       (generated)
      x in [-1, 1]^n

    The wrong-copy control keeps a third, interval-identical coordinate and
    makes the generated row consume that independent coordinate.
    """

    if sorted(row_order) != [0, 1]:
        raise ValueError("row_order must be a permutation of (0, 1)")
    if wrong_copy:
        canonical_A = np.asarray(
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.0, -0.5],
            ],
            dtype=np.float64,
        )
        q = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    else:
        canonical_A = np.asarray(
            [
                [1.0, 0.5],
                [1.0, -0.5],
            ],
            dtype=np.float64,
        )
        q = np.asarray([[1.0, 0.0]], dtype=np.float64)
    ordered_A = canonical_A[np.asarray(row_order, dtype=np.int64)]
    canonical_tags = (
        ConstraintRowTag(
            0, "ub", "add_materialize:shared:forward", 0
        ),
        ConstraintRowTag(
            1, "ub", "property_micro_rlt:generated:shared", 0
        ),
    )
    tags = tuple(canonical_tags[index] for index in row_order)
    tags = tuple(
        ConstraintRowTag(
            global_row=position,
            sense=tag.sense,
            block_tag=tag.block_tag,
            block_local_row=tag.block_local_row,
        )
        for position, tag in enumerate(tags)
    )
    frame = OriginalFrameLP(
        A=sp.csr_matrix(ordered_A),
        rl=np.full(2, -np.inf, dtype=np.float64),
        ru=np.zeros(2, dtype=np.float64),
        lb=-np.ones(ordered_A.shape[1], dtype=np.float64),
        ub=np.ones(ordered_A.shape[1], dtype=np.float64),
        row_tags=tags,
    )
    position = {
        canonical_row: ordered_position
        for ordered_position, canonical_row in enumerate(row_order)
    }
    blocks = (
        CausalDualBlock(
            block_id="materialization",
            family_id="materialization",
            global_rows=(position[0],),
            stable_row_keys=("materialization:forward:0",),
            incident_columns=tuple(range(ordered_A.shape[1])),
        ),
        CausalDualBlock(
            block_id="generated",
            family_id="generated",
            global_rows=(position[1],),
            stable_row_keys=("generated:packet:0",),
            incident_columns=tuple(range(ordered_A.shape[1])),
        ),
    )
    unions = (
        CausalDirectionUnion(
            union_id="complete_causal_path",
            block_ids=("materialization", "generated"),
        ),
    )
    warm = np.zeros((1, 2), dtype=np.float64)
    warm[0, position[0]] = 1.0
    return frame, q, warm, blocks, unions


def _fraction_dual_support(
    materialization: Fraction,
    generated: Fraction,
) -> Fraction:
    return (
        abs(Fraction(1) - materialization - generated)
        + abs(generated - materialization) / 2
    )


def _highs_primal_upper(
    frame: OriginalFrameLP,
    q: np.ndarray,
    rows: tuple[int, ...],
) -> float:
    A_ub = (
        frame.A[np.asarray(rows, dtype=np.int64)]
        if rows
        else None
    )
    b_ub = (
        np.asarray(frame.ru, dtype=np.float64)[
            np.asarray(rows, dtype=np.int64)
        ]
        if rows
        else None
    )
    result = linprog(
        -np.asarray(q, dtype=np.float64).reshape(-1),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=list(
            zip(
                np.asarray(frame.lb, dtype=np.float64),
                np.asarray(frame.ub, dtype=np.float64),
            )
        ),
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun)


class PropertyCausalBlockDualTests(unittest.TestCase):
    def test_coordinate_stalls_but_block_exchange_reaches_exact_lp(self):
        frame, q, warm, blocks, unions = _stall_problem()
        old_joint = property_conditioned_coordinate_wavefront_duals(
            frame,
            q,
            max_updates=64,
            frontier_topk=64,
            refresh_batch=4,
        )
        np.testing.assert_array_equal(
            old_joint.d, np.asarray([[1.0, 0.0]], dtype=np.float64)
        )
        self.assertEqual(old_joint.updates, 1)
        self.assertEqual(old_joint.initial_support[0], 1.0)
        self.assertEqual(old_joint.candidate_support[0], 0.5)

        residual_q = q - np.asarray(
            frame.A.transpose() @ old_joint.d.transpose(),
            dtype=np.float64,
        ).transpose()
        packet_frame = OriginalFrameLP(
            A=frame.A[[1]],
            rl=frame.rl[[1]],
            ru=frame.ru[[1]],
            lb=frame.lb,
            ub=frame.ub,
            row_tags=(frame.row_tags[1],),
        )
        old_packet = property_conditioned_coordinate_wavefront_duals(
            packet_frame,
            residual_q,
            max_updates=32,
            frontier_topk=64,
            refresh_batch=4,
        )
        self.assertEqual(old_packet.updates, 0)
        np.testing.assert_array_equal(old_packet.d, np.zeros((1, 1)))
        self.assertEqual(old_packet.candidate_support[0], 0.5)

        candidate = property_causal_block_duals(
            frame,
            q,
            warm,
            blocks=blocks,
            direction_unions=unions,
        )
        self.assertFalse(candidate.proof_authority)
        self.assertGreaterEqual(candidate.block_updates, 1)
        self.assertEqual(candidate.candidate_support[0], 0.0)
        np.testing.assert_array_equal(
            candidate.d,
            np.asarray([[0.5, 0.5]], dtype=np.float64),
        )
        self.assertEqual(
            _fraction_dual_support(Fraction(1, 2), Fraction(1, 2)),
            Fraction(0),
        )
        self.assertEqual(_highs_primal_upper(frame, q[0], (0, 1)), 0.0)

    def test_family_masks_match_fraction_and_highs_ablations(self):
        frame, q, warm, blocks, unions = _stall_problem()
        cases = {
            ("materialization", "generated"): (
                Fraction(0),
                (0, 1),
            ),
            ("materialization",): (Fraction(1, 2), (0,)),
            ("generated",): (Fraction(1, 2), (1,)),
            (): (Fraction(1), ()),
        }
        for families, (expected, primal_rows) in cases.items():
            with self.subTest(enabled_families=families):
                candidate = property_causal_block_duals(
                    frame,
                    q,
                    warm,
                    blocks=blocks,
                    direction_unions=unions,
                    enabled_families=families,
                )
                self.assertFalse(candidate.proof_authority)
                self.assertEqual(
                    candidate.candidate_support[0], float(expected)
                )
                self.assertEqual(
                    _highs_primal_upper(frame, q[0], primal_rows),
                    float(expected),
                )

        self.assertEqual(
            _fraction_dual_support(Fraction(1), Fraction(0)),
            Fraction(1, 2),
        )
        self.assertEqual(
            _fraction_dual_support(Fraction(0), Fraction(1)),
            Fraction(1, 2),
        )
        self.assertEqual(
            _fraction_dual_support(Fraction(0), Fraction(0)),
            Fraction(1),
        )

    def test_wrong_copy_cannot_close_the_shared_coordinate_gap(self):
        shared = _stall_problem(wrong_copy=False)
        wrong = _stall_problem(wrong_copy=True)
        shared_candidate = property_causal_block_duals(
            shared[0],
            shared[1],
            shared[2],
            blocks=shared[3],
            direction_unions=shared[4],
        )
        wrong_candidate = property_causal_block_duals(
            wrong[0],
            wrong[1],
            wrong[2],
            blocks=wrong[3],
            direction_unions=wrong[4],
        )
        self.assertEqual(shared_candidate.candidate_support[0], 0.0)
        self.assertEqual(wrong_candidate.candidate_support[0], 0.5)
        self.assertEqual(
            _highs_primal_upper(wrong[0], wrong[1][0], (0, 1)),
            0.5,
        )
        self.assertFalse(wrong_candidate.proof_authority)

    def test_row_permutation_is_stable_under_semantic_row_keys(self):
        ordinary = _stall_problem(row_order=(0, 1))
        permuted = _stall_problem(row_order=(1, 0))
        first = property_causal_block_duals(
            ordinary[0],
            ordinary[1],
            ordinary[2],
            blocks=ordinary[3],
            direction_unions=ordinary[4],
        )
        second = property_causal_block_duals(
            permuted[0],
            permuted[1],
            permuted[2],
            blocks=permuted[3],
            direction_unions=permuted[4],
        )

        def by_stable_key(
            d: np.ndarray,
            blocks: tuple[CausalDualBlock, ...],
        ) -> dict[str, float]:
            return {
                key: float(d[0, row])
                for block in blocks
                for row, key in zip(
                    block.global_rows, block.stable_row_keys
                )
            }

        self.assertEqual(
            by_stable_key(first.d, ordinary[3]),
            by_stable_key(second.d, permuted[3]),
        )
        np.testing.assert_array_equal(
            first.candidate_support, second.candidate_support
        )
        self.assertEqual(first.candidate_support[0], 0.0)

    def test_deadline_nnz_and_update_caps_retain_best_full_state(self):
        frame, q, warm, blocks, unions = _stall_problem()
        deadline = property_causal_block_duals(
            frame,
            q,
            warm,
            blocks=blocks,
            direction_unions=unions,
            deadline=time.monotonic() - 1.0,
        )
        self.assertTrue(deadline.deadline_reached)
        self.assertEqual(deadline.updates, 0)
        self.assertEqual(deadline.candidate_support[0], 0.5)
        np.testing.assert_array_equal(deadline.d, warm)

        capped_nnz = property_causal_block_duals(
            frame,
            q,
            warm,
            blocks=blocks,
            direction_unions=unions,
            nnz_cap=1,
        )
        self.assertTrue(capped_nnz.nnz_cap_reached)
        self.assertEqual(capped_nnz.candidate_support[0], 0.5)
        np.testing.assert_array_equal(capped_nnz.d, warm)

        capped_updates = property_causal_block_duals(
            frame,
            q,
            warm,
            blocks=blocks,
            direction_unions=unions,
            max_updates=0,
        )
        self.assertTrue(capped_updates.update_cap_reached)
        self.assertEqual(capped_updates.candidate_support[0], 0.5)
        np.testing.assert_array_equal(capped_updates.d, warm)

    def test_nonnegative_projection_and_flat_cycle_keep_best(self):
        frame, q, _, blocks, unions = _stall_problem()
        projected = property_causal_block_duals(
            frame,
            q,
            np.asarray([[-1.0, 0.0]], dtype=np.float64),
            blocks=blocks,
            direction_unions=unions,
        )
        self.assertGreaterEqual(projected.projection_count, 1)
        self.assertTrue(bool(np.all(projected.d >= 0.0)))
        self.assertEqual(projected.candidate_support[0], 0.0)

        flat_frame = OriginalFrameLP(
            A=sp.csr_matrix([[1.0, -1.0]], dtype=np.float64),
            rl=np.asarray([-np.inf], dtype=np.float64),
            ru=np.asarray([0.0], dtype=np.float64),
            lb=-np.ones(2, dtype=np.float64),
            ub=np.ones(2, dtype=np.float64),
            row_tags=(
                ConstraintRowTag(0, "ub", "flat:transfer", 0),
            ),
        )
        flat_blocks = (
            CausalDualBlock(
                "flat",
                "flat",
                (0,),
                ("flat:transfer:0",),
                (0, 1),
            ),
        )
        flat_unions = (
            CausalDirectionUnion("flat_path", ("flat",)),
        )
        flat = property_causal_block_duals(
            flat_frame,
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            blocks=flat_blocks,
            direction_unions=flat_unions,
            max_updates=8,
            max_zero_gain_updates=8,
        )
        self.assertEqual(flat.zero_gain_updates, 1)
        self.assertGreaterEqual(flat.cycle_rejections, 1)
        self.assertEqual(flat.candidate_support[0], 1.0)
        # The current search state traverses a flat face, but the returned state
        # remains the historical best rather than the last zero-gain iterate.
        np.testing.assert_array_equal(flat.d, np.zeros((1, 1)))
        self.assertFalse(flat.proof_authority)

    def test_warm_single_coordinate_can_decrease_existing_multiplier(self):
        frame = OriginalFrameLP(
            A=sp.csr_matrix([[1.0]], dtype=np.float64),
            rl=np.asarray([-np.inf], dtype=np.float64),
            ru=np.asarray([0.0], dtype=np.float64),
            lb=np.asarray([-1.0], dtype=np.float64),
            ub=np.asarray([1.0], dtype=np.float64),
            row_tags=(
                ConstraintRowTag(0, "ub", "warm:decrease", 0),
            ),
        )
        blocks = (
            CausalDualBlock(
                "warm",
                "warm",
                (0,),
                ("warm:decrease:0",),
                (0,),
            ),
        )
        candidate = property_causal_block_duals(
            frame,
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([[2.0]], dtype=np.float64),
            blocks=blocks,
            direction_unions=(),
        )
        self.assertEqual(candidate.coordinate_updates, 1)
        self.assertEqual(candidate.block_updates, 0)
        self.assertEqual(candidate.d[0, 0], 1.0)
        self.assertEqual(candidate.candidate_support[0], 0.0)
        self.assertFalse(candidate.proof_authority)


if __name__ == "__main__":
    unittest.main()
