# ===- test_conditioned_difference_support.py - N1 tests --------------====#

import unittest

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.moe.conditioned_difference_support import (
    condition_on_affine_path_interval,
    conditioned_pair_difference_support,
)
from act.back_end.moe.weighted_top2 import shared_input_pair_hz
from act.back_end.solver.solver_hz import (
    hz_add_output_inequalities,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)


def _correlated_components(*, frame_id=701, guarded=False):
    entry = sparse_hz_from_bounds(
        Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
        frame_id=frame_id,
    )
    if guarded:
        entry = hz_add_output_inequalities(entry, [[1.0]], [0.75])
    router = sparse_hz_linear(entry, [[1.0], [-1.0]])
    expert_a = sparse_hz_linear(entry, [[1.0]])
    expert_b = sparse_hz_linear(entry, [[-1.0]])
    return entry, router, shared_input_pair_hz(entry, expert_a, expert_b)


class ConditionedDifferenceSupportTests(unittest.TestCase):
    def test_margin_segments_tighten_correlated_difference(self):
        _, router, pair = _correlated_components()
        result = conditioned_pair_difference_support(
            pair,
            router,
            (0, 1),
            [1.0],
            cut_points=(0.0,),
            margin_time_limit=2.0,
            feasibility_time_limit=2.0,
            difference_time_limit=2.0,
        )
        self.assertEqual(len(result.segments), 2)
        self.assertAlmostEqual(result.unconditional_bounds[0], -2.0, places=6)
        self.assertAlmostEqual(result.unconditional_bounds[1], 2.0, places=6)
        left, right = (segment.tightened_bounds for segment in result.segments)
        self.assertLessEqual(left[1], 1e-7)
        self.assertGreaterEqual(right[0], -1e-7)
        self.assertLessEqual(left[0], result.unconditional_bounds[0] + 1e-7)
        self.assertGreaterEqual(right[1], result.unconditional_bounds[1] - 1e-7)

    def test_closed_segments_cover_tie_in_both_branches(self):
        _, router, pair = _correlated_components()
        result = conditioned_pair_difference_support(
            pair,
            router,
            (0, 1),
            [1.0],
            cut_points=(0.0,),
            margin_time_limit=2.0,
            feasibility_time_limit=2.0,
            difference_time_limit=2.0,
        )
        tied = result.segment_for_value(0.0)
        self.assertEqual(len(tied), 2)
        self.assertTrue(result.closed_boundary_overlap)
        self.assertTrue(all(segment.feasibility.status == "feasible" for segment in tied))

    def test_sound_interval_union_contains_concrete_grid(self):
        _, router, pair = _correlated_components()
        result = conditioned_pair_difference_support(
            pair,
            router,
            (0, 1),
            [1.0],
            cut_points=(-0.5, 0.0, 0.75),
            margin_time_limit=2.0,
            feasibility_time_limit=2.0,
            difference_time_limit=2.0,
        )
        for value in np.linspace(-1.0, 1.0, 101):
            margin = 2.0 * value
            difference = 2.0 * value
            containing = result.segment_for_value(margin, tolerance=1e-8)
            self.assertTrue(containing)
            self.assertTrue(
                any(
                    segment.tightened_bounds[0] - 1e-7 <= difference
                    <= segment.tightened_bounds[1] + 1e-7
                    for segment in containing
                    if segment.tightened_bounds is not None
                )
            )

    def test_support_monotonicity_is_executable(self):
        _, router, pair = _correlated_components(guarded=True)
        result = conditioned_pair_difference_support(
            pair,
            router,
            (0, 1),
            [1.0],
            cut_points=(-0.25, 0.25),
            margin_time_limit=2.0,
            feasibility_time_limit=2.0,
            difference_time_limit=2.0,
        )
        lower, upper = result.unconditional_bounds
        for segment in result.segments:
            if segment.active:
                self.assertGreaterEqual(segment.tightened_bounds[0], lower)
                self.assertLessEqual(segment.tightened_bounds[1], upper)

    def test_zero_support_budget_falls_back_soundly_and_is_counted(self):
        _, router, pair = _correlated_components()
        result = conditioned_pair_difference_support(
            pair,
            router,
            (0, 1),
            [1.0],
            cut_points=(0.0,),
            margin_time_limit=0.0,
            feasibility_time_limit=2.0,
            difference_time_limit=0.0,
        )
        self.assertEqual(result.telemetry.fallback_segments, 2)
        self.assertEqual(result.telemetry.conditioned_support_solves, 0)
        self.assertEqual(result.union_bounds, result.unconditional_bounds)

    def test_n1_does_not_encode_or_segment_the_sigmoid(self):
        _, router, pair = _correlated_components()
        result = conditioned_pair_difference_support(
            pair,
            router,
            (0, 1),
            [1.0],
            cut_points=(0.0,),
            margin_time_limit=2.0,
            feasibility_time_limit=2.0,
            difference_time_limit=2.0,
        )
        self.assertEqual(result.telemetry.segmentation_axis, "affine_path_margin")
        self.assertFalse(result.telemetry.gate_function_encoded)
        self.assertEqual(result.telemetry.sigmoid_segments, 0)

    def test_same_frame_without_retained_constraints_is_rejected(self):
        entry, router, _ = _correlated_components(guarded=True)
        unguarded = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=entry.frame_id,
        )
        wrong_target = sparse_hz_linear(unguarded, [[1.0], [-1.0]])
        margin = sparse_hz_linear(router, [[1.0, -1.0]])
        with self.assertRaisesRegex(ValueError, "lost retained path constraints"):
            condition_on_affine_path_interval(wrong_target, margin, -1.0, 1.0)

    def test_distinct_frame_is_rejected(self):
        _, router, pair = _correlated_components()
        copied = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=999,
        )
        copied_pair = shared_input_pair_hz(
            copied,
            sparse_hz_linear(copied, [[1.0]]),
            sparse_hz_linear(copied, [[-1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "frame identity"):
            conditioned_pair_difference_support(
                copied_pair,
                router,
                (0, 1),
                [1.0],
                margin_time_limit=2.0,
                feasibility_time_limit=2.0,
                difference_time_limit=2.0,
            )


if __name__ == "__main__":
    unittest.main()
