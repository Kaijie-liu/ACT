#!/usr/bin/env python3
# ===- test_property_pairhull.py - exact PairHull toys -----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Soundness and tightness toys for the independent PairHull exact core."""

from __future__ import annotations

import copy
from fractions import Fraction
import itertools
import math
import random
import unittest

import numpy as np

from act.back_end.hybridz_tf.property_pairhull import (
    DEFAULT_PAIRHULL_DIRECTIONS,
    PropertyPairHullError,
    build_pairhull_projection,
    exact_pairhull_beta,
    verify_pairhull_receipt,
)


F = Fraction
ZERO = F(0)


def _intersection(left, right):
    determinant = left[0] * right[1] - left[1] * right[0]
    if determinant == 0:
        return None
    return (
        (left[2] * right[1] - left[1] * right[2]) / determinant,
        (left[0] * right[2] - left[2] * right[0]) / determinant,
    )


def _feasible(point, constraints):
    return all(
        row[0] * point[0] + row[1] * point[1] <= row[2]
        for row in constraints
    )


def _direct_value(point, q, slope):
    return (
        q[0] * max(point[0], ZERO)
        + q[1] * max(point[1], ZERO)
        - slope[0] * point[0]
        - slope[1] * point[1]
    )


def _axis_partition_oracle(projection, q, slope):
    """Independent oracle: polygon vertices plus its ReLU-axis crossings.

    This does not enumerate phases.  The axes partition a polygon into cells
    on which the objective is affine, so their boundary crossings together
    with the original polygon vertices contain an optimum.
    """

    constraints = projection.constraints
    candidates = set()
    for left, right in itertools.combinations(constraints, 2):
        point = _intersection(left, right)
        if point is not None and _feasible(point, constraints):
            candidates.add(point)

    axes = ((F(1), F(0), F(0)), (F(0), F(1), F(0)))
    for boundary in constraints:
        for axis in axes:
            point = _intersection(boundary, axis)
            if point is not None and _feasible(point, constraints):
                candidates.add(point)
    origin = (F(0), F(0))
    if _feasible(origin, constraints):
        candidates.add(origin)
    if not candidates:
        raise AssertionError("independent oracle found no candidate")
    return max(_direct_value(point, q, slope) for point in candidates)


class PropertyPairHullDecisiveToyTests(unittest.TestCase):
    def _ordered_residual_projection(self, *, offset, noise, joint):
        # z1=t, z2=t+offset+eps, t in [-1,1], |eps|<=noise.
        directions = (
            DEFAULT_PAIRHULL_DIRECTIONS
            if joint
            else ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        return build_pairhull_projection(
            center=(0, offset),
            generators=((1, 0), (1, noise)),
            directions=directions,
        )

    def _independent_upper_best(self, minimum_gap):
        # ReLU(z1) has secant z1/2+1/2.  An upper plane for
        # -ReLU(z2) is -alpha*z2, alpha in [0,1].  The support is
        # 1/2 + |1/2-alpha| - minimum_gap*alpha.  Its only interior kink is
        # alpha=1/2, so the endpoints and kink give the exact optimum.
        def independent_upper(alpha):
            return (
                F(1, 2)
                + abs(F(1, 2) - alpha)
                - minimum_gap * alpha
            )

        return min(
            independent_upper(alpha)
            for alpha in (F(0), F(1, 2), F(1))
        )

    def test_dyadic_pairhull_closes_independent_13_over_32_gap(self):
        # Fully binary-exact decisive toy:
        # z1=t, z2=t+1/4+eps, |eps|<=1/16.
        projection = self._ordered_residual_projection(
            offset=F(1, 4),
            noise=F(1, 16),
            joint=True,
        )
        result = exact_pairhull_beta(
            projection,
            q=(1, -1),
            candidate_slope=(0, 0),
        )
        self.assertEqual(result.beta_exact, F(0))
        self.assertEqual(result.beta_stored, 0.0)

        independent_best = self._independent_upper_best(F(3, 16))
        self.assertEqual(independent_best, F(13, 32))
        self.assertLess(result.beta_exact, independent_best)
        self.assertTrue(
            result.receipt["projection_uses_outward_stored_supports"]
        )
        self.assertFalse(
            result.receipt["candidate_slope_proof_authority"]
        )

    def test_original_one_fifth_toy_closes_81_over_200_gap(self):
        # Historical non-dyadic specification retained as an exact rational
        # toy; support storage is nevertheless audited outward as binary64.
        projection = self._ordered_residual_projection(
            offset=F(1, 5),
            noise=F(1, 100),
            joint=True,
        )
        result = exact_pairhull_beta(
            projection,
            q=(1, -1),
            candidate_slope=(0, 0),
        )
        self.assertEqual(result.beta_exact, F(0))
        self.assertEqual(
            self._independent_upper_best(F(19, 100)),
            F(81, 200),
        )

    def test_axis_only_negative_control_does_not_invent_correlation(self):
        projection = self._ordered_residual_projection(
            offset=F(1, 4),
            noise=F(1, 16),
            joint=False,
        )
        result = exact_pairhull_beta(
            projection,
            q=(1, -1),
            candidate_slope=(0, 0),
        )
        self.assertEqual(result.beta_exact, F(1))
        self.assertGreater(result.beta_exact, F(13, 32))

    def test_relu_axis_intersection_beats_every_original_polygon_vertex(self):
        # On the square, -2*ReLU(z1)+z1 = -|z1|.  Every original polygon
        # vertex has value -1, while each edge crossing z1=0 has value 0.
        # A vertex-only polygon audit would therefore be unsound.
        projection = build_pairhull_projection(
            center=(0, 0),
            generators=((1, 0), (0, 1)),
            directions=((1, 0), (-1, 0), (0, 1), (0, -1)),
        )
        q = (F(-2), F(0))
        slope = (F(-1), F(0))
        polygon_vertices = set()
        for left, right in itertools.combinations(
            projection.constraints, 2
        ):
            point = _intersection(left, right)
            if point is not None and _feasible(
                point, projection.constraints
            ):
                polygon_vertices.add(point)
        vertex_only = max(
            _direct_value(point, q, slope) for point in polygon_vertices
        )
        result = exact_pairhull_beta(
            projection,
            q=q,
            candidate_slope=slope,
        )
        self.assertEqual(vertex_only, F(-1))
        self.assertEqual(result.beta_exact, F(0))
        self.assertEqual(result.witness[0], F(0))

    def test_residual_correlation_tightens_without_global_lp(self):
        # z1=x+r, z2=x-r, x in [-1,1], r in [-1/16,1/16].
        joint = build_pairhull_projection(
            center=(0, 0),
            generators=((1, F(1, 16)), (1, -F(1, 16))),
            directions=DEFAULT_PAIRHULL_DIRECTIONS,
        )
        axis_only = build_pairhull_projection(
            center=(0, 0),
            generators=((1, F(1, 16)), (1, -F(1, 16))),
            directions=((1, 0), (-1, 0), (0, 1), (0, -1)),
        )
        joint_result = exact_pairhull_beta(
            joint, q=(1, -1), candidate_slope=(0, 0)
        )
        axis_result = exact_pairhull_beta(
            axis_only, q=(1, -1), candidate_slope=(0, 0)
        )
        self.assertEqual(joint_result.beta_exact, F(1, 8))
        self.assertEqual(axis_result.beta_exact, F(17, 16))
        self.assertLess(joint_result.beta_exact, axis_result.beta_exact)


class PropertyPairHullNumericalTests(unittest.TestCase):
    def test_rowwise_errors_are_independent_in_every_support(self):
        projection = build_pairhull_projection(
            center=(0, 0),
            generators=((), ()),
            error=(F(1, 4), F(1, 8)),
            directions=DEFAULT_PAIRHULL_DIRECTIONS,
        )
        supports = dict(zip(projection.directions, projection.supports))
        self.assertEqual(supports[(F(1), F(0))], F(1, 4))
        self.assertEqual(supports[(F(0), F(-1))], F(1, 8))
        # Opposite signs cannot cancel independent row errors.
        self.assertEqual(supports[(F(1), F(-1))], F(3, 8))
        self.assertEqual(projection.error, (F(1, 4), F(1, 8)))
        error_result = exact_pairhull_beta(
            projection,
            q=(1, 0),
            candidate_slope=(0, 0),
        )
        self.assertEqual(error_result.beta_exact, F(1, 4))

        without_error = build_pairhull_projection(
            center=(0, 0),
            generators=((), ()),
            directions=DEFAULT_PAIRHULL_DIRECTIONS,
        )
        self.assertNotEqual(
            projection.source_affine_sha256,
            without_error.source_affine_sha256,
        )

    def test_point_projection_and_exact_objective(self):
        projection = build_pairhull_projection(
            center=(F(3, 2), -F(1, 4)),
            generators=((), ()),
            directions=DEFAULT_PAIRHULL_DIRECTIONS,
        )
        result = exact_pairhull_beta(
            projection,
            q=(2, -3),
            candidate_slope=(F(1, 2), F(1, 4)),
        )
        self.assertEqual(result.beta_exact, F(37, 16))
        self.assertEqual(result.witness, (F(3, 2), -F(1, 4)))
        self.assertEqual(result.phase, (True, False))

    def test_tie_is_deterministic_across_phases_and_vertices(self):
        projection = build_pairhull_projection(
            center=(0, 0),
            generators=((1, 0), (0, 1)),
            directions=((1, 0), (-1, 0), (0, 1), (0, -1)),
        )
        first = exact_pairhull_beta(
            projection,
            q=(1, 1),
            candidate_slope=(F(1, 2), F(1, 2)),
        )
        second = exact_pairhull_beta(
            projection,
            q=(1, 1),
            candidate_slope=(F(1, 2), F(1, 2)),
        )
        self.assertEqual(first.beta_exact, F(1))
        self.assertEqual(first.phase, (False, False))
        self.assertEqual(first.witness, (F(-1), F(-1)))
        self.assertEqual(first.receipt, second.receipt)
        self.assertTrue(verify_pairhull_receipt(first.receipt))

    def test_half_subnormal_beta_is_stored_toward_positive_infinity(self):
        tiny = float(np.nextafter(0.0, np.inf))
        projection = build_pairhull_projection(
            center=(tiny, 0.0),
            generators=((), ()),
            directions=((1, 0), (-1, 0), (0, 1), (0, -1)),
        )
        result = exact_pairhull_beta(
            projection,
            q=(0.5, 0.0),
            candidate_slope=(0.0, 0.0),
        )
        self.assertEqual(
            result.beta_exact,
            Fraction.from_float(tiny) / 2,
        )
        self.assertEqual(result.beta_stored, tiny)
        self.assertGreater(
            Fraction.from_float(result.beta_stored),
            result.beta_exact,
        )

    def test_1e16_cancellation_is_evaluated_as_stored_fractions(self):
        projection = build_pairhull_projection(
            center=(1.0e16, 1.0),
            generators=((), ()),
            directions=((1, 0), (-1, 0), (0, 1), (0, -1)),
        )
        result = exact_pairhull_beta(
            projection,
            q=(1.0, 1.0),
            candidate_slope=(1.0, 0.0),
        )
        naive = max(1.0e16, 0.0) + max(1.0, 0.0) - 1.0e16
        self.assertEqual(naive, 0.0)
        self.assertEqual(result.beta_exact, F(1))
        self.assertEqual(result.beta_stored, 1.0)

    def test_every_projection_support_and_beta_are_outward(self):
        projection = build_pairhull_projection(
            center=(0.1, -0.2),
            generators=((0.3, -0.07), (0.11, 0.13)),
            directions=DEFAULT_PAIRHULL_DIRECTIONS,
        )
        self.assertTrue(
            all(
                Fraction.from_float(stored) >= required
                for stored, required in zip(
                    projection.stored_supports,
                    projection.required_supports,
                )
            )
        )
        result = exact_pairhull_beta(
            projection,
            q=(-0.7, 0.2),
            candidate_slope=(0.31, -0.19),
        )
        if math.isfinite(result.beta_stored):
            self.assertGreaterEqual(
                Fraction.from_float(result.beta_stored),
                result.beta_exact,
            )


class PropertyPairHullReceiptAndValidationTests(unittest.TestCase):
    def test_receipt_hash_detects_nested_tampering(self):
        projection = build_pairhull_projection(
            center=(0, F(1, 5)),
            generators=((1, 0), (1, F(1, 100))),
            directions=DEFAULT_PAIRHULL_DIRECTIONS,
        )
        result = exact_pairhull_beta(
            projection, q=(1, -1), candidate_slope=(0, 0)
        )
        self.assertTrue(verify_pairhull_receipt(result.receipt))
        tampered = copy.deepcopy(result.receipt)
        tampered["phase_records"][0]["vertices"] += 1
        self.assertFalse(verify_pairhull_receipt(tampered))

        changed = build_pairhull_projection(
            center=(0, F(1, 5)),
            generators=((1, 0), (1, F(1, 99))),
            directions=DEFAULT_PAIRHULL_DIRECTIONS,
        )
        self.assertNotEqual(
            projection.constraints_sha256,
            changed.constraints_sha256,
        )

    def test_invalid_or_unbounded_templates_are_rejected(self):
        with self.assertRaisesRegex(PropertyPairHullError, "bounded"):
            build_pairhull_projection(
                center=(0, 0),
                generators=((1,), (1,)),
                directions=((1, 0), (0, 1)),
            )
        with self.assertRaisesRegex(PropertyPairHullError, "zero"):
            build_pairhull_projection(
                center=(0, 0),
                generators=((1,), (1,)),
                directions=(
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (0, 0),
                ),
            )
        with self.assertRaisesRegex(PropertyPairHullError, "finite"):
            build_pairhull_projection(
                center=(math.nan, 0),
                generators=((1,), (1,)),
                directions=((1, 0), (-1, 0), (0, 1), (0, -1)),
            )


class PropertyPairHullSeededFuzzTests(unittest.TestCase):
    def test_300_seeded_rational_cases_match_axis_partition_oracle(self):
        rng = random.Random(0x5041495248554C4C)
        base_directions = list(DEFAULT_PAIRHULL_DIRECTIONS)
        phases_seen = set()
        for case in range(300):
            center = (
                F(rng.randint(-2, 2), 10),
                F(rng.randint(-2, 2), 10),
            )
            width0 = F(rng.randint(4, 7), 4)
            width1 = F(rng.randint(4, 7), 4)
            shared0 = F(rng.randint(-1, 1), 16)
            shared1 = F(rng.randint(-1, 1), 16)
            generators = (
                (width0, shared0),
                (shared1, width1),
            )
            error = (
                F(rng.randint(0, 2), 32),
                F(rng.randint(0, 2), 32),
            )
            directions = list(base_directions)
            while len(directions) < 12:
                direction = (
                    rng.randint(-4, 4),
                    rng.randint(-4, 4),
                )
                if direction != (0, 0):
                    directions.append(direction)
            projection = build_pairhull_projection(
                center=center,
                generators=generators,
                error=error,
                directions=directions,
            )
            q = (
                F(rng.randint(-5, 5), rng.randint(1, 8)),
                F(rng.randint(-5, 5), rng.randint(1, 8)),
            )
            slope = (
                F(rng.randint(-5, 5), rng.randint(1, 8)),
                F(rng.randint(-5, 5), rng.randint(1, 8)),
            )
            result = exact_pairhull_beta(
                projection,
                q=q,
                candidate_slope=slope,
            )
            oracle = _axis_partition_oracle(projection, q, slope)
            self.assertEqual(
                result.beta_exact,
                oracle,
                msg=f"seeded case {case}",
            )
            self.assertEqual(
                _direct_value(result.witness, q, slope),
                result.beta_exact,
                msg=f"witness case {case}",
            )
            self.assertTrue(
                _feasible(result.witness, projection.constraints),
                msg=f"feasibility case {case}",
            )
            self.assertEqual(result.receipt["phases_feasible"], 4)
            phases_seen.update(
                record["phase"]
                for record in result.receipt["phase_records"]
                if record["feasible"]
            )
            self.assertTrue(verify_pairhull_receipt(result.receipt))
        self.assertEqual(phases_seen, {"00", "01", "10", "11"})


if __name__ == "__main__":
    unittest.main()
