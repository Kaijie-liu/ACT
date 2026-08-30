#!/usr/bin/env python3
# ===- test_property_cross_layer_residual_facet.py - exact toy gates ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Exact, independent gates for the toy-only cross-layer residual facet."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import random
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

import act.back_end.hybridz_tf.property_cross_layer_residual_facet as core
from act.back_end.hybridz_tf.property_cross_layer_residual_facet import (
    MAX_PAIR_BUDGET,
    PairProposal,
    ResidualFacet,
    ResidualJoinToy,
    bounded_pair_prefix,
    default_facet_binding,
    derive_cross_layer_residual_facet,
    enumerate_joint_phase_projection,
    evaluate_graph,
    exact_budget_frontier,
    exact_downstream_rcmph_upper,
    exact_graph_range,
    exact_layer_jacobian,
    exact_triangle_upper,
    raw_vnnlib_margin,
    uncorrelated_join_residuals,
    validate_cross_layer_residual_facet,
    validate_facet_against_original_phases,
)


Q = Fraction


def _independent_triangle_lp(*, add_facet: bool) -> tuple[float, np.ndarray]:
    """SciPy primal replay independent of the module's Fraction LP."""

    # Variables (x,y,v), z=y-x-1/2.  Both exact scalar preactivation
    # intervals are already used: x in [-1,1], z in [-1/2,1/2].
    A = np.asarray(
        [
            [0.0, -1.0, 0.0],       # y>=0
            [1.0, -1.0, 0.0],       # y>=x
            [-0.5, 1.0, 0.0],       # y<=(x+1)/2
            [0.0, 0.0, -1.0],       # v>=0
            [-1.0, 1.0, -1.0],      # v>=z
            [0.5, -0.5, 1.0],       # v<=z/2+1/4
        ],
        dtype=np.float64,
    )
    b = np.asarray([0.0, 0.0, 0.5, 0.0, 0.5, 0.0])
    if add_facet:
        # rho_y-rho_v<=1/4, expanded using
        # rho_y=y-x/2, rho_v=v-z/2, z=y-x-1/2.
        A = np.vstack([A, [-1.0, 1.5, -1.0]])
        b = np.concatenate([b, [0.5]])
    result = linprog(
        np.asarray([2.0, -3.0, 3.0]),  # minimize -q
        A_ub=A,
        b_ub=b,
        bounds=[(-1.0, 1.0), (0.0, 1.0), (0.0, 0.5)],
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun), np.asarray(result.x)


def _independent_downstream_bundle_lp() -> tuple[float, np.ndarray]:
    """Primal hypograph replay for the current downstream RC-MPH boundary."""

    # Variables (x,y,t).  t<=-2x+3y comes from v>=0;
    # t<=x+3/2 comes from v>=z.  The upstream ReLU remains its triangle.
    A = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-0.5, 1.0, 0.0],
            [2.0, -3.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    b = np.asarray([0.0, 0.0, 0.5, 0.0, 1.5])
    result = linprog(
        np.asarray([0.0, 0.0, -1.0]),
        A_ub=A,
        b_ub=b,
        bounds=[(-1.0, 1.0), (0.0, 1.0), (None, None)],
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun), np.asarray(result.x)


def _independent_exact_relu_milp() -> tuple[float, np.ndarray]:
    """Independent two-binary Big-M oracle for the original graph."""

    # Variables (x,y,z,v,b_y,b_v).  The two binary columns encode the two
    # exact ReLUs; z=y-x-1/2 is the residual join equality.
    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def append_upper(row, rhs):
        rows.append(list(row))
        lower.append(-np.inf)
        upper.append(float(rhs))

    def append_equal(row, rhs):
        rows.append(list(row))
        lower.append(float(rhs))
        upper.append(float(rhs))

    append_upper([1, -1, 0, 0, 0, 0], 0)       # y>=x
    append_upper([-1, 1, 0, 0, 1, 0], 1)       # y<=x+1(1-b_y)
    append_upper([0, 1, 0, 0, -1, 0], 0)       # y<=b_y
    append_equal([1, -1, 1, 0, 0, 0], -0.5)    # z=y-x-1/2
    append_upper([0, 0, 1, -1, 0, 0], 0)       # v>=z
    append_upper([0, 0, -1, 1, 0, 0.5], 0.5)   # v<=z+.5(1-b_v)
    append_upper([0, 0, 0, 1, 0, -0.5], 0)     # v<=.5*b_v
    result = milp(
        np.asarray([2.0, -3.0, 0.0, 3.0, 0.0, 0.0]),
        integrality=np.asarray([0, 0, 0, 0, 1, 1]),
        bounds=Bounds(
            [-1.0, 0.0, -0.5, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
        ),
        constraints=LinearConstraint(np.asarray(rows), lower, upper),
        options={"time_limit": 2.0},
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun), np.asarray(result.x)


class CrossLayerResidualFacetDecisiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.toy = ResidualJoinToy()
        self.binding = default_facet_binding()
        self.certificate = derive_cross_layer_residual_facet(
            self.toy, self.binding
        )

    def test_complete_four_phase_projection_derives_quarter_facet(self):
        phases = self.certificate.phases
        self.assertEqual(len(phases), 4)
        self.assertEqual(sum(phase.feasible for phase in phases), 3)
        self.assertEqual(
            tuple(
                (
                    phase.upstream_active,
                    phase.downstream_active,
                    phase.feasible,
                    phase.lower,
                    phase.upper,
                )
                for phase in phases
            ),
            (
                (False, False, True, Q(-1, 2), Q(0)),
                (False, True, True, Q(-1), Q(-1, 2)),
                (True, False, True, Q(0), Q(1)),
                (True, True, False, None, None),
            ),
        )
        self.assertEqual(
            self.certificate.hull_vertices,
            ((Q(0), Q(1, 4)), (Q(1, 4), Q(0)), (Q(1, 2), Q(1, 4))),
        )
        self.assertEqual(
            self.certificate.selected_facet,
            ResidualFacet(Q(1), Q(-1), Q(1, 4)),
        )
        self.assertEqual(
            self.certificate.relaxed_witness_residual,
            (Q(1, 2), Q(0)),
        )
        self.assertEqual(self.certificate.selected_violation, Q(1, 4))
        self.assertTrue(
            all(
                validate_facet_against_original_phases(self.toy, facet)
                for facet in self.certificate.hull_facets
            )
        )

    def test_baseline_bundle_facet_fraction_lp_and_milp_values(self):
        baseline = exact_triangle_upper(self.toy)
        bundle = exact_downstream_rcmph_upper(self.toy)
        tightened = exact_triangle_upper(
            self.toy, (self.certificate.selected_facet,)
        )
        exact = exact_graph_range(self.toy)
        self.assertEqual(baseline.upper, Q(3, 2))
        self.assertEqual(bundle.upper, Q(3, 2))
        self.assertEqual(tightened.upper, Q(1))
        self.assertEqual(exact, (Q(0), Q(1)))

        scipy_baseline, baseline_witness = _independent_triangle_lp(
            add_facet=False
        )
        scipy_facet, _ = _independent_triangle_lp(add_facet=True)
        scipy_bundle, bundle_witness = _independent_downstream_bundle_lp()
        scipy_milp, _ = _independent_exact_relu_milp()
        self.assertEqual(scipy_baseline, 1.5)
        self.assertEqual(scipy_bundle, 1.5)
        self.assertEqual(scipy_facet, 1.0)
        self.assertEqual(scipy_milp, 1.0)
        np.testing.assert_allclose(baseline_witness, [0.0, 0.5, 0.0])
        np.testing.assert_allclose(bundle_witness, [0.0, 0.5, 1.5])

        # Both legal downstream suffix planes equal 3/2 at the same upstream
        # triangle fake point.  A downstream-only plane bundle cannot remove
        # a point that its immutable prefix still admits.
        x, y = 0.0, 0.5
        self.assertEqual(-2 * x + 3 * y, 1.5)
        self.assertEqual(x + 1.5, 1.5)

    def test_existing_scalar_bounds_and_triangle_rows_do_not_imply_facet(self):
        self.assertEqual(
            (self.toy.input_lower, self.toy.input_upper),
            (Q(-1), Q(1)),
        )
        self.assertEqual(
            (self.toy.downstream_lower, self.toy.downstream_upper),
            (Q(-1, 2), Q(1, 2)),
        )
        self.assertLess(self.toy.input_lower, 0)
        self.assertGreater(self.toy.input_upper, 0)
        self.assertLess(self.toy.downstream_lower, 0)
        self.assertGreater(self.toy.downstream_upper, 0)
        baseline = exact_triangle_upper(self.toy)
        self.assertEqual(baseline.witness, (Q(0), Q(1, 2), Q(0)))
        facet = self.certificate.selected_facet
        self.assertGreater(
            facet.value(self.certificate.relaxed_witness_residual),
            facet.rhs,
        )

    def test_raw_vnnlib_assert_crosses_only_after_sound_tightening(self):
        threshold = Q(5, 4)
        exact_margin = max(
            raw_vnnlib_margin(point, threshold)
            for phase in enumerate_joint_phase_projection(self.toy)
            for point in phase.endpoints
        )
        baseline, tightened = exact_budget_frontier(
            self.toy, self.binding, self.certificate
        )
        self.assertEqual(exact_margin, Q(-1, 4))
        self.assertEqual(baseline.upper - threshold, Q(1, 4))
        self.assertEqual(tightened.upper - threshold, Q(-1, 4))

    def test_complete_ideal_k2_phase_hull_matches_not_loses_to_facet(self):
        # This is the explicit dominance boundary: an ideal PCOH/phase hull
        # with exact conditional maxima also obtains 1 on this toy.  The new
        # object is a reusable projection/compression, not stronger geometry.
        phase_uppers = tuple(
            max(point.q for point in phase.endpoints)
            for phase in self.certificate.phases
            if phase.feasible
        )
        self.assertEqual(phase_uppers, (Q(1), Q(1), Q(1)))
        facet_upper = exact_triangle_upper(
            self.toy, (self.certificate.selected_facet,)
        ).upper
        self.assertEqual(max(phase_uppers), facet_upper)


class CrossLayerResidualFacetGraphAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.toy = ResidualJoinToy()

    def test_point_consistency_layer_widths_and_exact_values(self):
        self.assertEqual(
            self.toy.layer_widths,
            (
                ("input:x", 1),
                ("relu:y", 1),
                ("skip:-x-theta", 1),
                ("add:z", 1),
                ("relu:v", 1),
                ("property:q", 1),
            ),
        )
        expected = {
            Q(-1): (Q(0), Q(1, 2), Q(1, 2), Q(1, 2)),
            Q(-1, 2): (Q(0), Q(0), Q(0), Q(1)),
            Q(0): (Q(0), Q(-1, 2), Q(0), Q(0)),
            Q(1): (Q(1), Q(-1, 2), Q(0), Q(1)),
        }
        for x, values in expected.items():
            point = evaluate_graph(self.toy, x)
            self.assertEqual((point.y, point.z, point.v, point.q), values)
            self.assertEqual(point.skip, -x - Q(1, 2))
            self.assertEqual(point.z, point.y + point.skip)
            self.assertEqual(point.y, max(Q(0), point.x))
            self.assertEqual(point.v, max(Q(0), point.z))

    def test_exact_piecewise_jacobian(self):
        by_region = {
            Q(-3, 4): {
                "relu:y": Q(0),
                "skip:-x-theta": Q(-1),
                "add:z": Q(-1),
                "relu:v": Q(-1),
                "property:q": Q(1),
            },
            Q(-1, 4): {
                "relu:y": Q(0),
                "skip:-x-theta": Q(-1),
                "add:z": Q(-1),
                "relu:v": Q(0),
                "property:q": Q(-2),
            },
            Q(1, 2): {
                "relu:y": Q(1),
                "skip:-x-theta": Q(-1),
                "add:z": Q(0),
                "relu:v": Q(0),
                "property:q": Q(1),
            },
        }
        for point, expected in by_region.items():
            observed = dict(exact_layer_jacobian(self.toy, point))
            self.assertEqual(observed["input:x"], Q(1))
            for layer, derivative in expected.items():
                self.assertEqual(observed[layer], derivative)
        for kink in (Q(-1, 2), Q(0)):
            with self.assertRaisesRegex(ValueError, "set-valued"):
                exact_layer_jacobian(self.toy, kink)

    def test_seeded_dyadic_dags_contain_every_sampled_graph_point(self):
        rng = random.Random(0xC20520260809)
        cases = points = 0
        for _ in range(64):
            theta = Q(rng.randint(1, 31), 32)
            toy = ResidualJoinToy(theta=theta)
            phases = enumerate_joint_phase_projection(toy)
            self.assertEqual(len(phases), 4)
            self.assertEqual(sum(phase.feasible for phase in phases), 3)
            # Derive through the complete phase oracle, not sampled points.
            from act.back_end.hybridz_tf.property_cross_layer_residual_facet import (
                exact_convex_hull_2d,
                exact_hull_facets,
                phase_projection_vertices,
            )

            hull = exact_convex_hull_2d(phase_projection_vertices(phases))
            facets = exact_hull_facets(hull)
            self.assertEqual(len(hull), 3)
            self.assertEqual(len(facets), 3)
            for step in range(-32, 33):
                point = evaluate_graph(toy, Q(step, 32))
                residual = (point.rho_upstream, point.rho_downstream)
                self.assertTrue(all(facet.contains(residual) for facet in facets))
                points += 1
            cases += 1
        self.assertEqual(cases, 64)
        self.assertEqual(points, 64 * 65)


class CrossLayerResidualFacetBindingAndBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.toy = ResidualJoinToy()
        self.binding = default_facet_binding()
        self.certificate = derive_cross_layer_residual_facet(
            self.toy, self.binding
        )

    def test_wrong_stable_id_row_tag_and_receipt_tamper_fail_closed(self):
        self.assertTrue(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, self.certificate
            )
        )
        wrong_id = replace(
            self.binding, upstream_output_stable_id=999
        )
        wrong_tag = replace(
            self.binding, downstream_row_tag="relu:v:wrong_copy"
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, wrong_id, self.certificate
            )
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, wrong_tag, self.certificate
            )
        )
        tampered = replace(
            self.certificate,
            selected_violation=self.certificate.selected_violation + Q(1, 64),
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, tampered
            )
        )

    def test_bad_binding_cannot_fresh_self_sign_or_survive_aba(self):
        import act.back_end.hybridz_tf.property_cross_layer_residual_facet as core

        bad = replace(
            self.binding,
            upstream_output_stable_id=999,
            join_row_tag="add:z:self_signed_wrong_copy",
        )
        with self.assertRaisesRegex(ValueError, "canonical causal anchor"):
            derive_cross_layer_residual_facet(self.toy, bad)

        # Re-sign every public binding field coherently.  Validation must be
        # anchored to the canonical causal source rather than trusting this
        # internally consistent attacker-owned digest.
        bad_payload = core._certificate_payload(
            self.toy,
            bad,
            self.certificate.phases,
            self.certificate.hull_vertices,
            self.certificate.hull_facets,
            self.certificate.selected_facet,
            self.certificate.relaxed_witness_residual,
            self.certificate.selected_violation,
        )
        self_signed = replace(
            self.certificate,
            binding_digest=bad.semantic_digest,
            receipt_sha256=core._digest(bad_payload),
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(self.toy, bad, self_signed)
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, self_signed
            )
        )

        # ABA mutation: restoring the public digest while retaining the
        # coherently signed B-state receipt cannot resurrect the candidate.
        aba = replace(
            self_signed,
            binding_digest=self.binding.semantic_digest,
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(self.toy, self.binding, aba)
        )

    def test_binding_consumer_rejects_value_equal_nonexact_types(self):
        class StrSubclass(str):
            pass

        mutations = (
            ("upstream_preactivation_stable_id", 101.0),
            ("upstream_preactivation_stable_id", True),
            (
                "join_row_tag",
                StrSubclass("add:z:y_plus_same_x_skip"),
            ),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=repr(value)):
                poisoned = default_facet_binding()
                object.__setattr__(poisoned, field, value)
                with self.assertRaisesRegex(
                    ValueError, "canonical causal anchor"
                ):
                    derive_cross_layer_residual_facet(self.toy, poisoned)
                self.assertFalse(
                    validate_cross_layer_residual_facet(
                        self.toy, poisoned, self.certificate
                    )
                )

    def test_coefficient_sign_is_replayed_against_all_original_phases(self):
        good = ResidualFacet(Q(1), Q(-1), Q(1, 4))
        bad = ResidualFacet(Q(1), Q(1), Q(1, 4))
        self.assertTrue(validate_facet_against_original_phases(self.toy, good))
        self.assertFalse(validate_facet_against_original_phases(self.toy, bad))
        tampered = replace(self.certificate, selected_facet=bad)
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, tampered
            )
        )

    def test_residual_facet_requires_exact_nonboolean_nonzero_state(self):
        for values in (
            (True, Q(-1), Q(1, 4)),
            (Q(1), False, Q(1, 4)),
            (Q(1), Q(-1), True),
            (Q(0), Q(0), Q(1)),
            (1.0, Q(-1), Q(1, 4)),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ResidualFacet(*values)

        corrupted = ResidualFacet(Q(1), Q(-1), Q(1, 4))
        object.__setattr__(corrupted, "upstream_coefficient", True)
        self.assertFalse(
            validate_facet_against_original_phases(self.toy, corrupted)
        )
        with self.assertRaisesRegex(ValueError, "state is malformed"):
            exact_triangle_upper(self.toy, (corrupted,))

        # This exact-valued row is not valid on the original joint phases and
        # would under-approximate the true upper from 1 to 0 if consumed.
        invalid = ResidualFacet(Q(1, 4), Q(-2), Q(-1, 2))
        self.assertFalse(
            validate_facet_against_original_phases(self.toy, invalid)
        )
        with self.assertRaisesRegex(ValueError, "not valid"):
            exact_triangle_upper(self.toy, (invalid,))

        mutated_invalid = ResidualFacet(Q(1), Q(-1), Q(1, 4))
        object.__setattr__(
            mutated_invalid, "upstream_coefficient", Q(1, 4)
        )
        object.__setattr__(
            mutated_invalid, "downstream_coefficient", Q(-2)
        )
        object.__setattr__(mutated_invalid, "rhs", Q(-1, 2))
        with self.assertRaisesRegex(ValueError, "not valid"):
            exact_triangle_upper(self.toy, (mutated_invalid,))

    def test_deleting_causal_join_invalidates_facet_and_binding(self):
        facet = self.certificate.selected_facet
        # Same shapes and scalar bounds, but upstream and skip consume fresh
        # copies p=1 and q=1/2.  Then z=0, rho_y=1/2, rho_v=0.
        wrong_copy = uncorrelated_join_residuals(
            self.toy, Q(1), Q(1, 2)
        )
        self.assertEqual(wrong_copy, (Q(1, 2), Q(0)))
        self.assertFalse(facet.contains(wrong_copy))
        deleted_join = replace(
            self.binding, join_row_tag="add:z:deleted_causal_equality"
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, deleted_join, self.certificate
            )
        )

    def test_budget_prefix_is_nested_capped_and_bound_monotone(self):
        proposals = tuple(
            PairProposal((10 + index, 20 + index), Q(8 - index, 8))
            for index in range(5)
        )
        prefixes = tuple(
            bounded_pair_prefix(proposals, budget)
            for budget in range(MAX_PAIR_BUDGET + 1)
        )
        self.assertEqual([len(prefix) for prefix in prefixes], [0, 1, 2, 3, 4])
        self.assertTrue(
            all(
                left == right[: len(left)]
                for left, right in zip(prefixes, prefixes[1:])
            )
        )
        with self.assertRaisesRegex(ValueError, r"\[0,4\]"):
            bounded_pair_prefix(proposals, 5)

        baseline, tightened = exact_budget_frontier(
            self.toy, self.binding, self.certificate
        )
        self.assertEqual((baseline.upper, tightened.upper), (Q(3, 2), Q(1)))
        self.assertLessEqual(tightened.upper, baseline.upper)

    def test_pair_proposals_are_exact_canonical_undirected_and_rechecked(self):
        for key, score in (
            ([1, 2], Q(1)),
            ((2, 1), Q(1)),
            ((1, 1), Q(1)),
            ((True, 2), Q(1)),
            ((1, 2), True),
            ((1, 2), -1),
        ):
            with self.subTest(key=key, score=score):
                with self.assertRaises(ValueError):
                    PairProposal(key, score)

        negative = PairProposal((1, 2), Q(1))
        object.__setattr__(negative, "score", Q(-1))
        with self.assertRaisesRegex(ValueError, "state is malformed"):
            bounded_pair_prefix((negative,), 1)

        reversed_pair = PairProposal((3, 4), Q(1))
        object.__setattr__(reversed_pair, "pair_key", (4, 3))
        with self.assertRaisesRegex(ValueError, "state is malformed"):
            bounded_pair_prefix((reversed_pair,), 1)

        first = PairProposal((5, 6), Q(1))
        reversed_duplicate = PairProposal((7, 8), Q(1, 2))
        object.__setattr__(reversed_duplicate, "pair_key", (6, 5))
        with self.assertRaisesRegex(ValueError, "state is malformed"):
            bounded_pair_prefix((first, reversed_duplicate), 2)

        duplicate = PairProposal((7, 8), Q(1, 2))
        object.__setattr__(duplicate, "pair_key", (5, 6))
        with self.assertRaisesRegex(ValueError, "repeat"):
            bounded_pair_prefix((first, duplicate), 2)

    def test_pair_budget_snapshots_one_iteration_before_sorting(self):
        first = PairProposal((1, 2), Q(1))
        second = PairProposal((3, 4), Q(1, 2))

        class ThirdIterationAttack:
            def __init__(self):
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations >= 2:
                    object.__setattr__(first, "score", Q(-1))
                    object.__setattr__(second, "pair_key", (4, 3))
                return iter((first, second))

        hostile = ThirdIterationAttack()
        selected = bounded_pair_prefix(hostile, 2)
        self.assertEqual(hostile.iterations, 1)
        self.assertEqual(
            tuple((item.pair_key, item.score) for item in selected),
            (((1, 2), Q(1)), ((3, 4), Q(1, 2))),
        )

        after_yield_first = PairProposal((5, 6), Q(3, 4))
        after_yield_second = PairProposal((7, 8), Q(1, 4))

        class MutateAfterYield:
            def __iter__(self):
                yield after_yield_first
                object.__setattr__(after_yield_first, "score", Q(-9))
                object.__setattr__(after_yield_first, "pair_key", (6, 5))
                yield after_yield_second

        snapshotted = bounded_pair_prefix(MutateAfterYield(), 2)
        self.assertEqual(
            tuple((item.pair_key, item.score) for item in snapshotted),
            (((5, 6), Q(3, 4)), ((7, 8), Q(1, 4))),
        )

    def test_certificate_deep_exact_state_rejects_numeric_equal_attacks(self):
        def fresh():
            return derive_cross_layer_residual_facet(self.toy, self.binding)

        certificate = fresh()
        object.__setattr__(certificate.phases[0], "upstream_active", 0)
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, certificate
            )
        )

        certificate = fresh()
        object.__setattr__(
            certificate.phases[0],
            "lower",
            float(certificate.phases[0].lower),
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, certificate
            )
        )

        certificate = fresh()
        object.__setattr__(
            certificate.phases[0].endpoints[0],
            "x",
            float(certificate.phases[0].endpoints[0].x),
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, certificate
            )
        )

        certificate = fresh()
        first = certificate.hull_vertices[0]
        object.__setattr__(
            certificate,
            "hull_vertices",
            ((float(first[0]), first[1]), *certificate.hull_vertices[1:]),
        )
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, certificate
            )
        )

        certificate = fresh()
        object.__setattr__(certificate, "proof_authority", 0)
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, certificate
            )
        )

        for field, value in (
            ("upstream_coefficient", True),
            ("upstream_coefficient", 1.0),
            ("rhs", 0.25),
        ):
            with self.subTest(field=field, value=value):
                certificate = fresh()
                # selected_facet is deliberately shared with hull_facets;
                # old dataclass equality treated these values as Fraction-
                # equal on both paths and accepted the unchanged SHA.
                object.__setattr__(certificate.selected_facet, field, value)
                self.assertFalse(
                    validate_cross_layer_residual_facet(
                        self.toy, self.binding, certificate
                    )
                )

    def test_certificate_validation_uses_one_private_deep_snapshot(self):
        certificate = derive_cross_layer_residual_facet(
            self.toy, self.binding
        )
        original_derive = derive_cross_layer_residual_facet

        def derive_then_mutate_caller(toy, binding):
            expected = original_derive(toy, binding)
            object.__setattr__(
                certificate.selected_facet,
                "upstream_coefficient",
                True,
            )
            return expected

        with patch.object(
            core,
            "derive_cross_layer_residual_facet",
            side_effect=derive_then_mutate_caller,
        ):
            # The validation result is bound to the exact entry snapshot, not
            # to a numerically equal live object changed during re-derivation.
            self.assertTrue(
                validate_cross_layer_residual_facet(
                    self.toy, self.binding, certificate
                )
            )

        # A later call snapshots the now-malformed public object and rejects it.
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, certificate
            )
        )

    def test_certificate_is_permanently_non_authoritative(self):
        self.assertFalse(self.certificate.proof_authority)
        self.assertFalse(self.certificate.verdict_authority)
        promoted = replace(self.certificate, proof_authority=True)
        self.assertFalse(
            validate_cross_layer_residual_facet(
                self.toy, self.binding, promoted
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
