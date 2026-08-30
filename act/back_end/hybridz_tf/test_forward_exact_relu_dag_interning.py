#!/usr/bin/env python3
# ===- test_forward_exact_relu_dag_interning.py - exact DAG gates --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Independent exact gates for the toy-only DAG-EPIC MVP."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import statistics
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from act.back_end.hybridz_tf import forward_exact_relu_dag_interning as core
from act.back_end.hybridz_tf.forward_exact_relu_dag_interning import (
    ExactAffineForm,
    ExactConstraintArena,
    ExactConstraintRow,
    ExactDAGRowView,
    ExactReLUInterner,
    RAW_MARGIN,
    build_fixed_residual_fanout_toy,
    build_synthetic_residual_family,
    enumerate_phase_projections,
    evaluate_fixed_toy,
    exact_fixed_toy_jacobian,
    projection_probe_points,
    projection_values_at,
    raw_unsafe_margins,
    raw_vnnlib_assert,
)


Q = Fraction


class _EqualString(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def _add_upper(rows, lower, upper, coefficients, rhs):
    rows.append(np.asarray(coefficients, dtype=np.float64))
    lower.append(-np.inf)
    upper.append(float(rhs))


def _add_equality(rows, lower, upper, coefficients, rhs=0.0):
    rows.append(np.asarray(coefficients, dtype=np.float64))
    lower.append(float(rhs))
    upper.append(float(rhs))


def _add_exact_relu(rows, lower, upper, *, width, y, pre, constant, lo, hi, z):
    """Independent exact 0/1 big-M graph; never a triangle relaxation."""

    def vector():
        return np.zeros(width, dtype=np.float64)

    row = vector()                       # y >= 0
    row[y] = -1.0
    _add_upper(rows, lower, upper, row, 0.0)

    row = vector()                       # y >= pre
    for index, value in pre.items():
        row[index] += float(value)
    row[y] -= 1.0
    _add_upper(rows, lower, upper, row, -float(constant))

    row = vector()                       # y <= pre-lo*(1-z)
    row[y] = 1.0
    for index, value in pre.items():
        row[index] -= float(value)
    row[z] -= float(lo)
    _add_upper(rows, lower, upper, row, float(constant - lo))

    row = vector()                       # y <= hi*z
    row[y] = 1.0
    row[z] = -float(hi)
    _add_upper(rows, lower, upper, row, 0.0)


def _independent_milp_max(*, compact: bool, objective) -> float:
    # Common network columns are x,a,l,r,b.  Baseline then has four physical
    # phase binaries; compact has two representative phase binaries.
    width = 7 if compact else 9
    x, a, left, right, b = range(5)
    rows = []
    lower = []
    upper = []
    if compact:
        phase_a, phase_t = 5, 6
        _add_exact_relu(
            rows, lower, upper, width=width, y=a, pre={x: 1},
            constant=Q(0), lo=Q(-1), hi=Q(1), z=phase_a,
        )
        _add_exact_relu(
            rows, lower, upper, width=width, y=left, pre={a: 1},
            constant=Q(-1, 2), lo=Q(-1, 2), hi=Q(1, 2), z=phase_t,
        )
        row = np.zeros(width)
        row[right], row[left] = 1.0, -1.0
        _add_equality(rows, lower, upper, row)
        row = np.zeros(width)
        row[b], row[a] = 1.0, -1.0
        _add_equality(rows, lower, upper, row)
        integrality = np.asarray([0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    else:
        phase_a, phase_left, phase_right, phase_b = 5, 6, 7, 8
        _add_exact_relu(
            rows, lower, upper, width=width, y=a, pre={x: 1},
            constant=Q(0), lo=Q(-1), hi=Q(1), z=phase_a,
        )
        for output, phase in ((left, phase_left), (right, phase_right)):
            _add_exact_relu(
                rows, lower, upper, width=width, y=output, pre={a: 1},
                constant=Q(-1, 2), lo=Q(-1, 2), hi=Q(1, 2), z=phase,
            )
        _add_exact_relu(
            rows, lower, upper, width=width, y=b, pre={x: 1},
            constant=Q(0), lo=Q(-1), hi=Q(1), z=phase_b,
        )
        integrality = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)

    objective_vector = np.zeros(width, dtype=np.float64)
    objective_vector[:5] = np.asarray([float(value) for value in objective])
    bounds = Bounds(
        [-1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * (width - 5),
        [1.0, 1.0, 0.5, 0.5, 1.0] + [1.0] * (width - 5),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected: .*mip_feasibility_tolerance.*",
            category=RuntimeWarning,
        )
        result = milp(
            -objective_vector,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(np.asarray(rows), lower, upper),
            options={
                "time_limit": 2.0,
                "mip_rel_gap": 0.0,
                "mip_feasibility_tolerance": 1.0e-10,
            },
        )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun)


def _fraction_objective_max(objective) -> Fraction:
    values = []
    for x in (Q(-1), Q(0), Q(1, 2), Q(1)):
        point = evaluate_fixed_toy(x)
        coordinates = (point.x, point.a, point.left, point.right, point.b)
        values.append(sum((coefficient * value for coefficient, value in zip(objective, coordinates)), Q(0)))
    return max(values)


class ForwardExactReLUDAGInterningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = build_fixed_residual_fanout_toy(compact=False)
        self.compact = build_fixed_residual_fanout_toy(compact=True)

    def test_fixed_structural_gate_is_twofold_or_better(self):
        self.assertEqual(
            (
                self.baseline.phase_handle_count,
                self.baseline.phase_assignment_count,
                self.baseline.unique_row_count,
                self.baseline.unique_nnz,
                self.baseline.physical_row_count,
                self.baseline.physical_nnz,
            ),
            (4, 16, 12, 32, 12, 32),
        )
        self.assertEqual(
            (
                self.compact.phase_handle_count,
                self.compact.phase_assignment_count,
                self.compact.unique_row_count,
                self.compact.unique_nnz,
                self.compact.physical_row_count,
                self.compact.physical_nnz,
            ),
            (2, 4, 6, 16, 6, 16),
        )
        self.assertGreaterEqual(
            self.baseline.unique_row_count / self.compact.unique_row_count,
            1.5,
        )
        self.assertGreaterEqual(
            self.baseline.unique_nnz / self.compact.unique_nnz,
            1.5,
        )
        handles = dict(self.compact.node_handle_digests)
        self.assertEqual(handles["a"], handles["b"])
        self.assertEqual(handles["l"], handles["r"])
        self.assertEqual(self.compact.exact_output_form.constant, Q(0))
        self.assertEqual(self.compact.exact_output_form.continuous_terms, ())
        self.assertFalse(self.compact.proof_authority)
        self.assertFalse(self.compact.verdict_authority)
        for field in ("proof_authority", "verdict_authority"):
            with self.assertRaises(AttributeError):
                object.__setattr__(self.compact, field, True)
            self.assertIs(getattr(self.compact, field), False)
        shadowed = replace(self.compact)
        shadowed.__dict__["proof_authority"] = True
        shadowed.__dict__["verdict_authority"] = True
        self.assertIs(shadowed.proof_authority, False)
        self.assertIs(shadowed.verdict_authority, False)
        with self.assertRaises(ValueError):
            replace(self.compact, receipt_sha256="0" * 64)
        self.assertEqual(
            self.compact.dag_row_view.row_ids,
            frozenset(row.stable_row_id for row in self.compact.physical_rows),
        )

    def test_point_consistency_and_every_layer_width(self):
        expected = {
            Q(-1): (Q(0), Q(-1, 2), Q(0), Q(-1), Q(0)),
            Q(0): (Q(0), Q(-1, 2), Q(0), Q(0), Q(0)),
            Q(1, 4): (Q(1, 4), Q(-1, 4), Q(0), Q(1, 4), Q(0)),
            Q(1, 2): (Q(1, 2), Q(0), Q(0), Q(1, 2), Q(0)),
            Q(3, 4): (Q(3, 4), Q(1, 4), Q(1, 4), Q(3, 4), Q(0)),
            Q(1): (Q(1), Q(1, 2), Q(1, 2), Q(1), Q(0)),
        }
        for x, values in expected.items():
            point = evaluate_fixed_toy(x)
            self.assertEqual(
                (point.a, point.t_left, point.left, point.joined, point.q),
                values,
            )
            self.assertEqual(point.t_left, point.t_right)
            self.assertEqual(point.left, point.right)
            self.assertEqual(point.a, point.b)
            self.assertEqual(point.joined, point.x)
        # Network layer widths remain scalar; only encoder-private columns fall.
        self.assertTrue(all(len((getattr(evaluate_fixed_toy(Q(1, 4)), name),)) == 1 for name in (
            "x", "a", "t_left", "left", "t_right", "right", "skip", "joined", "b", "q"
        )))

    def test_piecewise_fraction_jacobian_is_unchanged(self):
        expected = {
            Q(-1, 2): (Q(0), Q(0), Q(0), Q(1), Q(0), Q(0)),
            Q(1, 4): (Q(1), Q(0), Q(0), Q(1), Q(1), Q(0)),
            Q(3, 4): (Q(1), Q(1), Q(1), Q(1), Q(1), Q(0)),
        }
        for x, wanted in expected.items():
            observed = dict(exact_fixed_toy_jacobian(x))
            self.assertEqual(
                (
                    observed["a"], observed["left"], observed["right"],
                    observed["joined"], observed["b"], observed["q"],
                ),
                wanted,
            )
        for kink in (Q(0), Q(1, 2)):
            with self.assertRaises(ValueError):
                exact_fixed_toy_jacobian(kink)

    def test_complete_16_to_4_phase_projection_is_exact(self):
        baseline = enumerate_phase_projections(compact=False)
        compact = enumerate_phase_projections(compact=True)
        self.assertEqual((len(baseline), len(compact)), (16, 4))
        self.assertEqual(
            (sum(item.feasible for item in baseline), sum(item.feasible for item in compact)),
            (7, 3),
        )
        for x in projection_probe_points(baseline, compact):
            baseline_values = projection_values_at(baseline, x)
            compact_values = projection_values_at(compact, x)
            self.assertEqual(baseline_values, compact_values)
            self.assertEqual(len(baseline_values), 1)
            self.assertEqual(baseline_values[0][-1], Q(0))
        # Both representative phase choices remain feasible at each kink.
        at_zero = {
            item.assignment[0]
            for item in compact
            if item.feasible and item.lower <= 0 <= item.upper
        }
        at_half = {
            item.assignment[1]
            for item in compact
            if item.feasible and item.lower <= Q(1, 2) <= item.upper
        }
        self.assertEqual(at_zero, {0, 1})
        self.assertEqual(at_half, {0, 1})

    def test_fraction_milp_and_raw_assert_agree_exactly(self):
        q_objective = (Q(0), Q(1), Q(1), Q(-1), Q(-1))
        minus_q = tuple(-value for value in q_objective)
        for objective in (q_objective, minus_q):
            exact = _fraction_objective_max(objective)
            self.assertEqual(exact, Q(0))
            self.assertAlmostEqual(
                _independent_milp_max(compact=False, objective=objective),
                float(exact),
                places=10,
            )
            self.assertAlmostEqual(
                _independent_milp_max(compact=True, objective=objective),
                float(exact),
                places=10,
            )
        self.assertEqual(raw_unsafe_margins(evaluate_fixed_toy(Q(-1))), (-RAW_MARGIN, -RAW_MARGIN))
        self.assertEqual(raw_unsafe_margins(evaluate_fixed_toy(Q(1))), (-RAW_MARGIN, -RAW_MARGIN))
        self.assertEqual(
            raw_vnnlib_assert(),
            "(assert (or (>= Y_0 0.0625) (<= Y_0 -0.0625)))",
        )
        self.assertLess(_independent_milp_max(compact=False, objective=q_objective), float(RAW_MARGIN))
        self.assertLess(_independent_milp_max(compact=True, objective=minus_q), float(RAW_MARGIN))

    def test_seeded_fraction_objectives_match_both_exact_milps(self):
        rng = np.random.default_rng(0xDACE91C)
        for _ in range(32):
            objective = tuple(Q(int(value), 8) for value in rng.integers(-16, 17, size=5))
            exact = _fraction_objective_max(objective)
            baseline = _independent_milp_max(compact=False, objective=objective)
            compact = _independent_milp_max(compact=True, objective=objective)
            self.assertAlmostEqual(baseline, float(exact), places=9)
            self.assertAlmostEqual(compact, float(exact), places=9)
            self.assertAlmostEqual(baseline, compact, places=9)

    def test_exact_predicate_key_reuses_only_identical_kappa_one_state(self):
        arena = ExactConstraintArena()
        interner = ExactReLUInterner(
            arena,
            reuse_predicates=True,
            next_continuous_id=100,
            next_binary_id=200,
            next_row_id=300,
        )
        predicate = ExactAffineForm(Q(0), ((1, Q(1)),))
        first = interner.intern(predicate, -1, 1, semantic_node_id=10)
        first_digest = first.handle.handle_digest
        object.__setattr__(first.handle, "representative_node_id", 999)
        duplicate = interner.intern(predicate, -1, 1, semantic_node_id=11)
        shifted = interner.intern(predicate.shift(Q(1, 4)), -1, 1, semantic_node_id=12)
        negated = interner.intern(predicate.scale(-1), -1, 1, semantic_node_id=13)
        different_bounds = interner.intern(predicate, -2, 2, semantic_node_id=14)
        self.assertTrue(first.created)
        self.assertFalse(duplicate.created)
        self.assertEqual(first_digest, duplicate.handle.handle_digest)
        self.assertEqual(duplicate.handle.representative_node_id, 10)
        with self.assertRaises(ValueError):
            replace(duplicate.handle, representative_node_id=11)
        for field in ("proof_authority", "verdict_authority"):
            with self.assertRaises(AttributeError):
                object.__setattr__(duplicate.handle, field, True)
            self.assertIs(getattr(duplicate.handle, field), False)
        self.assertTrue(shifted.created)
        self.assertTrue(negated.created)
        self.assertTrue(different_bounds.created)
        self.assertEqual(arena.row_count, 12)

    def test_forced_digest_collisions_never_define_predicate_or_row_truth(self):
        forced_digest = "a" * 64
        with patch.object(core, "_digest", return_value=forced_digest):
            arena = ExactConstraintArena()
            interner = ExactReLUInterner(
                arena,
                reuse_predicates=True,
                next_continuous_id=100,
                next_binary_id=200,
                next_row_id=300,
            )
            positive = ExactAffineForm(Q(0), ((1, Q(1)),))
            negative = ExactAffineForm(Q(0), ((1, Q(-1)),))
            first = interner.intern(positive, -1, 1, semantic_node_id=10)
            second = interner.intern(negative, -1, 1, semantic_node_id=11)
            duplicate = interner.intern(positive, -1, 1, semantic_node_id=12)
            self.assertTrue(first.created)
            self.assertTrue(second.created)
            self.assertFalse(duplicate.created)
            self.assertEqual(positive.semantic_digest, negative.semantic_digest)
            self.assertEqual(first.handle.handle_digest, second.handle.handle_digest)
            self.assertNotEqual(first.handle.row_ids, second.handle.row_ids)
            self.assertEqual(first.handle.row_ids, duplicate.handle.row_ids)
            self.assertEqual(arena.row_count, 6)

            row_arena = ExactConstraintArena()
            original = ExactConstraintRow(
                7, "row:7", "le", ((1, Q(1)),), (), Q(1)
            )
            different = ExactConstraintRow(
                7, "row:7", "le", ((1, Q(1)),), (), Q(2)
            )
            self.assertEqual(original.row_digest, different.row_digest)
            row_arena.intern(original)
            with self.assertRaisesRegex(ValueError, "collides"):
                row_arena.intern(different)
            self.assertEqual(row_arena.row_count, 1)
            distinct_id = ExactConstraintRow(
                8, "row:8", "le", ((1, Q(1)),), (), Q(1)
            )
            self.assertEqual(original.row_digest, distinct_id.row_digest)
            row_arena.intern(distinct_id)
            self.assertEqual(row_arena.row_count, 2)

    def test_allocator_is_globally_fresh_and_three_row_commit_is_atomic(self):
        continuous_collision_arena = ExactConstraintArena()
        continuous_collision = ExactReLUInterner(
            continuous_collision_arena,
            reuse_predicates=True,
            next_continuous_id=100,
            next_binary_id=200,
            next_row_id=300,
        )
        fake_y_predicate = ExactAffineForm(Q(0), ((100, Q(1)),))
        with self.assertRaisesRegex(ValueError, "collide"):
            continuous_collision.intern(
                fake_y_predicate, -1, 1, semantic_node_id=10
            )
        self.assertEqual(continuous_collision_arena.row_count, 0)
        recovered = continuous_collision.intern(
            ExactAffineForm(Q(0), ((1, Q(1)),)),
            -1,
            1,
            semantic_node_id=13,
        )
        self.assertEqual(
            (
                recovered.handle.xi1_stable_id,
                recovered.handle.xi2_stable_id,
                recovered.handle.phase_stable_id,
                recovered.handle.row_ids,
            ),
            (100, 101, 200, (300, 301, 302)),
        )

        binary_collision_arena = ExactConstraintArena()
        binary_collision = ExactReLUInterner(
            binary_collision_arena,
            reuse_predicates=True,
            next_continuous_id=100,
            next_binary_id=200,
            next_row_id=300,
        )
        binary_predicate = ExactAffineForm(Q(0), (), ((200, Q(1)),))
        with self.assertRaisesRegex(ValueError, "collide"):
            binary_collision.intern(
                binary_predicate, -1, 1, semantic_node_id=11
            )
        self.assertEqual(binary_collision_arena.row_count, 0)

        live_arena = ExactConstraintArena()
        live_arena.intern(
            ExactConstraintRow(900, "live", "le", ((100, Q(1)),), (), Q(1))
        )
        live_collision = ExactReLUInterner(
            live_arena,
            reuse_predicates=True,
            next_continuous_id=100,
            next_binary_id=200,
            next_row_id=300,
        )
        with self.assertRaisesRegex(ValueError, "collide"):
            live_collision.intern(
                ExactAffineForm(Q(0), ((1, Q(1)),)),
                -1,
                1,
                semantic_node_id=12,
            )
        self.assertEqual(live_arena.row_count, 1)

        transactional = ExactConstraintArena()
        blocker = ExactConstraintRow(
            3, "blocker", "le", ((9, Q(1)),), (), Q(1)
        )
        transactional.intern(blocker)
        batch = (
            ExactConstraintRow(1, "one", "le", ((1, Q(1)),), (), Q(1)),
            ExactConstraintRow(2, "two", "le", ((2, Q(1)),), (), Q(1)),
            ExactConstraintRow(3, "three", "le", ((3, Q(1)),), (), Q(1)),
        )
        with self.assertRaisesRegex(ValueError, "collides"):
            transactional.intern_many(batch)
        self.assertEqual(transactional.stable_row_ids, frozenset({3}))

        row_id_arena = ExactConstraintArena()
        row_id_arena.intern(
            ExactConstraintRow(
                302, "occupied-row", "le", ((999, Q(1)),), (), Q(1)
            )
        )
        row_id_interner = ExactReLUInterner(
            row_id_arena,
            reuse_predicates=True,
            next_continuous_id=100,
            next_binary_id=200,
            next_row_id=300,
        )
        with self.assertRaisesRegex(ValueError, "row IDs collide"):
            row_id_interner.intern(
                ExactAffineForm(Q(0), ((1, Q(1)),)),
                -1,
                1,
                semantic_node_id=14,
            )
        self.assertEqual(row_id_arena.stable_row_ids, frozenset({302}))

    def test_generic_immutable_dag_row_view_union(self):
        left = ExactDAGRowView(frozenset({1, 2}))
        right = ExactDAGRowView(frozenset({2, 3}))
        tail = ExactDAGRowView(frozenset({4}))
        expected = frozenset({1, 2, 3, 4})
        self.assertEqual(left.union(right, tail).row_ids, expected)
        self.assertEqual(tail.union(right, left).row_ids, expected)
        self.assertEqual(
            left.union(right).union(tail).row_ids,
            left.union(right.union(tail)).row_ids,
        )
        self.assertEqual(left.union(left).row_ids, left.row_ids)

        arena = ExactConstraintArena()
        for row_id in expected:
            arena.intern(
                ExactConstraintRow(
                    row_id,
                    f"row:{row_id}",
                    "le",
                    ((10 + row_id, Q(1)),),
                    (),
                    Q(1),
                )
            )
        materialized = left.union(right, tail).materialize(arena)
        self.assertEqual(
            tuple(row.stable_row_id for row in materialized),
            (1, 2, 3, 4),
        )
        with self.assertRaises(ValueError):
            arena.rows_for_ids((1, 1))
        object.__setattr__(left, "row_ids", {1, 2})
        with self.assertRaises(ValueError):
            left.union(right)

    def test_generated_hybridz_rows_have_exact_fraction_phase_witnesses(self):
        arena = ExactConstraintArena()
        interner = ExactReLUInterner(
            arena,
            reuse_predicates=True,
            next_continuous_id=100,
            next_binary_id=200,
            next_row_id=300,
        )
        predicate = ExactAffineForm(Q(0), ((1, Q(1)),))
        handle = interner.intern(
            predicate, -1, 1, semantic_node_id=10
        ).handle
        rows = arena.rows_for_ids(handle.row_ids)
        self.assertEqual(tuple(row.kind for row in rows), ("eq", "le", "le"))

        for preactivation in (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1)):
            # HZ binary factors are {-1,+1}: +1 is the inactive segment and
            # -1 is the active segment.  Both witnesses exist at the kink.
            phases = (Q(1), Q(-1)) if preactivation == 0 else (
                (Q(1),) if preactivation < 0 else (Q(-1),)
            )
            for phase in phases:
                if phase == 1:
                    xi1 = 2 * preactivation / handle.lower - 1
                    xi2 = Q(1)
                    expected_output = Q(0)
                else:
                    xi1 = Q(1)
                    xi2 = 1 - 2 * preactivation / handle.upper
                    expected_output = preactivation
                self.assertTrue(-1 <= xi1 <= 1)
                self.assertTrue(-1 <= xi2 <= 1)
                continuous = {
                    1: preactivation,
                    handle.xi1_stable_id: xi1,
                    handle.xi2_stable_id: xi2,
                }
                binary = {handle.phase_stable_id: phase}
                for row in rows:
                    lhs = sum(
                        (coefficient * continuous[stable_id]
                         for stable_id, coefficient in row.continuous_terms),
                        Q(0),
                    ) + sum(
                        (coefficient * binary[stable_id]
                         for stable_id, coefficient in row.binary_terms),
                        Q(0),
                    )
                    if row.kind == "eq":
                        self.assertEqual(lhs, row.rhs)
                    else:
                        self.assertLessEqual(lhs, row.rhs)
                output = handle.output.constant + sum(
                    (coefficient * continuous[stable_id]
                     for stable_id, coefficient in handle.output.continuous_terms),
                    Q(0),
                )
                self.assertEqual(output, expected_output)

    def test_hostile_stable_id_schema_row_kind_rhs_and_types_fail_closed(self):
        for unsafe_id in (True, 1.0, np.int64(1)):
            with self.assertRaises(ValueError):
                ExactAffineForm(Q(0), ((unsafe_id, Q(1)),))
        for unsafe_coefficient in (True, 1.0, np.float64(1.0)):
            with self.assertRaises(ValueError):
                ExactAffineForm(Q(0), ((1, unsafe_coefficient),))
        with self.assertRaises(ValueError):
            ExactAffineForm(Q(0), ((1, Q(1)),), ((1, Q(1)),))
        with self.assertRaises(ValueError):
            ExactConstraintRow(
                8, "overlap", "le", ((1, Q(1)),), ((1, Q(1)),), Q(1)
            )

        predicate = ExactAffineForm(Q(0), ((1, Q(1)),))
        object.__setattr__(predicate, "schema", _EqualString(predicate.schema))
        arena = ExactConstraintArena()
        interner = ExactReLUInterner(
            arena,
            reuse_predicates=True,
            next_continuous_id=10,
            next_binary_id=20,
            next_row_id=30,
        )
        with self.assertRaises(ValueError):
            interner.intern(predicate, -1, 1, semantic_node_id=1)

        row = ExactConstraintRow(7, "row:7", "le", ((1, Q(1)),), (), Q(1))
        attacks = (
            ("stable_row_id", 7.0),
            ("row_tag", _EqualString("row:7")),
            ("kind", _EqualString("le")),
            ("rhs", 1.0),
            ("rhs", True),
            ("schema", _EqualString(row.schema)),
        )
        for field, value in attacks:
            with self.subTest(field=field, value=value):
                attacked = replace(row)
                object.__setattr__(attacked, field, value)
                with self.assertRaises(ValueError):
                    ExactConstraintArena().intern(attacked)

        arena = ExactConstraintArena()
        arena.intern(row)
        arena.intern(replace(row))
        self.assertEqual(arena.row_count, 1)
        collision = ExactConstraintRow(7, "row:7", "le", ((1, Q(1)),), (), Q(2))
        with self.assertRaisesRegex(ValueError, "collides"):
            arena.intern(collision)

        point = evaluate_fixed_toy(Q(1, 4))
        object.__setattr__(point, "q", True)
        with self.assertRaises(ValueError):
            raw_unsafe_margins(point)

        projection = enumerate_phase_projections(compact=True)[0]
        object.__setattr__(projection, "feasible", 0)
        with self.assertRaises(ValueError):
            projection_values_at((projection,), Q(0))

    def test_row_and_predicate_validate_then_use_aba_consume_private_snapshots(self):
        row = ExactConstraintRow(9, "row:9", "eq", ((1, Q(1)),), (), Q(1))
        arena = ExactConstraintArena()
        original_validate_row = core._validate_row_snapshot

        def validate_row_then_attack(snapshot):
            original_validate_row(snapshot)
            object.__setattr__(row, "rhs", Q(99))

        with patch.object(core, "_validate_row_snapshot", side_effect=validate_row_then_attack):
            stored = arena.intern(row)
        self.assertEqual(stored.rhs, Q(1))
        self.assertEqual(arena.rows[0].rhs, Q(1))
        with self.assertRaises(ValueError):
            arena.intern(row)

        row_snapshot_attack = ExactConstraintRow(
            10, "row:10", "eq", ((1, Q(1)),), (), Q(1)
        )

        def validate_then_mutate_private_row(snapshot):
            original_validate_row(snapshot)
            object.__setattr__(snapshot, "rhs", Q(77))

        private_arena = ExactConstraintArena()
        with patch.object(
            core,
            "_validate_row_snapshot",
            side_effect=validate_then_mutate_private_row,
        ):
            with self.assertRaises(ValueError):
                private_arena.intern(row_snapshot_attack)
        self.assertEqual(private_arena.row_count, 0)

        predicate = ExactAffineForm(Q(0), ((5, Q(1)),))
        original_validate_predicate = core._validate_predicate_snapshot
        interner = ExactReLUInterner(
            ExactConstraintArena(), reuse_predicates=True,
            next_continuous_id=100, next_binary_id=200, next_row_id=300,
        )

        def validate_predicate_then_attack(snapshot):
            original_validate_predicate(snapshot)
            object.__setattr__(predicate, "continuous_terms", ((6, Q(1)),))

        with patch.object(
            core,
            "_validate_predicate_snapshot",
            side_effect=validate_predicate_then_attack,
        ):
            result = interner.intern(predicate, -1, 1, semantic_node_id=5)
        self.assertTrue(result.created)
        with self.assertRaises(ValueError):
            interner.intern(predicate, -1, 1, semantic_node_id=6)

        private_predicate = ExactAffineForm(Q(0), ((7, Q(1)),))
        private_interner = ExactReLUInterner(
            ExactConstraintArena(), reuse_predicates=True,
            next_continuous_id=400, next_binary_id=500, next_row_id=600,
        )

        def validate_then_mutate_private_predicate(snapshot):
            original_validate_predicate(snapshot)
            object.__setattr__(snapshot, "continuous_terms", ((8, Q(1)),))

        with patch.object(
            core,
            "_validate_predicate_snapshot",
            side_effect=validate_then_mutate_private_predicate,
        ):
            with self.assertRaises(ValueError):
                private_interner.intern(
                    private_predicate, -1, 1, semantic_node_id=7
                )

    def test_64_block_synthetic_rows_nnz_and_wall_gate(self):
        # Warm import/allocator paths before measuring, then alternate order.
        build_synthetic_residual_family(blocks=64, compact=False)
        build_synthetic_residual_family(blocks=64, compact=True)
        baseline_times = []
        compact_times = []
        baseline_receipt = compact_receipt = None
        for trial in range(7):
            order = (False, True) if trial % 2 == 0 else (True, False)
            for compact in order:
                receipt = build_synthetic_residual_family(blocks=64, compact=compact)
                if compact:
                    compact_times.append(receipt.elapsed_seconds)
                    compact_receipt = receipt
                else:
                    baseline_times.append(receipt.elapsed_seconds)
                    baseline_receipt = receipt
        assert baseline_receipt is not None and compact_receipt is not None
        self.assertEqual(
            (
                baseline_receipt.unique_rows,
                baseline_receipt.unique_nnz,
                baseline_receipt.materialized_rows,
                baseline_receipt.materialized_nnz,
            ),
            (64 * 12, 64 * 32, 64 * 12, 64 * 32),
        )
        self.assertEqual(
            (
                compact_receipt.unique_rows,
                compact_receipt.unique_nnz,
                compact_receipt.materialized_rows,
                compact_receipt.materialized_nnz,
            ),
            (64 * 6, 64 * 16, 64 * 6, 64 * 16),
        )
        wall_ratio = statistics.median(baseline_times) / statistics.median(compact_times)
        self.assertGreaterEqual(
            wall_ratio,
            1.5,
            msg=f"synthetic wall stop gate missed: ratio={wall_ratio:.6f}",
        )
        self.assertFalse(compact_receipt.proof_authority)
        self.assertFalse(compact_receipt.verdict_authority)
        for field in ("proof_authority", "verdict_authority"):
            with self.assertRaises(AttributeError):
                object.__setattr__(compact_receipt, field, True)
            self.assertIs(getattr(compact_receipt, field), False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
