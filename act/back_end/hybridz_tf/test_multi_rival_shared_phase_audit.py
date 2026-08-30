"""Fraction and HiGHS gates for the shared-phase multi-rival audit."""

from __future__ import annotations

from fractions import Fraction
import unittest

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from act.back_end.hybridz_tf.multi_rival_shared_phase_audit import (
    audit_duplicate_relu_shared_phase,
    predicate_disjunction_hull_upper,
)


Q = Fraction


class MultiRivalSharedPhaseFractionTests(unittest.TestCase):
    def test_shared_phase_strictly_tightens_both_rivals(self) -> None:
        result = audit_duplicate_relu_shared_phase()
        self.assertEqual(result.status, "exact_shared_phase_cover")
        self.assertEqual(result.root_upper, (Q(2, 5), Q(2, 5)))
        self.assertEqual(
            result.shared_phase_upper, (Q(-1, 10), Q(-1, 10))
        )
        self.assertEqual(
            result.independent_phase_upper, result.shared_phase_upper
        )
        self.assertEqual(result.one_sided_copy_upper, (Q(-1, 10), Q(2, 5)))
        self.assertFalse(result.proof_authority)

    def test_sharing_saves_nodes_but_not_bound_at_equal_cover(self) -> None:
        result = audit_duplicate_relu_shared_phase()
        self.assertEqual(result.shared_tree_nodes, 3)
        self.assertEqual(result.independent_tree_nodes, 6)
        self.assertEqual(
            result.shared_phase_upper, result.independent_phase_upper
        )

    def test_predicate_disjunction_hull_cannot_improve_fixed_relaxation(self) -> None:
        result = audit_duplicate_relu_shared_phase()
        self.assertEqual(result.predicate_hull_upper, Q(2, 5))
        self.assertEqual(
            result.predicate_hull_upper, max(result.root_upper)
        )
        # A second unrelated rational polytope guards against accidentally
        # relying on the symmetric duplicate-ReLU geometry.
        points = ((Q(-2), Q(1)), (Q(1), Q(3)), (Q(4), Q(-1)))
        forms = (
            ((Q(2), Q(-1)), Q(1, 3)),
            ((Q(-1), Q(3)), Q(2, 5)),
            ((Q(1), Q(1)), Q(-1, 7)),
        )
        individual = tuple(
            max(
                sum(c * x for c, x in zip(coefficients, point))
                - threshold
                for point in points
            )
            for coefficients, threshold in forms
        )
        self.assertEqual(
            predicate_disjunction_hull_upper(
                relaxation_vertices=points, rival_forms=forms
            ),
            max(individual),
        )

    def test_wrong_copy_and_rival_permutation_fail_safe(self) -> None:
        wrong = audit_duplicate_relu_shared_phase(shared_stable_id=False)
        self.assertEqual(wrong.status, "wrong_copy_fail_closed")
        self.assertIsNone(wrong.shared_phase_upper)
        self.assertEqual(wrong.root_upper, (Q(2, 5), Q(2, 5)))
        permuted = audit_duplicate_relu_shared_phase(rival_ids=(20, 10))
        self.assertEqual(permuted.root_upper, (Q(2, 5), Q(2, 5)))
        self.assertEqual(
            permuted.shared_phase_upper, (Q(-1, 10), Q(-1, 10))
        )


class MultiRivalSharedPhaseHighsTests(unittest.TestCase):
    @staticmethod
    def _triangle_constraints() -> tuple[np.ndarray, np.ndarray]:
        # Variables are (x, a, b).
        A = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        b = np.asarray(
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        return A, b

    def test_highs_lp_and_exact_shared_binary_milp_match_fraction(self) -> None:
        A, b = self._triangle_constraints()
        for objective in (
            np.asarray([0.0, 1.0, -1.0]),
            np.asarray([0.0, -1.0, 1.0]),
        ):
            relaxed = linprog(
                -objective,
                A_ub=A,
                b_ub=b,
                bounds=[(None, None)] * 3,
                method="highs",
            )
            self.assertTrue(relaxed.success)
            self.assertAlmostEqual(
                -float(relaxed.fun) - 0.1, 0.4, places=10
            )

            # Exact graph with a shared active-phase binary z.  For each of
            # a,b: y>=0, y>=x, y<=x+1-z, y<=z.
            rows = []
            upper = []
            for y_column in (1, 2):
                row = np.zeros(4)
                row[y_column] = -1.0
                rows.append(row)
                upper.append(0.0)
                row = np.zeros(4)
                row[0] = 1.0
                row[y_column] = -1.0
                rows.append(row)
                upper.append(0.0)
                row = np.zeros(4)
                row[0] = -1.0
                row[y_column] = 1.0
                row[3] = 1.0
                rows.append(row)
                upper.append(1.0)
                row = np.zeros(4)
                row[y_column] = 1.0
                row[3] = -1.0
                rows.append(row)
                upper.append(0.0)
            objective4 = np.concatenate([objective, [0.0]])
            exact = milp(
                -objective4,
                integrality=np.asarray([0, 0, 0, 1], dtype=np.int8),
                bounds=Bounds(
                    [-1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ),
                constraints=LinearConstraint(
                    np.asarray(rows), -np.inf, np.asarray(upper)
                ),
                options={"time_limit": 1.0},
            )
            self.assertTrue(exact.success)
            self.assertAlmostEqual(
                -float(exact.fun) - 0.1, -0.1, places=10
            )

    def test_wrong_copy_exact_milp_is_not_safe(self) -> None:
        # Fresh x2 has the same interval/shape as x1 but is a different
        # semantic source.  relu(x1)-relu(x2)-1/10 reaches 9/10.
        # Variables: x1,x2,y1,y2,z1,z2.
        rows = []
        upper = []
        for x_column, y_column, z_column in ((0, 2, 4), (1, 3, 5)):
            row = np.zeros(6)
            row[y_column] = -1.0
            rows.append(row)
            upper.append(0.0)
            row = np.zeros(6)
            row[x_column] = 1.0
            row[y_column] = -1.0
            rows.append(row)
            upper.append(0.0)
            row = np.zeros(6)
            row[x_column] = -1.0
            row[y_column] = 1.0
            row[z_column] = 1.0
            rows.append(row)
            upper.append(1.0)
            row = np.zeros(6)
            row[y_column] = 1.0
            row[z_column] = -1.0
            rows.append(row)
            upper.append(0.0)
        exact = milp(
            np.asarray([0.0, 0.0, -1.0, 1.0, 0.0, 0.0]),
            integrality=np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int8),
            bounds=Bounds(
                [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            constraints=LinearConstraint(
                np.asarray(rows), -np.inf, np.asarray(upper)
            ),
            options={"time_limit": 1.0},
        )
        self.assertTrue(exact.success)
        self.assertAlmostEqual(-float(exact.fun) - 0.1, 0.9, places=10)


if __name__ == "__main__":
    unittest.main()
