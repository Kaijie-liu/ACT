import types
import unittest

import numpy as np
import scipy.sparse as sp

from act.back_end.moe.monolithic_f0 import (
    SAFE_MONOLITHIC_WEIGHTED_RANGE,
    UNKNOWN_MONOLITHIC_RELAXATION,
    solve_monolithic_weighted_top2_f0,
)
from act.back_end.solver.solver_hz import SparseHZono, sparse_empty


def _interval(lower, upper, pair):
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    hz = SparseHZono(
        c=np.asarray([center], dtype=np.float64),
        Gc=sp.csr_matrix([[radius]], dtype=np.float64),
        Gb=sparse_empty(1, 0),
        Ac=sparse_empty(0, 1),
        Ab=sparse_empty(0, 0),
        b=np.zeros(0, dtype=np.float64),
        Auc=sparse_empty(0, 1),
        Aub=sparse_empty(0, 0),
        ub=np.zeros(0, dtype=np.float64),
        frame_id=7,
        exact=False,
    )
    return types.SimpleNamespace(output_hz=hz, input_hz=hz, pair=pair)


class MonolithicF0Tests(unittest.TestCase):
    def test_union_minimum_equals_smallest_branch_minimum(self):
        decision = solve_monolithic_weighted_top2_f0(
            [_interval(1.0, 2.0, (0, 1)), _interval(3.0, 4.0, (0, 2))],
            input_shape=(1, 1),
            time_limit=5.0,
        )
        self.assertEqual(decision.status, "SAFE")
        self.assertEqual(decision.reason, SAFE_MONOLITHIC_WEIGHTED_RANGE)
        self.assertAlmostEqual(decision.minimum, 1.0, places=6)
        self.assertEqual(decision.active_pair, (0, 1))
        self.assertEqual(tuple(decision.candidate_input.shape), (1,))
        self.assertAlmostEqual(float(decision.candidate_input.item()), 1.0, places=6)

    def test_negative_relaxation_is_never_unsafe(self):
        decision = solve_monolithic_weighted_top2_f0(
            [_interval(-2.0, -1.0, (1, 2)), _interval(3.0, 4.0, (2, 3))],
            input_shape=(1, 1),
            time_limit=5.0,
        )
        self.assertEqual(decision.status, "UNKNOWN")
        self.assertEqual(decision.reason, UNKNOWN_MONOLITHIC_RELAXATION)
        self.assertEqual(decision.active_pair, (1, 2))
        self.assertLess(float(decision.candidate_objective), 0.0)


if __name__ == "__main__":
    unittest.main()
