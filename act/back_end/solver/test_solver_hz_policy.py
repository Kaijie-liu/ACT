# ===- act/back_end/solver/test_solver_hz_policy.py - HZ Policy Tests -====#

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    HZ_NUMERICAL_POLICY,
    SparseHZono,
    hz_minimize_output,
    hz_support_bounds,
    sparse_hz_from_bounds,
)


class HZNumericalPolicyTests(unittest.TestCase):
    @staticmethod
    def _fake_result(fun=0.75, dual=-0.25):
        return SimpleNamespace(
            success=True,
            status=0,
            x=np.asarray([0.0]),
            fun=fun,
            mip_dual_bound=dual,
            mip_gap=0.0,
        )

    def test_minimum_certificate_uses_dual_not_primal_incumbent(self):
        hz = SparseHZono(
            c=np.zeros(1),
            Gc=sp.csr_matrix((1, 0)),
            Gb=sp.csr_matrix([[1.0]]),
            Ac=sp.csr_matrix((0, 0)),
            Ab=sp.csr_matrix((0, 1)),
            b=np.zeros(0),
            frame_id=71,
        )
        with patch(
            "act.back_end.solver.solver_hz.milp",
            return_value=self._fake_result(),
        ):
            result = hz_minimize_output(
                hz,
                0,
                input_hz=hz,
                input_shape=(1, 1),
                time_limit=1.0,
            )
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.solver_certified_lower_bound, -0.25)
        self.assertEqual(result.solver_bound_kind, "mip_dual_bound")
        self.assertEqual(result.candidate_objective, -1.0)
        self.assertLess(result.minimum, -1.25)
        self.assertGreater(result.minimum, -1.250001)

    def test_support_bounds_use_dual_bounds_on_both_sides(self):
        hz = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=73,
        )
        with patch(
            "act.back_end.solver.solver_hz.milp",
            side_effect=lambda **kwargs: self._fake_result(
                fun=-0.25, dual=None
            ),
        ):
            result = hz_support_bounds(
                hz,
                [0],
                time_limit=1.0,
                relax_binaries=True,
            )
        self.assertLess(float(result.bounds.lb.item()), -0.25)
        self.assertGreater(float(result.bounds.ub.item()), 0.25)
        self.assertLess(float(result.bounds.ub.item()), 0.250001)

    def test_safe_margin_is_stric_and_frozen(self):
        self.assertEqual(HZ_NUMERICAL_POLICY.safe_positive_margin, 1e-7)
        self.assertEqual(HZ_NUMERICAL_POLICY.feasibility_tolerance, 1e-7)
        self.assertEqual(HZ_NUMERICAL_POLICY.integrality_tolerance, 1e-7)
        self.assertEqual(HZ_NUMERICAL_POLICY.mip_relative_gap, 0.0)


if __name__ == "__main__":
    unittest.main()
