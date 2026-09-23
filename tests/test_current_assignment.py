import copy
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver import solver_hz as sh
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from act.back_end.solver.current_assignment import (
    AssignmentScope, ProposalUnavailable, model_fingerprint,
    propose_current_assignment, check_current_assignment,
)


class AssignmentControls(unittest.TestCase):
    def setUp(self):
        self.scope = AssignmentScope('1'*64, '2'*64, 'current-test-request')
        self.input = sh.sparse_hz_from_bounds(Bounds(torch.tensor([[-1.]], dtype=torch.float64),
            torch.tensor([[1.]], dtype=torch.float64)), frame_id=741)
        self.relu = sparse_hz_apply_relu_exact(self.input, [-1.], [1.], [(1, 2, 0)], 3, 1)
        self.model = sh._lower_hz_milp(self.relu)

    def propose(self, model=None, seed=None):
        return propose_current_assignment(model or self.model, self.scope, time.monotonic()+5, free_values=seed)

    def check(self, proposal, model=None, scope=None):
        return check_current_assignment(model or self.model, proposal, scope or self.scope, time.monotonic()+5)

    def test_active_inactive_and_tie_no_solver(self):
        with patch.object(sh, 'milp', side_effect=AssertionError('must not solve')):
            for seed in (-.75, 0., .75):
                with self.subTest(seed=seed):
                    p = self.propose(seed=[seed])
                    x, report = self.check(p)
                    self.assertTrue(report['accepted'])
                    self.assertEqual(report['solver_calls'], 0)
                    out = self.model.value_center+self.model.value_matrix @ x
                    self.assertAlmostEqual(out.item(), max(seed, 0.))

    def test_two_layers_nontrivial_shared_dependency(self):
        pre = sh.sparse_hz_linear(self.relu, np.array([[2.]]), np.array([-.25]))
        out = sparse_hz_apply_relu_exact(pre, [-.25], [1.75], [(3, 4, 1)], 5, 2)
        model = sh._lower_hz_milp(out)
        for seed in (-1., .125, .75):
            p = self.propose(model, [seed])
            x, report = self.check(p, model)
            self.assertTrue(report['accepted'])
            self.assertAlmostEqual((model.value_center+model.value_matrix @ x).item(), max(2*max(seed, 0)-.25, 0))

    def test_differential_with_native_base_feasibility(self):
        point, report = self.check(self.propose())
        native = sh._solve_hz_feasibility(self.model, time.monotonic()+5)
        self.assertTrue(report['accepted'])
        self.assertEqual(native.status, 'feasible')
        self.assertIsNotNone(point)

    def test_affine_no_relu_supported(self):
        model = sh._lower_hz_milp(self.input)
        p = self.propose(model)
        self.assertTrue(self.check(p, model)[1]['accepted'])

    def test_constant_zero_dimension_supported(self):
        inp = sh.sparse_hz_from_bounds(Bounds(torch.zeros((1, 1)), torch.zeros((1, 1))))
        model = sh._lower_hz_milp(inp)
        self.assertTrue(self.check(self.propose(model), model)[1]['accepted'])

    def test_current_guard_rejects_center_not_in_branch(self):
        # Add x <= -0.25: seed0 is NOT a feasible guard witness.
        m = copy.deepcopy(self.model)
        m = replace(m, A=sp.vstack([m.A, sp.csr_matrix([[1., 0., 0., 0.]])], format='csr'),
                    row_lb=np.r_[m.row_lb, -np.inf], row_ub=np.r_[m.row_ub, -.25])
        x, report = self.check(self.propose(m), m)
        self.assertIsNone(x)
        self.assertEqual(report['reason'], 'full_feasibility_check_failed')
        self.assertTrue(self.check(self.propose(m, [-.5]), m)[1]['accepted'])

    def test_wrong_request_input_and_nonce_rejected(self):
        p = self.propose()
        for key, val in [('request_sha256', 'a'*64), ('input_sha256', 'b'*64), ('evaluation_nonce', 'other')]:
            with self.subTest(key=key):
                self.assertFalse(self.check(p, scope=replace(self.scope, **{key: val}))[1]['accepted'])

    def test_changed_matrix_rejected_even_if_point_still_feasible(self):
        p = self.propose()
        m = copy.deepcopy(self.model)
        m.row_ub[-1] += .5
        self.assertIn('identity', self.check(p, m)[1]['reason'])

    def test_changed_output_mapping_rejected(self):
        p = self.propose()
        m = copy.deepcopy(self.model)
        m.value_center[0] += 1.
        self.assertFalse(self.check(p, m)[1]['accepted'])

    def test_partial_assignment_rejected(self):
        p = self.propose()
        self.assertFalse(self.check(replace(p, point=p.point[:-1]))[1]['accepted'])

    def test_binary_corruption_rejected(self):
        p = self.propose()
        x = p.point.copy(); x[-1] = .5
        self.assertFalse(self.check(replace(p, point=x))[1]['accepted'])

    def test_equality_corruption_rejected(self):
        p = self.propose()
        x = p.point.copy(); x[1] = .5
        self.assertFalse(self.check(replace(p, point=x))[1]['accepted'])

    def test_bad_seed_rejected_without_clipping(self):
        for seed in ([2.], [float('nan')], []):
            with self.subTest(seed=seed), self.assertRaises(ProposalUnavailable):
                self.propose(seed=seed)

    def test_unknown_encoding_refused_not_false_infeasibility(self):
        m = copy.deepcopy(self.model)
        pos = np.flatnonzero(m.A.indices[:m.A.indptr[1]] == m.n_cont)[0]
        m.A.data[pos] *= 3
        with self.assertRaises(ProposalUnavailable):
            self.propose(m)

    def test_nonfinite_matrix_and_impossible_infinity_rejected(self):
        for kind in ('A', 'row'):
            m = copy.deepcopy(self.model)
            if kind == 'A': m.A.data[0] = np.nan
            else: m.row_lb[-1] = m.row_ub[-1] = np.inf
            with self.subTest(kind=kind), self.assertRaises(ProposalUnavailable):
                self.propose(m)

    def test_deadline_before_proposal(self):
        with self.assertRaisesRegex(ProposalUnavailable, 'deadline'):
            propose_current_assignment(self.model, self.scope, time.monotonic()-1)

    def test_deadline_before_checker(self):
        p = self.propose()
        self.assertFalse(check_current_assignment(self.model, p, self.scope, time.monotonic()-1)[1]['accepted'])

    def test_deadline_during_check_demotes_valid_point(self):
        p = self.propose()
        with patch('act.back_end.solver.current_assignment.time.monotonic', side_effect=[0., 0., 2., 2., 2.]):
            self.assertFalse(check_current_assignment(self.model, p, self.scope, 1.)[1]['accepted'])

    def test_checker_does_not_trust_constructor_metadata(self):
        p = replace(self.propose(), relu_rows=-1, free_prefix=100000)
        self.assertTrue(self.check(p)[1]['accepted'])

    def test_base_point_is_not_property_proof(self):
        p = self.propose(seed=[.75])
        x, report = self.check(p)
        self.assertTrue(report['accepted'])
        # A constraint added for an unrelated safety query still must be checked.
        A, lb, ub = sh._combined_constraints(self.model, self.model.value_matrix,
            [-np.inf], [.1-self.model.value_center.item()])
        self.assertFalse(sh._valid_milp_point(self.model, x, A, lb, ub, 1e-7))


if __name__ == '__main__':
    unittest.main()
