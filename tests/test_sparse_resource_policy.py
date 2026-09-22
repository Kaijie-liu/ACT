import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from act.back_end.core import Bounds, Fact, ConSet
from act.back_end.hybridz_tf import HybridzTF
from act.back_end.hybridz_tf.sparse_budget import (SparseBudget, SparseResourceLimit,
                                                MATRICES, estimate, storage)
from act.back_end.moe.model import OutputLevelMoE
from act.back_end.moe.schema import OutputLevelMoESpec, GateKind
from act.back_end.moe.factory import build_act_moe_program
from act.back_end.moe.route_a import _analyze_router
from act.back_end.solver.solver_hz import sparse_hz_from_bounds, hz_add_output_inequalities
from act.back_end.transfer_functions import set_transfer_function, set_solver_mode
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import initialize_device


def config(limit=2**30):
    return HybridZConfig(sparse_resource_policy='csr_bytes_v1', sparse_representation_bytes=limit)


class ResourceControls(unittest.TestCase):
    def setUp(self):
        initialize_device('cpu', 'float64')
        torch.set_num_threads(1)
        self.bounds = Bounds(torch.tensor([[-1., 0., -1., 1.]]), torch.tensor([[1., 1., 0., 2.]]))
        self.hz = sparse_hz_from_bounds(self.bounds, frame_id=7)
        self.layer = SimpleNamespace(id=10, kind='RELU', params={})
        self.fact = Fact(Bounds(self.bounds.lb.relu(), self.bounds.ub.relu()), ConSet())

    def tf(self, limit=2**30):
        tf = HybridzTF(config(limit))
        tf._sparse_hz_cache[10] = self.hz
        tf._sparse_frame_widths[7] = (self.hz.n_cont, self.hz.n_bin)
        return tf

    def test_invalid_config_and_legacy_default(self):
        self.assertIsNone(HybridzTF()._sparse_budget)
        for kw in [dict(sparse_resource_policy='x'), dict(sparse_representation_bytes=True),
                   dict(sparse_representation_bytes=-1), dict(sparse_representation_bytes=1),
                   dict(sparse_resource_policy='csr_bytes_v1')]:
            with self.assertRaises(ValueError):
                HybridZConfig(**kw)

    def test_full_entry_low_budget_is_unknown(self):
        from act.back_end.moe.class_separated_top1 import ClassSeparatedTop1, verify_class_separated_box
        router = nn.Linear(1, 2).double().eval()
        experts = [nn.Linear(1, 2).double().eval() for _ in range(2)]
        with torch.no_grad():
            router.weight.zero_()
            router.bias.copy_(torch.tensor([1., 2.]))
            for expert in experts:
                expert.weight.zero_()
                expert.bias.fill_(2.)
        model = ClassSeparatedTop1(router, experts, (2, 2))
        x = torch.zeros(1, 1)
        result = verify_class_separated_box(model, center=x, lower=x-.1, upper=x+.1,
            rows=torch.ones(1, 4), thresholds=torch.tensor([.1]), total_seconds=3,
            hybridz_config=config(1))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['reason'], 'sparse_representation_resource_limit')

    def test_all_arrays_counted_and_aliases_deduplicated(self):
        h = hz_add_output_inequalities(self.hz, torch.tensor([[-1., 0., 0., 0.]]), torch.tensor([-.5]))
        detail = storage(h)
        self.assertGreater(detail['bytes'], storage(self.hz)['bytes'])
        budget = SparseBudget(detail['bytes'])
        self.assertTrue(budget.admit('exact', 0, [h, h], 0)['accepted'])
        with self.assertRaises(SparseResourceLimit):
            budget.admit('one_extra', 0, [h, h], 1)

    def test_relu_pre_refusal_no_slot_mutation(self):
        tf = self.tf(1)
        frames = dict(tf._sparse_frame_widths)
        with self.assertRaises(SparseResourceLimit):
            tf._propagate_sparse_hz(self.layer, self.bounds, self.fact)
        self.assertEqual(tf._sparse_relu_slots, {})
        self.assertEqual(tf._sparse_frame_widths, frames)
        self.assertIsNone(tf.get_hz(10))
        self.assertIsNone(tf.get_sparse_hz(10))

    def test_relu_post_refusal_and_constructor_error_transaction(self):
        for failure in ('post', 'construct'):
            tf = self.tf()
            before = dict(tf._sparse_frame_widths)
            original = tf._budget_admit
            def admit(stage, *a, **kw):
                if stage == 'operator_post':
                    raise SparseResourceLimit({'stage': stage})
                return original(stage, *a, **kw)
            target = (patch.object(tf, '_budget_admit', side_effect=admit) if failure == 'post' else
                      patch('act.back_end.hybridz_tf.tf_mlp.sparse_hz_apply_relu_exact', side_effect=ValueError('construct')))
            with target, self.assertRaises((SparseResourceLimit, ValueError)):
                tf._propagate_sparse_hz(self.layer, self.bounds, self.fact)
            self.assertEqual(tf._sparse_frame_widths, before)
            self.assertEqual(tf._sparse_relu_slots, {})
            self.assertIsNone(tf._pending_sparse_slots)

    def test_relu_zero_boundaries_and_repeat_slots(self):
        tf = self.tf()
        tf._propagate_sparse_hz(self.layer, self.bounds, self.fact)
        out = tf.get_sparse_hz(10)
        self.assertEqual(out.n_bin, 1)
        slots = dict(tf._sparse_relu_slots)
        tf._sparse_hz_cache[10] = self.hz
        tf._propagate_sparse_hz(self.layer, self.bounds, self.fact)
        self.assertEqual(tf._sparse_relu_slots, slots)
        self.assertEqual(tf.get_sparse_hz(10).n_bin, out.n_bin)

    def test_large_shape_low_nnz_not_dense_allocation(self):
        b = Bounds(torch.full((1, 9000), -1.), torch.ones((1, 9000)))
        hz = sparse_hz_from_bounds(b, frame_id=0)
        legacy = HybridzTF()
        self.assertTrue(legacy._sparse_exceeds_limit(hz, 9000))
        tf = HybridzTF(config(64*2**20))
        tf._sparse_hz_cache[10] = hz
        with patch.object(sp.csr_matrix, 'toarray', side_effect=AssertionError('dense forbidden')):
            tf._propagate_sparse_hz(self.layer, b, Fact(Bounds(b.lb.relu(), b.ub.relu()), ConSet()))
        self.assertEqual(tf.get_sparse_hz(10).n_bin, 9000)

    def test_grouped_dilated_conv_pool_estimates_with_constraints(self):
        from act.back_end.hybridz_tf.tf_cnn import sparse_hz_apply_layer
        b = Bounds(torch.full((1, 2, 5, 5), -1.), torch.ones(1, 2, 5, 5))
        tf = HybridzTF(config())
        tf._sparse_hz_cache[10] = sparse_hz_from_bounds(b, frame_id=0)
        tf._propagate_sparse_hz(self.layer, b, Fact(Bounds(b.lb.relu(), b.ub.relu()), ConSet()))
        hz = tf.get_sparse_hz(10)
        self.assertGreater(hz.n_bin, 0)
        self.assertGreater(hz.n_eq+hz.n_ineq, 0)
        layers = [SimpleNamespace(id=20, kind='CONV2D', params={'input_shape': (1, 2, 5, 5),
            'weight': torch.ones(4, 1, 2, 2), 'groups': 2, 'stride': 2, 'dilation': 2}),
            SimpleNamespace(id=21, kind='AVGPOOL2D', params={'input_shape': (1, 2, 5, 5),
            'kernel_size': 2, 'stride': 2, 'ceil_mode': True, 'count_include_pad': False})]
        for layer in layers:
            tf = HybridzTF(config())
            handled, out, reason = sparse_hz_apply_layer(layer, hz, b, Fact(b, ConSet()), tf)
            self.assertTrue(handled)
            self.assertIsNone(reason)
            plan = estimate(layer, hz, out.n_out)
            self.assertLessEqual(storage(out)['bytes'], plan['estimated_retained_bytes'])

    def test_unknown_op_and_fill_in_reject_before_native_builder(self):
        tf = self.tf()
        with patch.dict(tf._LAYER_REGISTRY, {'CONSTANT': unittest.mock.Mock(side_effect=AssertionError('must not dispatch'))}):
            with self.assertRaises(SparseResourceLimit):
                tf.apply(SimpleNamespace(id=11, kind='CONSTANT'), self.bounds, None, {}, {})
            tf._LAYER_REGISTRY['CONSTANT'].assert_not_called()
        with self.assertRaises(SparseResourceLimit):
            tf._propagate_sparse_hz(SimpleNamespace(id=10, kind='MATMUL', params={}), self.bounds, self.fact)
        tf = self.tf(2000)
        layer = SimpleNamespace(id=10, kind='DENSE', params={'weight': torch.ones(200, 4)})
        fact = Fact(Bounds(torch.full((1, 200), -2.), torch.full((1, 200), 4.)), ConSet())
        with patch('act.back_end.hybridz_tf.tf_mlp.sparse_hz_apply_layer', side_effect=AssertionError('must not construct')):
            with self.assertRaises(SparseResourceLimit):
                tf._propagate_sparse_hz(layer, self.bounds, fact)

    def program(self):
        router = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                               nn.Flatten(), nn.Linear(8, 2)).double().eval()
        with torch.no_grad():
            for p in router.parameters():
                p.fill_(.5 if p.ndim > 1 else 0.)
        experts = [nn.Sequential(nn.Flatten(), nn.Linear(16, 2)).double().eval() for _ in range(2)]
        model = OutputLevelMoE(router, experts, OutputLevelMoESpec(2, 1, GateKind.HARD_TOP1)).eval()
        x = torch.zeros(1, 1, 4, 4)
        return build_act_moe_program(model, center=x, lower=x-1, upper=x+1,
            output_spec=OutputSpec(kind=OutKind.LINEAR_LE, c=torch.tensor([[-1., 1.]]), d=torch.zeros(1)))

    def propagate(self, program, tf):
        set_transfer_function(tf)
        set_solver_mode('hybridz')
        return _analyze_router(program.router, tf)

    def test_cnn_matrix_differential_and_guarded_sparse_entry(self):
        p = self.program()
        old, _ = self.propagate(p, HybridzTF())
        tf = HybridzTF(config())
        new, input_hz = self.propagate(p, tf)
        for name in MATRICES:
            a, b = getattr(old, name), getattr(new, name)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual((a-b).nnz, 0)
        for name in ('c', 'b', 'ub'):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
        guard = hz_add_output_inequalities(input_hz, torch.ones(1, 16), torch.tensor([0.]))
        tf.set_entry_hz(guard)
        with (patch.object(sp.csr_matrix, 'toarray', side_effect=AssertionError('dense forbidden')),
              patch('act.back_end.hybridz_tf.hybridz_tf.sparse_hz_to_dense', side_effect=AssertionError('dense entry forbidden'))):
            guarded, seed = self.propagate(p, tf)
        self.assertIs(seed, guard)
        self.assertEqual(guarded.frame_id, guard.frame_id)
        self.assertGreaterEqual(guarded.n_ineq, guard.n_ineq)
        self.assertGreater(tf._sparse_next_frame_id, guard.frame_id)
        self.assertFalse(tf._hz_cache)


if __name__ == '__main__':
    unittest.main()
