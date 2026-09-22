import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, Fact, ConSet
from act.back_end.hybridz_tf import HybridzTF
from act.back_end.hybridz_tf.sparse_budget import MATRICES, SparseResourceLimit, estimate, storage
from act.back_end.hybridz_tf.sparse_conv_plan import conv_support_plan, scratch_bytes, geometry
from act.back_end.hybridz_tf.tf_cnn import sparse_conv2d_matrix_from_layer
from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_linear
from act.config.config import HybridZConfig
from act.util.device_manager import initialize_device


def example(batch=2, channels=2, groups=2, shape=(5, 6), kernel=(2, 3), stride=(2, 1), dilation=(2, 1), padding=(1, 1), nc=11, nb=5):
    rng = np.random.default_rng(20260922)
    n = batch*channels*shape[0]*shape[1]
    def matrix(rows, cols):
        values = rng.integers(-2, 3, size=(rows, cols)).astype(float)
        values[rng.random(values.shape) < .8] = 0.
        return sp.csr_matrix(values)
    hz = SparseHZono(rng.integers(-2, 3, n).astype(float), matrix(n, nc), matrix(n, nb),
        matrix(2, nc), matrix(2, nb), np.zeros(2), matrix(1, nc), matrix(1, nb), np.ones(1), frame_id=3)
    layer = SimpleNamespace(id=8, kind='CONV2D', params={'input_shape': (batch, channels, *shape),
        'weight': torch.tensor(rng.integers(-1, 2, size=(4, channels//groups, *kernel)), dtype=torch.float64),
        'groups': groups, 'stride': stride, 'dilation': dilation, 'padding': padding})
    op, bias = sparse_conv2d_matrix_from_layer(layer)
    op = sp.kron(sp.eye(batch, format='csr'), op, format='csr')
    return hz, layer, op


def oracle(matrix, operator):
    # Independent small-control reference using the ACT numerical convolution
    # row pattern, not the production planner's spatial loop.
    total = 0
    for row in range(operator.shape[0]):
        factors = set()
        for source in operator.indices[operator.indptr[row]:operator.indptr[row+1]]:
            factors.update(matrix.indices[matrix.indptr[source]:matrix.indptr[source+1]].tolist())
        total += len(factors)
    return total


class ConvPlanControls(unittest.TestCase):
    def setUp(self):
        initialize_device('cpu', 'float64')
        torch.set_num_threads(1)

    def test_structural_oracle_and_numeric_upper_bound(self):
        for kw in ({}, {'batch': 1, 'groups': 1}, {'groups': 2, 'kernel': (1, 1), 'padding': 0, 'stride': 1},
                   {'nc': 0}, {'nb': 0}, {'nc': 0, 'nb': 0},
                   {'kernel': (3, 2), 'dilation': (1, 2), 'padding': (2, 1)}):
            hz, layer, op = example(**kw)
            before = copy.deepcopy(hz)
            plan = conv_support_plan(layer, hz, op.shape[0])
            self.assertEqual(plan['continuous_nnz_upper_bound'], oracle(hz.Gc, op))
            self.assertEqual(plan['binary_nnz_upper_bound'], oracle(hz.Gb, op))
            out = sparse_hz_linear(hz, op)
            self.assertLessEqual(storage(out)['nnz'], plan['nnz_upper_bound'])
            self.assertLessEqual(storage(out)['bytes'], plan['estimated_retained_bytes'])
            self.assertLessEqual(plan['nnz_upper_bound'], estimate(layer, hz, op.shape[0])['nnz_upper_bound'])
            for name in MATRICES:
                for field in ('data', 'indices', 'indptr'):
                    np.testing.assert_array_equal(getattr(getattr(hz, name), field), getattr(getattr(before, name), field))

    def test_duplicate_unsorted_zero_indices_conservative(self):
        hz, layer, op = example(batch=1, shape=(2, 2), kernel=(1, 1), stride=1, dilation=1, padding=0)
        # Inject noncanonical stored entries AFTER HZ normalization.
        indices = np.tile([3, 1, 3, 0], hz.n_out)
        hz.Gc = sp.csr_matrix((np.tile([1., 0., -1., 2.], hz.n_out), indices,
                               np.arange(0, 4*hz.n_out+1, 4)), shape=hz.Gc.shape)
        plan = conv_support_plan(layer, hz, op.shape[0])
        self.assertEqual(plan['continuous_nnz_upper_bound'], oracle(hz.Gc, op))
        self.assertFalse(hz.Gc.has_canonical_format)

    def test_zero_weights_not_used_to_shrink(self):
        hz, layer, op = example()
        first = conv_support_plan(layer, hz, op.shape[0])
        layer.params['weight'].zero_()
        second = conv_support_plan(layer, hz, op.shape[0])
        self.assertEqual(first['nnz_upper_bound'], second['nnz_upper_bound'])

    def test_duplicate_indices_across_scratch_chunks(self):
        hz, layer, op = example(batch=1, shape=(2, 2), kernel=(1, 1), stride=1, dilation=1, padding=0, nc=6000, nb=0)
        indices = np.concatenate([np.arange(5000), np.arange(4999, -1, -1)])
        indptr = np.full(hz.n_out+1, indices.size, dtype=np.int64)
        indptr[0] = 0
        hz.Gc = sp.csr_matrix((np.ones(indices.size), indices, indptr), shape=hz.Gc.shape)
        plan = conv_support_plan(layer, hz, op.shape[0])
        self.assertEqual(plan['continuous_nnz_upper_bound'], 10000)
        self.assertEqual(plan['continuous_nnz_upper_bound'], oracle(hz.Gc, op))

    def test_different_batch_support_not_first_times_batch(self):
        hz, layer, op = example(batch=2)
        split = hz.n_out//2
        hz.Gc.data[:hz.Gc.indptr[split]] = 0.
        hz.Gc.eliminate_zeros()
        hz.Gb.data[:hz.Gb.indptr[split]] = 0.
        hz.Gb.eliminate_zeros()
        plan = conv_support_plan(layer, hz, op.shape[0])
        self.assertGreater(plan['nnz_upper_bound'], hz.Ac.nnz+hz.Ab.nnz)
        self.assertEqual(plan['continuous_nnz_upper_bound'], oracle(hz.Gc, op))

    def test_invalid_geometry_and_indices_rejected(self):
        hz, layer, op = example()
        for key, value in [('groups', 0), ('stride', 0), ('dilation', -1), ('padding', -1), ('groups', 3)]:
            bad = copy.deepcopy(layer)
            bad.params[key] = value
            with self.assertRaises(ValueError):
                geometry(bad, hz, op.shape[0])
        with self.assertRaises(ValueError):
            geometry(layer, hz, op.shape[0]+1)
        hz.Gc.indices[0] = hz.n_cont
        with self.assertRaises(ValueError):
            conv_support_plan(layer, hz, op.shape[0])

    def tf(self, hz, limit=2**31):
        tf = HybridzTF(HybridZConfig(sparse_resource_policy='csr_spatial_v2', sparse_representation_bytes=limit))
        tf._sparse_hz_cache[8] = hz
        return tf

    def test_scratch_refused_before_marker_or_builder(self):
        hz, layer, op = example()
        bounds = Bounds(torch.full((1, op.shape[0]), -1000.), torch.full((1, op.shape[0]), 1000.))
        tf = self.tf(hz, storage(hz)['bytes']+scratch_bytes(hz)-1)
        with (patch('act.back_end.hybridz_tf.sparse_conv_plan.np.zeros', side_effect=AssertionError('no marker')) as marker,
              patch('act.back_end.hybridz_tf.tf_cnn.sparse_conv2d_matrix_from_layer', side_effect=AssertionError('no op')) as builder):
            with self.assertRaises(SparseResourceLimit):
                tf._propagate_sparse_hz(layer, bounds, Fact(bounds, ConSet()))
        marker.assert_not_called()
        builder.assert_not_called()
        self.assertIsNone(tf.get_sparse_hz(8))

    def test_operator_refused_after_count_before_builder(self):
        hz, layer, op = example()
        tf = self.tf(hz, storage(hz)['bytes']+scratch_bytes(hz)+1)
        bounds = Bounds(torch.full((1, op.shape[0]), -1000.), torch.full((1, op.shape[0]), 1000.))
        with patch('act.back_end.hybridz_tf.tf_cnn.sparse_conv2d_matrix_from_layer') as builder:
            with self.assertRaises(SparseResourceLimit):
                tf._propagate_sparse_hz(layer, bounds, Fact(bounds, ConSet()))
        builder.assert_not_called()

    def test_no_partial_plan_on_exception(self):
        hz, layer, op = example()
        tf = self.tf(hz)
        bounds = Bounds(torch.full((1, op.shape[0]), -1000.), torch.full((1, op.shape[0]), 1000.))
        with patch('act.back_end.hybridz_tf.sparse_conv_plan.np.count_nonzero', side_effect=TimeoutError('control')):
            with self.assertRaises(TimeoutError):
                tf._propagate_sparse_hz(layer, bounds, Fact(bounds, ConSet()))
        self.assertIsNone(tf.get_sparse_hz(8))
        self.assertEqual(tf._sparse_relu_slots, {})

    def test_v1_v2_matrices_identical_no_dense(self):
        hz, layer, op = example()
        outputs = []
        bounds = Bounds(torch.full((1, op.shape[0]), -1000.), torch.full((1, op.shape[0]), 1000.))
        for policy in ('csr_bytes_v1', 'csr_spatial_v2'):
            tf = HybridzTF(HybridZConfig(sparse_resource_policy=policy, sparse_representation_bytes=2**31))
            tf._sparse_hz_cache[8] = hz
            with patch.object(sp.csr_matrix, 'toarray', side_effect=AssertionError('dense forbidden')):
                tf._propagate_sparse_hz(layer, bounds, Fact(bounds, ConSet()))
            outputs.append(tf.get_sparse_hz(8))
        self.assertEqual(outputs[0].frame_id, outputs[1].frame_id)
        self.assertEqual(outputs[0].exact, outputs[1].exact)
        for name in MATRICES:
            for field in ('data', 'indices', 'indptr'):
                np.testing.assert_array_equal(getattr(getattr(outputs[0], name), field), getattr(getattr(outputs[1], name), field))
        for name in ('c', 'b', 'ub'):
            np.testing.assert_array_equal(getattr(outputs[0], name), getattr(outputs[1], name))

    def test_complete_guarded_cnn_chain_differential(self):
        from test_sparse_resource_policy import ResourceControls
        from act.back_end.solver.solver_hz import hz_add_output_inequalities
        helper = ResourceControls()
        program = helper.program()
        results = []
        for policy in ('csr_bytes_v1', 'csr_spatial_v2'):
            tf = HybridzTF(HybridZConfig(sparse_resource_policy=policy, sparse_representation_bytes=2**31))
            plain, input_hz = helper.propagate(program, tf)
            guard = hz_add_output_inequalities(input_hz, torch.ones(1, 16), torch.tensor([0.]))
            tf.set_entry_hz(guard)
            with patch.object(sp.csr_matrix, 'toarray', side_effect=AssertionError('no dense')):
                guarded, seed = helper.propagate(program, tf)
            self.assertIs(seed, guard)
            results.append((plain, guarded, dict(tf._sparse_relu_slots)))
        self.assertEqual(results[0][2], results[1][2])
        for a, b in zip(results[0][:2], results[1][:2]):
            self.assertEqual((a.frame_id, a.exact), (b.frame_id, b.exact))
            for name in MATRICES:
                for field in ('data', 'indices', 'indptr'):
                    np.testing.assert_array_equal(getattr(getattr(a, name), field), getattr(getattr(b, name), field))
            for name in ('c', 'b', 'ub'):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


if __name__ == '__main__':
    unittest.main()
