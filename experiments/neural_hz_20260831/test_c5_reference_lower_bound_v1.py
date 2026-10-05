from types import SimpleNamespace

import numpy as np
import pytest
import torch

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import expanded_roots
from experiments.neural_hz_20260831.c5_reference_lower_bound_v1 import lower_bound_roots, reachable_operators
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


def fixture(stride=1, dilation=1, groups=1, masked=False):
    params = dict(weight=torch.arange(4 * (4 // groups) * 9, dtype=torch.float64).reshape(4, 4 // groups, 3, 3) / 7.3,
                  input_shape=(1, 4, 12, 12), stride=stride, padding=dilation, dilation=dilation, groups=groups)
    plain = ImplicitConv2DOp(params['weight'].numpy(), params['input_shape'], stride=stride, padding=dilation,
                            dilation=dilation, groups=groups)
    mask = np.arange(plain.shape[0]) % 3 != 0 if masked else None
    op = ImplicitConv2DOp(params['weight'].numpy(), params['input_shape'], stride=stride, padding=dilation,
                         dilation=dilation, groups=groups, row_mask=mask)
    hz = source(op.shape[1])
    term = cnn.SparseHZAffineTerm(hz, (op,))
    expr = cnn.SparseHZAffineExpr((term, term), np.zeros(op.shape[0]), op.shape[0], hz.frame_id)
    tf = SimpleNamespace(expr=expr, shared_expr=expr, explicit_op=op, cache={5: hz})
    net = SimpleNamespace(layers=[SimpleNamespace(id=7, kind='CONV2D', params=params)])
    return tf, net


def measure(values):
    return snapshot_partial_csr_owners(WholeStateRoots(active=values, consumer_gc_enabled=False))


@pytest.mark.parametrize('geometry', [(1, 1, 1, False), (2, 1, 1, True), (1, 2, 1, False), (1, 1, 2, True)])
def test_reference_subset_is_real_lower_bound_on_complete_shared_reference(geometry):
    tf, net = fixture(*geometry)
    roots = collect(tf)
    witness, records = lower_bound_roots(roots, net)
    complete, _ = expanded_roots(roots, net)
    assert len(reachable_operators(roots)) == 1
    assert records[0]['reachable_implicit_objects'] == 1
    assert records[0]['every_row_bitwise_verified'] and not records[0]['complete_reference_materialized']
    # Add aliases and a shared predicate-bearing HZ on BOTH sides.
    for retained in ({}, {'hz': tf.cache[5], 'same_hz': tf.cache[5]}):
        lower, full = measure({**witness, **retained}), measure({**complete, **retained})
        assert lower.resident_bytes <= full.resident_bytes and lower.resident_entries <= full.resident_entries
        candidate = roots.measure(retained)
        if candidate.resident_bytes < lower.resident_bytes and candidate.resident_entries < lower.resident_entries:
            assert candidate.resident_bytes < full.resident_bytes and candidate.resident_entries < full.resident_entries
    assert collect(tf).fingerprint == roots.fingerprint


def test_comparison_does_not_hide_large_retained_candidate_root():
    tf, net = fixture()
    roots = collect(tf)
    witness, _ = lower_bound_roots(roots, net)
    lower = measure(witness)
    assert roots.measure().resident_bytes < lower.resident_bytes
    tf.required_pending = np.ones(lower.resident_bytes, dtype=np.uint8)
    assert collect(tf).measure().resident_bytes > lower.resident_bytes


def test_largest_witness_keeps_full_other_operator_and_shared_kernel_in_reference():
    tf, net = fixture()
    second_params = dict(weight=torch.ones((4, 4, 1, 1), dtype=torch.float64), input_shape=(1, 4, 12, 12))
    second = ImplicitConv2DOp(second_params['weight'].numpy(), second_params['input_shape'])
    tf.second = tf.same_second = second
    tf.model_kernel = tf.explicit_op._kernel
    net.layers.append(SimpleNamespace(id=8, kind='CONV2D', params=second_params))
    roots = collect(tf)
    witness, records = lower_bound_roots(roots, net)
    complete, _ = expanded_roots(roots, net)
    assert records[0]['reachable_implicit_objects'] == 2
    assert records[0]['selected_layer_for_provenance_only'] == 7
    assert records[0]['all_reference_logical_nnz'] > records[0]['constructed_reference_nnz']
    assert measure(witness).resident_bytes < measure(complete).resident_bytes
    assert measure(witness).resident_entries < measure(complete).resident_entries
    assert roots.measure().resident_bytes < measure(witness).resident_bytes
    assert roots.measure().resident_entries < measure(witness).resident_entries


@pytest.mark.parametrize('cap', [0, 64_000_001])
def test_budget_rejects_before_any_reference_allocation(monkeypatch, cap):
    tf, net = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected reference allocation')
    monkeypatch.setattr(cnn, 'sparse_conv2d_matrix_from_layer_csr', forbidden)
    with pytest.raises((ValueError, MemoryError)):
        lower_bound_roots(collect(tf), net, max_nnz=cap)


def test_coefficient_corruption_rejects(monkeypatch):
    tf, net = fixture()
    original = cnn.sparse_conv2d_matrix_from_layer_csr
    def corrupt(*args, **kwargs):
        matrix, bias = original(*args, **kwargs)
        matrix.data[1] = np.nextafter(matrix.data[1], np.inf)
        return matrix, bias
    monkeypatch.setattr(cnn, 'sparse_conv2d_matrix_from_layer_csr', corrupt)
    with pytest.raises(ValueError, match='coefficient mismatch'):
        lower_bound_roots(collect(tf), net)


def test_absent_or_unmatched_witness_rejects():
    tf, net = fixture()
    with pytest.raises(ValueError, match='no reachable'):
        lower_bound_roots(collect(SimpleNamespace(value=np.ones(3))), net)
    net.layers = []
    with pytest.raises(ValueError, match='unmatched'):
        lower_bound_roots(collect(tf), net)
