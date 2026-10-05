from types import SimpleNamespace

import numpy as np
import pytest
import torch

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c7_factored_hz_v1 import lift
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import audit, reference_subset
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import expanded_roots
from experiments.neural_hz_20260831.test_c5_reference_lower_bound_v1 import fixture as reference_fixture, measure
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture


def test_every_actual_definition_matches_original_unfused_rows():
    expr, op = fixture(False)
    lifted = lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    record = audit(lifted)
    assert record['all_defining_rows_checked'] == lifted.hz.n_cont - lifted.old_n_cont
    assert record['all_original_dyadic_coefficients_exact'] and record['all_old_predicates_preserved']
    numeric = lifted.numeric_roots()
    assert len(numeric) >= 4 * len(lifted.nodes)
    assert collect(SimpleNamespace(), numeric).measure().resident_bytes > 0


@pytest.mark.parametrize('mutation', ['coefficient', 'predicate', 'root', 'box'])
def test_independent_audit_rejects_corruption_even_after_local_resealing(mutation):
    expr, op = fixture(False)
    lifted = lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    if mutation == 'coefficient':
        lifted.hz.b[lifted.old_n_eq] = np.nextafter(lifted.hz.b[lifted.old_n_eq], np.inf)
    elif mutation == 'predicate':
        lifted.hz.ub[0] += .1
    elif mutation == 'root':
        lifted.hz.c[0] += .1
    else:
        first = lifted.nodes[0]
        index = np.flatnonzero(first['needed'])[0]
        first['exponents'][index] -= 10
    lifted.seal = lifted.fingerprint()
    with pytest.raises(ValueError):
        audit(lifted)


def test_candidate_never_calls_original_conv_expansion_or_row_oracle(monkeypatch):
    expr, op = fixture(False)
    def forbidden(*args):
        raise AssertionError('candidate requested expanded/native Conv row')
    monkeypatch.setattr(ImplicitConv2DOp, '_row', forbidden)
    monkeypatch.setattr(ImplicitConv2DOp, 'to_csr_reference', forbidden)
    assert lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True).report['exact_power_two_coefficients']


def test_numeric_report_cannot_hide_an_unregistered_array():
    expr, op = fixture()
    lifted = lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    lifted.report['unregistered'] = np.ones(8)
    with pytest.raises(ValueError):
        lifted.numeric_roots()


def two_leaves():
    tf, net = reference_fixture()
    params = dict(weight=torch.ones((4, 4, 1, 1), dtype=torch.float64), input_shape=(1, 4, 12, 12))
    second = ImplicitConv2DOp(params['weight'].numpy(), params['input_shape'])
    tf.second = second
    net.layers.append(SimpleNamespace(id=8, kind='CONV2D', params=params))
    return tf, net


def test_two_leaf_lower_bound_is_below_the_complete_same_reference():
    tf, net = two_leaves()
    # A same-content descriptor must not create a second supposedly unique leaf.
    tf.duplicate = ImplicitConv2DOp(tf.second._kernel, tf.second.input_shape)
    roots = collect(tf)
    witness, record = reference_subset(roots, net)
    complete, _ = expanded_roots(roots, net)
    assert len(record['leaves']) == 2 and len(witness) == 2
    assert measure(witness).resident_bytes <= measure(complete).resident_bytes
    assert measure(witness).resident_entries <= measure(complete).resident_entries
    assert collect(tf).fingerprint == roots.fingerprint


def test_reference_aggregate_cap_rejects_even_when_each_leaf_would_fit():
    tf, net = two_leaves()
    limit = tf.explicit_op.logical_expanded_nnz
    with pytest.raises(MemoryError, match='aggregate'):
        reference_subset(collect(tf), net, max_nnz=limit)
