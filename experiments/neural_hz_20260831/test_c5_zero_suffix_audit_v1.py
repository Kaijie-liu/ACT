import copy
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import tf_cnn as cnn, tf_mlp as mlp
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import verify_zero_transfer, verify_negative_relu


def fixture():
    op = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 2, 2))
    one, two = source(8), source(8)
    two.b[0] = .25
    d = sp.diags(np.arange(8) / 7., format='csr')
    term = cnn.SparseHZAffineTerm(one, (op, d, op, d, op, d))
    expr = cnn.SparseHZAffineExpr((term, term, cnn.SparseHZAffineTerm(two, (op, d))),
        np.arange(8) / 3.7, 8, one.frame_id)
    return expr, op


def test_mixed_length_paths_and_repeated_sources_need_no_operator_rows(monkeypatch):
    expr, op = fixture()
    def forbidden(*args):
        raise AssertionError('zero map requested a Conv row')
    monkeypatch.setattr(ImplicitConv2DOp, '_row', forbidden)
    actual = cnn._lazy_materialize(expr, np.zeros(expr.n_out, dtype=bool), 64_000_000)
    assert all(verify_zero_transfer(expr, actual).values())
    assert actual.n_bin == 2 and actual.Gc.nnz == actual.Gb.nnz == 0


@pytest.mark.parametrize('mutation', ['center', 'predicate', 'frame'])
def test_zero_transfer_corruption_rejects(mutation):
    expr, op = fixture()
    actual = cnn._lazy_materialize(expr, np.zeros(expr.n_out, dtype=bool), 64_000_000)
    if mutation == 'center':
        actual.c[0] += 1.
    elif mutation == 'predicate':
        actual.ub[0] += .5
    else:
        actual.frame_id += 1
    with pytest.raises(ValueError):
        verify_zero_transfer(expr, actual)


def test_all_negative_actual_relu_keeps_binary_predicates_and_slot_map():
    expr, op = fixture()
    pre = cnn._lazy_materialize(expr, np.zeros(expr.n_out, dtype=bool), 64_000_000)
    bounds = Bounds(torch.full((1, 8), -2.), torch.full((1, 8), -1.))
    tf = HybridzTF()
    tf._sparse_frame_widths[expr.frame_id] = (pre.n_cont, pre.n_bin)
    tf._sparse_relu_slots[(expr.frame_id, 4, 0)] = (0, 1, 1)
    before = dict(tf._sparse_relu_slots)
    actual, reason = mlp._sparse_apply_relu(SimpleNamespace(id=10), pre, bounds, tf, forced_stable_negative=np.ones(8, dtype=bool))
    assert reason is None and tf._sparse_relu_slots == before
    assert all(verify_negative_relu(pre, actual, bounds, tf._sparse_frame_widths).values())
    invalid = copy.deepcopy(bounds)
    invalid.ub[0, 0] = .1
    with pytest.raises(ValueError, match='authoritative'):
        verify_negative_relu(pre, actual, invalid, tf._sparse_frame_widths)
