from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import tf_cnn as cnn, tf_mlp as mlp
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c9_live_relu_audit_v1 import verify_plain
from experiments.neural_hz_20260831.c9_integrated_suffix_audit_v1 import audit
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture
from experiments.neural_hz_20260831.test_c9_integrated_suffix_v1 import rational_outputs
from experiments.neural_hz_20260831.test_c7_factored_hz_v1 import rational_original


def frame():
    tf = HybridzTF()
    tf._sparse_frame_widths = {7: (12, 6), 29: (3, 1)}
    tf._sparse_relu_slots = {(7, -8, 3): (9, 10, 5)}
    tf._neural_hz_transient_relu_input = True
    return tf


def test_default_off_preserves_hooks_and_does_not_touch_input(monkeypatch):
    old = (HybridzTF.apply, cnn._lazy_materialize)
    monkeypatch.setattr(runtime, 'lift', lambda *a, **k: pytest.fail('default off constructed'))
    with runtime.installed():
        assert (HybridzTF.apply, cnn._lazy_materialize) == old
    assert (HybridzTF.apply, cnn._lazy_materialize) == old


@pytest.mark.parametrize('guard', ['budget', 'frame', 'terms', 'no_ops', 'no_conv', 'distinct', 'square', 'csc'])
def test_nonmatching_structures_do_not_select(guard):
    expr, op = fixture()
    limit = 64_000_000
    assert runtime.supported(expr, limit)
    # Malformed-input guards must be reached without the native immutable
    # constructor rejecting the intentionally malformed fixture first.
    expr = SimpleNamespace(**vars(expr))
    if guard == 'budget':
        limit -= 1
    elif guard == 'frame':
        expr.frame_id = None
    elif guard == 'terms':
        expr.terms = ()
    elif guard == 'no_ops':
        expr.terms = (cnn.SparseHZAffineTerm(expr.terms[0].source, ()),)
    elif guard == 'no_conv':
        expr.terms = tuple(cnn.SparseHZAffineTerm(t.source, (t.operators[-1],)) for t in expr.terms)
    elif guard == 'distinct':
        expr.terms = tuple(cnn.SparseHZAffineTerm(t.source, (*t.operators[:-1], t.operators[-1].copy())) for t in expr.terms)
    else:
        reducer = sp.eye(expr.n_out, format='csr') if guard == 'square' else expr.terms[0].operators[-1].tocsc()
        expr.terms = tuple(cnn.SparseHZAffineTerm(t.source, (*t.operators[:-1], reducer)) for t in expr.terms)
    assert not runtime.supported(expr, limit)


@pytest.mark.parametrize('dyadic', [True, False])
def test_one_definition_graph_exact_views_and_old_global_frame(monkeypatch, dyadic):
    expr, op = fixture(dyadic)
    tf, retained, calls = frame(), [], []
    actual_lift = runtime.lift
    def counted(*args, **kwargs):
        calls.append(1)
        return actual_lift(*args, **kwargs)
    monkeypatch.setattr(runtime, 'lift', counted)
    masks = [np.arange(expr.n_out) % 2 == 0, np.ones(expr.n_out, bool), np.zeros(expr.n_out, bool)]
    def fake_apply(self, layer, *args):
        return [cnn._lazy_materialize(expr, keep, 64_000_000) for keep in masks]
    monkeypatch.setattr(HybridzTF, 'apply', fake_apply)
    with runtime.installed(enabled=True, ready=retained.append):
        views = HybridzTF.apply(tf, SimpleNamespace(id=123, kind='RELU'))
    lifted = retained[0]['lifted']
    assert len(calls) == 1 and lifted.old_n_cont == 12 and lifted.old_n_bin == 6
    assert audit(lifted)['all_original_coefficients_exact']
    assert rational_outputs(lifted) == rational_original(expr, 12, 6, lifted.keep)
    for view, mask in zip(views, masks):
        assert np.array_equal(view.c, expr.bias)
        assert (view.Gc[mask] != lifted.hz.Gc[mask]).nnz == 0 and view.Gc[~mask].nnz == 0
        for key in ('Ac', 'Ab', 'Auc', 'Aub'):
            assert getattr(view, key) is getattr(lifted.hz, key)
        for key in ('b', 'ub'):
            assert np.shares_memory(getattr(view, key), getattr(lifted.hz, key))
            assert np.array_equal(getattr(view, key), getattr(lifted.hz, key))
        assert view.frame_id == 7 and view.n_cont == lifted.hz.n_cont and view.n_bin == 6
    assert tf._sparse_frame_widths == {7: (12, 6), 29: (3, 1)}
    assert tf._sparse_relu_slots == {(7, -8, 3): (9, 10, 5)}


@pytest.mark.parametrize('layer_id', [1, 78, 9999])
def test_native_phase_probe_followup_and_disjoint_relu_slots(monkeypatch, layer_id):
    expr, op = fixture()
    tf, retained = frame(), []
    tf._neural_hz_sparse_phase_selective_materialization = True
    tf._neural_hz_sparse_phase_separated_relu = True
    bounds = Bounds(torch.tensor([-2., .2, -2., -2., .1, -2.], dtype=torch.float64),
                    torch.tensor([2., 2., -1., 2., 2., 2.], dtype=torch.float64))
    layer = SimpleNamespace(id=layer_id, kind='RELU')
    def fake_apply(self, selected, *args):
        return cnn.sparse_hz_apply_affine_expr_layer(selected, expr, bounds, None, self)
    monkeypatch.setattr(HybridzTF, 'apply', fake_apply)
    with runtime.installed(enabled=True, ready=retained.append):
        handled, actual, separated, reason = HybridzTF.apply(tf, layer)
    state = retained[0]
    assert handled and actual is not None and separated is None and reason is None
    assert len(state['views']) == 2
    check = verify_plain(state['views'][-1], bounds, layer, actual, tf, state['entry_widths'], state['entry_slots'])
    assert check['new_slots_disjoint_from_c9'] and check['old_phase_slots_unchanged']
    assert check['new_phase_binaries'] > 0


@pytest.mark.parametrize('mutation', ['source', 'expression', 'width', 'slots', 'mask', 'construction'])
def test_selected_failure_never_falls_back_and_restores_hooks(monkeypatch, mutation):
    expr, op = fixture()
    tf = frame()
    def forbidden_native(*args, **kwargs):
        pytest.fail('selected failure retried native path')
    monkeypatch.setattr(cnn, '_lazy_materialize', forbidden_native)
    if mutation == 'construction':
        def failed(*a, **k):
            raise MemoryError('test work cap')
        monkeypatch.setattr(runtime, 'lift', failed)
    def fake_apply(self, layer):
        mask = np.ones(expr.n_out, bool)
        cnn._lazy_materialize(expr, mask, 64_000_000)
        next_expr = expr
        if mutation == 'source':
            expr.terms[0].source.ub[0] += 1
        elif mutation == 'expression':
            next_expr = cnn.SparseHZAffineExpr(expr.terms, expr.bias, expr.n_out, expr.frame_id)
        elif mutation == 'width':
            tf._sparse_frame_widths[7] = (13, 6)
        elif mutation == 'slots':
            tf._sparse_relu_slots[(7, 2, 1)] = (12, 13, 6)
        elif mutation == 'mask':
            mask = mask.astype(int)
        return cnn._lazy_materialize(next_expr, mask, 64_000_000)
    monkeypatch.setattr(HybridzTF, 'apply', fake_apply)
    old = (HybridzTF.apply, cnn._lazy_materialize)
    with pytest.raises(runtime.SelectedRejected):
        with runtime.installed(enabled=True):
            HybridzTF.apply(tf, SimpleNamespace(id=2, kind='RELU'))
    assert (HybridzTF.apply, cnn._lazy_materialize) == old


def test_unsupported_and_outside_apply_delegate_without_construction(monkeypatch):
    expr, op = fixture()
    calls = []
    monkeypatch.setattr(runtime, 'lift', lambda *a, **k: pytest.fail('nonselected construction'))
    monkeypatch.setattr(cnn, '_lazy_materialize', lambda *a, **k: calls.append(a) or 'native')
    monkeypatch.setattr(HybridzTF, 'apply', lambda self, layer: cnn._lazy_materialize(expr, np.ones(expr.n_out, bool), 63))
    with runtime.installed(enabled=True):
        assert cnn._lazy_materialize(expr, np.ones(expr.n_out, bool), 64_000_000) == 'native'
        assert HybridzTF.apply(frame(), SimpleNamespace(id=78)) == 'native'
    assert len(calls) == 2
