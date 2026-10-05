import pickle
from types import SimpleNamespace
from fractions import Fraction
import numpy as np
import pytest
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import c10_live_runtime_v1 as runtime
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as base
from experiments.neural_hz_20260831.c10_portable_binding_v1 import identity, verify, reconstruct_fraction
from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c10_fused_emission_v1 import lift
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c9_live_relu_audit_v1 import verify_plain
from experiments.neural_hz_20260831.test_c10_fused_emission_v1 import expr_fixture
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture
from experiments.neural_hz_20260831.test_c9_live_runtime_v1 import frame


def candidate():
    return lift(expr_fixture(), np.ones(2, bool), enabled=True)


def test_portable_identity_preserves_pickle_sharing_not_addresses():
    old = candidate()
    fresh = pickle.loads(pickle.dumps(old))
    fresh.origin_binding = expression_binding(fresh.expression)
    fresh.seal = fresh.fingerprint()
    assert identity(fresh) == identity(old)
    bound = {'identity': identity(old), 'all_rows_proved': True, 'result_sha256': 'fixture'}
    assert verify(fresh, bound)['all_content_and_sharing_bound']


def test_equal_valued_but_unshared_operator_does_not_match_proof():
    old = candidate()
    bound = identity(old)
    term = old.expression.terms[0]
    copied = term.operators[0].copy()
    old.expression = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(term.source,
        (copied, *term.operators[1:])),), old.expression.bias, old.expression.n_out, old.expression.frame_id)
    old.origin_binding = expression_binding(old.expression)
    old.seal = old.fingerprint()
    assert identity(old) != bound


@pytest.mark.parametrize('mutation', ['hash', 'proof', 'ratio', 'report'])
def test_portable_proof_rejects_changed_payload_or_proof(mutation):
    old = candidate()
    bound = {'identity': identity(old), 'all_rows_proved': True, 'result_sha256': 'fixture'}
    if mutation == 'hash':
        bound['identity']['hz_sha256'] = 'changed'
    elif mutation == 'proof':
        bound['all_rows_proved'] = False
    elif mutation == 'ratio':
        tag = np.flatnonzero(old.eq_roots < 0)[0]
        old.eq_scales.view(np.float64)[tag] *= .5
        old.seal = old.fingerprint()
    else:
        old.report['total_work_upper'] += 1
        old.seal = old.fingerprint()
    with pytest.raises(ValueError):
        verify(old, bound)


def test_reconstruct_without_old_hz_preserves_original_prefix():
    old = candidate()
    x = [Fraction(1, 3)] * old.hz.n_cont
    got = reconstruct_fraction(old, x)
    cols, parents, ratios, unused = aliases(old)
    assert got[:old.old_n_cont] == x[:old.old_n_cont]
    for c, p, r in zip(cols, parents, ratios):
        assert got[int(c)] == Fraction(float(r)) * x[int(p)]
    assert all(got[c] == x[c] for c in range(len(x)) if c not in cols)
    assert all(abs(v) <= 1 for v in got)
    with pytest.raises(ValueError, match='width'):
        reconstruct_fraction(old, x[:-1])
    with pytest.raises(ValueError, match='box'):
        reconstruct_fraction(old, [2.] * len(x))


def test_scoped_hook_default_off_and_exception_restore():
    original = (base.lift, HybridzTF.apply, cnn._lazy_materialize)
    with runtime.installed():
        assert (base.lift, HybridzTF.apply, cnn._lazy_materialize) == original
    with pytest.raises(RuntimeError):
        with runtime.installed(enabled=True):
            assert base.lift is runtime.fused_lift
            raise RuntimeError('fixture')
    assert (base.lift, HybridzTF.apply, cnn._lazy_materialize) == original


@pytest.mark.parametrize('layer_id', [78, 9123])
def test_real_native_relu_with_fused_graph_and_shared_phase_slots(monkeypatch, layer_id):
    expr, op = fixture()
    tf, retained = frame(), []
    tf._neural_hz_sparse_phase_selective_materialization = True
    tf._neural_hz_sparse_phase_separated_relu = True
    bounds = Bounds(torch.tensor([-2., .2, -2., -2., .1, -2.], dtype=torch.float64),
                    torch.tensor([2., 2., -1., 2., 2., 2.], dtype=torch.float64))
    layer = SimpleNamespace(id=layer_id, kind='RELU')
    def fake_apply(self, current, *args):
        return cnn.sparse_hz_apply_affine_expr_layer(current, expr, bounds, None, self)
    monkeypatch.setattr(HybridzTF, 'apply', fake_apply)
    with runtime.installed(enabled=True, ready=retained.append):
        handled, actual, separated, reason = HybridzTF.apply(tf, layer)
    state = retained[0]
    assert handled and actual is not None and separated is None and reason is None
    assert state['lifted'].report['alias_quotient']['selected_aliases'] > 0
    assert len(state['views']) == 2
    proof = verify_plain(state['views'][-1], bounds, layer, actual, tf, state['entry_widths'], state['entry_slots'])
    assert proof['old_phase_slots_unchanged'] and proof['new_slots_disjoint_from_c9']


def test_selected_fused_failure_has_no_native_fallback(monkeypatch):
    expr, op = fixture()
    monkeypatch.setattr(runtime, 'fused_lift', lambda *a, **k: (_ for _ in ()).throw(MemoryError('cap')))
    monkeypatch.setattr(cnn, '_lazy_materialize', lambda *a, **k: pytest.fail('fallback'))
    monkeypatch.setattr(HybridzTF, 'apply', lambda self, layer:
        cnn._lazy_materialize(expr, np.ones(expr.n_out, bool), 64_000_000))
    with pytest.raises(runtime.SelectedRejected):
        with runtime.installed(enabled=True):
            HybridzTF.apply(frame(), SimpleNamespace(id=78))
