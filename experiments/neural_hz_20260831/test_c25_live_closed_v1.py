from dataclasses import fields
from fractions import Fraction
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import c25_live_runtime_v1 as runtime
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as base
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.c24_dense_emission_v1 import lift
from experiments.neural_hz_20260831.c24_closed_state_v1 import close, export, Closed
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c10_portable_binding_v1 import reconstruct_fraction
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c9_live_relu_audit_v1 import verify_plain
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture
from experiments.neural_hz_20260831.test_c9_live_runtime_v1 import frame


def proof_fixture():
    expr, op = fixture()
    keep = np.ones(expr.n_out, bool)
    draft = lift(expr, keep, enabled=True, frame_widths=(12,6))
    original = original_lift(expr, keep, enabled=True, frame_widths=(12,6))
    maps = {k:getattr(original,k) for k in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')}
    closed, proof = close(draft, original.hz, maps, enabled=True)
    unused, raw = export(closed)
    return expr, raw, hashlib.sha256(raw).hexdigest()


def test_default_off_does_not_read_proof_or_change_hooks():
    old = (base.lift, HybridzTF.apply, cnn._lazy_materialize, cnn._sparse_apply_relu)
    assert bind(object(), object(), expected_proof_sha256=object()) is None
    with runtime.installed(proof_bytes=object()):
        assert (base.lift, HybridzTF.apply, cnn._lazy_materialize, cnn._sparse_apply_relu) == old


def test_fresh_binding_preserves_own_buffers_and_nonzero_fraction_extension():
    expr, raw, sha = proof_fixture()
    fresh = lift(expr, np.ones(expr.n_out,bool), enabled=True, frame_widths=(12,6))
    closed, report = bind(fresh, raw, expected_proof_sha256=sha, enabled=True)
    assert closed.hz is fresh.hz and closed.owners is fresh.owners
    assert not report['archived_HZ_loaded_or_substituted']
    assert reconstruct_fraction(closed, [Fraction(1,3)]*closed.hz.n_cont) == reconstruct_fraction(fresh, [Fraction(1,3)]*fresh.hz.n_cont)
    assert type(closed) is Closed and not hasattr(closed,'nodes')


@pytest.mark.parametrize('changed', ['proof','missing','hz','source','owners','slabs','map','report','sharing','frame'])
def test_binding_rejects_changes_even_after_fresh_draft_reseal(changed):
    expr, raw, sha = proof_fixture()
    fresh = lift(expr, np.ones(expr.n_out,bool), enabled=True, frame_widths=(12,6))
    if changed == 'proof': raw += b' '
    elif changed == 'missing': raw = None
    elif changed == 'hz': fresh.hz.b[0] += .125
    elif changed == 'source': fresh.expression.bias[0] += .125
    elif changed == 'owners': fresh.owners[0] += 1
    elif changed == 'slabs': fresh.uid_slabs[0] += np.uint64(1)
    elif changed == 'map': fresh.eq_scales[0] += 1
    elif changed == 'report': fresh.report['total_work_upper'] += 1
    elif changed == 'sharing':
        terms = list(fresh.expression.terms)
        t = terms[0]
        terms[0] = cnn.SparseHZAffineTerm(t.source, (*t.operators[:-1], t.operators[-1].copy()))
        fresh.expression = cnn.SparseHZAffineExpr(tuple(terms),fresh.expression.bias,fresh.expression.n_out,fresh.expression.frame_id)
    else: fresh.hz.frame_id += 1
    fresh.origin_binding = expression_binding(fresh.expression)
    try: fresh.seal = fresh.fingerprint()
    except ValueError: pass
    with pytest.raises(ValueError): bind(fresh,raw,expected_proof_sha256=sha,enabled=True)


def test_missing_proof_rejected_before_hook_installation():
    old = (base.lift, HybridzTF.apply, cnn._lazy_materialize)
    with pytest.raises(ValueError):
        with runtime.installed(enabled=True): pass
    assert (base.lift, HybridzTF.apply, cnn._lazy_materialize) == old


def run_native(monkeypatch, layer_id=78, *, ready=None, corrupt=None):
    expr, raw, sha = proof_fixture()
    tf, retained = frame(), []
    tf._neural_hz_sparse_phase_selective_materialization = True
    tf._neural_hz_sparse_phase_separated_relu = True
    bounds = Bounds(torch.tensor([-2.,.2,-2.,-2.,.1,-2.],dtype=torch.float64),
                    torch.tensor([2.,2.,-1.,2.,2.,2.],dtype=torch.float64))
    layer = SimpleNamespace(id=layer_id,kind='RELU')
    def apply(self,current,*args):
        result = cnn.sparse_hz_apply_affine_expr_layer(current,expr,bounds,None,self)
        if corrupt == 'rhs': result[1].b[0] += .125
        if corrupt != 'missing_cache': self._sparse_hz_cache[current.id] = result[1]
        return result
    monkeypatch.setattr(HybridzTF,'apply',apply)
    def on_ready(state):
        if ready is not None: ready(state)
        retained.append(state)
    with runtime.installed(enabled=True,proof_bytes=raw,expected_proof_sha256=sha,ready=on_ready):
        handled, actual, separated, reason = HybridzTF.apply(tf,layer)
    return tf,retained[0],actual,bounds,layer


@pytest.mark.parametrize('layer_id',[1,78,9123])
def test_actual_native_relu_and_complete_sparse_post_ownership(monkeypatch,layer_id):
    tf,state,actual,bounds,layer = run_native(monkeypatch,layer_id)
    closed = state['lifted']
    assert type(closed) is Closed and len(state['views']) == 2
    assert state['closed_binding']['all_new_graph_fields_physically_retired']
    check = verify_plain(state['views'][-1],bounds,layer,actual,tf,state['entry_widths'],state['entry_slots'])
    assert check['new_slots_disjoint_from_c9'] and check['old_phase_slots_unchanged']
    overlay = runtime.phase_overlay(state)
    eq,le = closed_uid_tables(closed)
    ne,nl = actual.n_eq-closed.hz.n_eq,actual.n_ineq-closed.hz.n_ineq
    first = closed.report['radix_uid_base']+16384
    expected = actual_words(actual,closed.old_n_cont,closed.logical_n_cont,
        np.r_[eq,np.arange(first,first+ne)],np.r_[le,np.arange(first+ne,first+ne+nl)])
    assert np.array_equal(list(overlay.iter_words(pool=WorkPool(0,0))),expected)
    roots = runtime.numeric_roots(state)
    assert roots['phase_ownership']['post_hz'] is actual
    assert 'receipt' not in roots['phase_ownership']
    assert overlay.base is closed.owners
    assert state['phase_ownership']['report']['whole_generation_plus_event_work'] <= 256_000_000
    assert state['phase_ownership']['construction']['measured_transient_gate']


@pytest.mark.parametrize('changed',['events','post','base','receipt','report','construction'])
def test_phase_mutation_and_outer_reseal_do_not_pass(monkeypatch,changed):
    tf,state,actual,bounds,layer = run_native(monkeypatch)
    owned = state['phase_ownership']
    if changed == 'events':
        owned['events'] = owned['events'].copy()
        owned['events'][0] += np.uint64(1)
        owned['event_sha256'] = hashlib.sha256(owned['events'].tobytes()).hexdigest()
    elif changed == 'post':
        actual.b[0] += .125
        from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
        owned['post_sha256'] = source_digest(actual)
    elif changed == 'base': owned['base'] = owned['base'].copy()
    elif changed == 'receipt': owned['receipt'] = object()
    elif changed == 'report': owned['report']['whole_generation_plus_event_work'] = 0
    else: owned['construction']['elapsed_s'] = 0.
    with pytest.raises(ValueError): runtime.phase_overlay(state)


@pytest.mark.parametrize('bad',['rhs','missing_cache'])
def test_incomplete_or_changed_native_result_rejected(monkeypatch,bad):
    old = (base.lift,cnn._lazy_materialize,cnn._sparse_apply_relu)
    with pytest.raises(runtime.SelectedRejected): run_native(monkeypatch,corrupt=bad)
    assert (base.lift,cnn._lazy_materialize,cnn._sparse_apply_relu) == old


def test_selected_failed_binding_does_not_fallback_or_publish(monkeypatch):
    expr,raw,sha = proof_fixture()
    tf = frame()
    actual_lift = runtime.fresh_lift
    def changed(*a,**k):
        draft = actual_lift(*a,**k)
        draft.report['total_work_upper'] += 1
        draft.seal = draft.fingerprint()
        return draft
    monkeypatch.setattr(runtime,'fresh_lift',changed)
    monkeypatch.setattr(cnn,'_lazy_materialize',lambda *a,**k: pytest.fail('selected fallback'))
    monkeypatch.setattr(HybridzTF,'apply',lambda self,layer: cnn._lazy_materialize(expr,np.ones(expr.n_out,bool),64_000_000))
    with pytest.raises(runtime.SelectedRejected):
        with runtime.installed(enabled=True,proof_bytes=raw,expected_proof_sha256=sha):
            HybridzTF.apply(tf,SimpleNamespace(id=78,kind='RELU'))
    assert 78 not in tf._sparse_hz_cache


def test_zero_hit_and_outside_apply_preserve_native_path(monkeypatch):
    expr,raw,sha = proof_fixture()
    calls=[]
    monkeypatch.setattr(runtime,'fresh_lift',lambda *a,**k: pytest.fail('zero-hit generation'))
    monkeypatch.setattr(cnn,'_lazy_materialize',lambda *a,**k: calls.append(a) or 'native')
    monkeypatch.setattr(HybridzTF,'apply',lambda self,layer: cnn._lazy_materialize(expr,np.ones(expr.n_out,bool),63))
    with runtime.installed(enabled=True,proof_bytes=raw,expected_proof_sha256=sha):
        assert cnn._lazy_materialize(expr,np.ones(expr.n_out,bool),64_000_000) == 'native'
        assert HybridzTF.apply(frame(),SimpleNamespace(id=78)) == 'native'
    assert len(calls)==2


def test_nonmatching_nested_hook_and_context_exception_restore(monkeypatch):
    expr,raw,sha = proof_fixture()
    old=(base.lift,HybridzTF.apply,cnn._lazy_materialize,cnn._sparse_apply_relu)
    with pytest.raises(RuntimeError):
        with runtime.installed(enabled=True,proof_bytes=raw,expected_proof_sha256=sha):
            with pytest.raises(ValueError):
                with runtime.installed(enabled=True,proof_bytes=raw,expected_proof_sha256=sha): pass
            raise RuntimeError('fixture')
    assert (base.lift,HybridzTF.apply,cnn._lazy_materialize,cnn._sparse_apply_relu)==old


def test_complete_live_checkpoint_is_portable_without_opaque_tokens(monkeypatch):
    from experiments.neural_hz_20260831.c25_live_relu_worker_v1 import checkpoint_fields, audit_actual_phase
    from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
    tf,state,actual,bounds,layer = run_native(monkeypatch)
    columns, report = audit_actual_phase(state)
    assert report['complete_post_incidence_equal'] and report['new_phase_executed']
    assert report['all_post_MAIN_columns_checked'] == len(state['lifted'].owners)
    payload = checkpoint_fields(state)
    roundtrip = pickle.loads(pickle.dumps(payload,protocol=5))
    assert 'receipt' not in roundtrip['phase_ownership']
    assert 'definition_graph' not in roundtrip and 'nodes' not in roundtrip['closed_fields']
    anchored = payload['closed_proof_sha256']
    restored = restore(roundtrip['closed_fields'],roundtrip['closed_proof_bytes'],expected_proof_sha256=anchored)
    restored.validate()
    assert restored.fingerprint() == state['lifted'].fingerprint()
    assert roundtrip['phase_ownership']['base'] is restored.owners
    assert roundtrip['numeric_roots']['phase_ownership']['post_hz'] is roundtrip['phase_ownership']['post_hz']
    with pytest.raises(TypeError): pickle.dumps(state['phase_ownership']['receipt'])
    with pytest.raises(TypeError): pickle.dumps(state['lifted'].receipt)


def test_phase_failure_cannot_return_a_selected_result_or_fallback(monkeypatch):
    old = (base.lift,cnn._lazy_materialize,cnn._sparse_apply_relu)
    def exhausted(*args,**kwargs): raise MemoryError('frozen coupled work exhausted')
    monkeypatch.setattr(runtime,'build_overlay',exhausted)
    with pytest.raises(runtime.SelectedRejected): run_native(monkeypatch)
    assert (base.lift,cnn._lazy_materialize,cnn._sparse_apply_relu) == old


def test_closed_identity_and_uid_ceiling_are_bound_to_actual_phase(monkeypatch):
    tf,state,actual,bounds,layer = run_native(monkeypatch)
    state['phase_ownership']['old_uid_ceiling'] += 1
    with pytest.raises(ValueError): runtime.phase_overlay(state)
    with pytest.raises(ValueError): runtime._PhaseBinding(object())
