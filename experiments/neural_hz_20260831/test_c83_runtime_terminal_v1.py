"""Actual new-state binding and ordinary terminal observation, no solver rescue."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from act.back_end.solver import solver_hz as backend
from act.front_end.specs import OutKind
from experiments.neural_hz_20260831.c83_runtime_final_binding_v1 import verify, final_extra
from experiments.neural_hz_20260831.c83_terminal_observer_v1 import installed
from experiments.neural_hz_20260831.c82_native_terminal_v2 import affine
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c9_live_runtime_v1 import SelectedRejected
from experiments.neural_hz_20260831.test_c82_native_terminal_v2 import fixture as base


def pool():
    return WorkPool(256_000_000)


def fixture(ids=(78, 79, 80)):
    new, net, producer, final, inp, shape, spec = base(ids)
    proof = affine(new, net, producer, final, inp, shape, spec, pool=pool(), enabled=True)
    raw = json.dumps(proof, sort_keys=True).encode()
    tf = SimpleNamespace(_net=net, _sparse_hz_cache={ids[0]: new.hz, ids[1]: final},
                         _sparse_affine_expr_cache={})
    state = dict(lifted=new, tf=tf, layer=net.layers[0], native_block_calls=1)
    kw = dict(input_shape=shape, batch_size=1, n_out=final.n_out, timelimit=45.)
    return state, final, inp, spec, kw, raw, hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('ids', [(78, 79, 80), (3, 8, 15), (812, 932, 1210)])
def test_complete_new_native_runtime_binding_and_prepaid_cost_without_id_menu(ids):
    args = fixture(ids)
    budget = pool()
    report = verify(*args, pool=budget, enabled=True)
    assert report['reconstructable_native_local_state_preserved']
    assert not report['base_feasibility_shortcut']
    assert budget.used + 512 == final_extra(len(args[0]['tf']._net.layers))


@pytest.mark.parametrize('bad', ['final', 'input', 'property', 'weight', 'bias', 'cache',
                                'predicate_copy', 'RHS_copy', 'topology', 'shape', 'budget', 'proof', 'cap'])
def test_changed_final_problem_or_actual_publication_rejected(bad):
    state, final, inp, spec, kw, raw, sha = fixture()
    budget = pool()
    if bad == 'final': final.c[0] += .125
    elif bad == 'input': inp.c[0] += .125
    elif bad == 'property': spec.kind = OutKind.LINEAR_LE
    elif bad == 'weight': state['tf']._net.layers[1].params['weight'][0, 0] += .125
    elif bad == 'bias': state['tf']._net.layers[1].params['bias'][0] += .125
    elif bad == 'cache': state['tf']._sparse_hz_cache.pop(79)
    elif bad == 'predicate_copy': final.Ac = final.Ac.copy()
    elif bad == 'RHS_copy': final.b = final.b.copy()
    elif bad == 'topology': state['tf']._net.preds[79] = []
    elif bad == 'shape': kw['input_shape'] = (inp.n_out, 1)
    elif bad == 'budget': kw['timelimit'] = 46.
    elif bad == 'proof': raw += b' '
    else: budget = WorkPool(0)
    with pytest.raises((ValueError, MemoryError)):
        verify(state, final, inp, spec, kw, raw, sha, pool=budget, enabled=True)


@pytest.mark.parametrize('status', [1, 2])
def test_base_unknown_does_not_create_property_or_extra_solve(monkeypatch, status):
    state, final, inp, spec, kw, *_ = fixture()
    model = backend._lower_hz_milp(final)
    proof = dict(native_lowered_n_cont=model.n_cont, native_lowered_n_bin=model.n_bin)
    calls, events = [], []
    def solve(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=status, message='fixture', success=False, x=None, mip_node_count=0)
    monkeypatch.setattr(backend, 'milp', solve)
    with installed(state['lifted'], final, inp, kw['input_shape'], proof, enabled=True,
                   emit=events.append, on_point=lambda *a: pytest.fail('base failure manufactured point')):
        result = backend.HZSolver().evaluate_spec(final, spec, input_hz=inp, **kw)
    assert len(calls) == 1 and calls[0]['options']['presolve'] is True
    assert 0 < calls[0]['options']['time_limit'] <= 45.
    assert np.count_nonzero(calls[0]['c']) == 0
    assert result[0].status.name == 'UNKNOWN'
    assert [e['event'] for e in events] == ['c83_actual_ordinary_model_bound',
        'c83_ordinary_milp_start', 'ordinary_milp_return']


@pytest.mark.parametrize('coordinate', [-.125, .125])
def test_actual_hook_saves_point_before_exact_nonzero_extension_and_restores(coordinate):
    state, final, inp, _, kw, *_ = fixture()
    model = backend._lower_hz_milp(final)
    proof = dict(native_lowered_n_cont=model.n_cont, native_lowered_n_bin=model.n_bin)
    before = (backend._lower_hz_milp, backend.HZSolver._recover_input, backend.milp)
    points, events = [], []
    with installed(state['lifted'], final, inp, kw['input_shape'], proof, enabled=True,
                   emit=events.append, on_point=lambda m, x: points.append(x.copy())):
        actual = backend._lower_hz_milp(final, prune_unused=True, coalesce_rows=True,
            project_inactive_cont=False, fix_implied_binary=False)
        point = np.zeros(actual.n_var)
        point[np.flatnonzero(actual.cont_source == 1)[0]] = coordinate
        result = backend.HZSolver._recover_input(actual, point, inp, kw['input_shape'], 0)
        assert result is not None and len(points) == 1
        with pytest.raises(SelectedRejected):
            backend.HZSolver._recover_input(actual, point, inp, kw['input_shape'], 0)
    assert before == (backend._lower_hz_milp, backend.HZSolver._recover_input, backend.milp)
    assert events[-1]['proof']['inverse']['all_equations_exact']
    assert events[-1]['construction']['measured_transient_gate']


def test_failed_extension_aborts_without_alternate_path_and_restores_hooks():
    state, final, inp, _, kw, *_ = fixture()
    model = backend._lower_hz_milp(final)
    proof = dict(native_lowered_n_cont=model.n_cont, native_lowered_n_bin=model.n_bin)
    before = (backend._lower_hz_milp, backend.HZSolver._recover_input, backend.milp)
    events = []
    with pytest.raises(SelectedRejected):
        with installed(state['lifted'], final, inp, kw['input_shape'], proof, enabled=True, emit=events.append):
            actual = backend._lower_hz_milp(final, prune_unused=True, coalesce_rows=True,
                project_inactive_cont=False, fix_implied_binary=False)
            point = np.zeros(actual.n_var); point[0] = 2.
            backend.HZSolver._recover_input(actual, point, inp, kw['input_shape'], 0)
    assert before == (backend._lower_hz_milp, backend.HZSolver._recover_input, backend.milp)
    assert events[-1]['event'] == 'c83_exact_witness_reconstruction_rejected'


def test_default_off_and_actual_81_layer_bound():
    assert verify(*([None] * 7), pool=None) is None
    old = (backend.milp, backend._lower_hz_milp, backend.HZSolver._recover_input)
    with installed(*([None] * 5)):
        assert old == (backend.milp, backend._lower_hz_milp, backend.HZSolver._recover_input)
    assert final_extra(81) == 6976
    assert 252083361 + final_extra(81) == 252090337
