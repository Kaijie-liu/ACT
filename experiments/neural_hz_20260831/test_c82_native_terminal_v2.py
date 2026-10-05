"""Ordinary original-affine and nonzero local-journal witness interface checks."""
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import pytest
import torch
from act.back_end.solver.solver_hz import sparse_hz_linear, _lower_hz_milp, HZSolver
from act.front_end.specs import OutputSpec, OutKind
from experiments.neural_hz_20260831.c82_native_terminal_v2 import affine, recover
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c74_native_binding_v1 import native


def pool():
    return WorkPool(256_000_000)


def fixture(ids=(78, 79, 80)):
    state, _, fresh = native()
    inp = fresh['fields']['expression'].terms[0].source
    width = state.hz.n_out
    weight = (np.arange(3 * width).reshape(3, width) % 7 - 3).astype(np.float64) / 8
    bias = np.array([.125, -.25, .5])
    spec = OutputSpec(OutKind.UNSAFE_LINEAR, c=torch.tensor([[1., -1., 0.]], dtype=torch.float64),
                      d=torch.tensor([-.125], dtype=torch.float64))
    params = spec.encode_linear(B=1, n_out=3, device=torch.device('cpu'), dtype=torch.float64)
    post = SimpleNamespace(id=ids[0], kind='RELU')
    dense = SimpleNamespace(id=ids[1], kind='DENSE', params=dict(weight=weight, bias=bias))
    assertion = SimpleNamespace(id=ids[2], kind='ASSERT', params=params)
    net = SimpleNamespace(layers=[post, dense, assertion],
        preds={ids[1]: [ids[0]], ids[2]: [ids[1]]},
        succs={ids[0]: [ids[1]], ids[1]: [ids[2]], ids[2]: []})
    output = sparse_hz_linear(state.hz, sp.csr_matrix(weight), bias)
    return state, net, ids[0], output, inp, (1, inp.n_out), spec


@pytest.mark.parametrize('ids', [(78, 79, 80), (4, 8, 19), (110, 120, 141)])
def test_complete_exact_native_affine_and_original_property_no_identity_menu(ids):
    report = affine(*fixture(ids), pool=pool(), enabled=True)
    assert report['all_final_affine_bits_match_original_binary64'] and report['all_predicates_shared_by_identity']
    assert not report['terminal_solve_executed'] and not report['concrete_witness']


@pytest.mark.parametrize('bad', ['center', 'generator', 'weight', 'predicate', 'RHS', 'property', 'topology', 'shape', 'budget'])
def test_final_binding_rejects_changed_equation_problem_or_frame(bad):
    state, net, lid, output, inp, shape, spec = fixture()
    budget = pool()
    if bad == 'center':
        output.c[0] += .125
    elif bad == 'generator':
        output.Gc.data[0] += .125
    elif bad == 'weight':
        net.layers[1].params['weight'][0, 0] += .125
    elif bad == 'predicate':
        output.Ac = output.Ac.copy()
    elif bad == 'RHS':
        output.b = output.b.copy()
    elif bad == 'property':
        spec.kind = OutKind.LINEAR_LE
    elif bad == 'topology':
        net.preds[net.layers[1].id] = []
    elif bad == 'shape':
        shape = (inp.n_out, 1)
    else:
        budget = WorkPool(0)
    with pytest.raises((ValueError, MemoryError)):
        affine(state, net, lid, output, inp, shape, spec, pool=budget, enabled=True)


@pytest.mark.parametrize('coordinate', [-.125, 0., .125])
def test_nonzero_original_input_recovery_and_complete_unit_local_extension(coordinate):
    state, _, _, output, inp, shape, _ = fixture()
    model = _lower_hz_milp(output, project_inactive_cont=False, fix_implied_binary=False)
    x = np.zeros(model.n_var)
    x[np.flatnonzero(model.cont_source == 1)[0]] = coordinate
    actual, report = recover(state, model, x, inp, shape, 0, HZSolver._recover_input,
                             pool=pool(), enabled=True)
    assert torch.equal(actual, HZSolver._recover_input(model, x, inp, shape, 0))
    assert report['inverse']['unit_equations'] > 0 and report['inverse']['local_equations'] > 0
    assert report['inverse']['all_equations_exact']
    assert report['concrete_network_validation_still_required']


@pytest.mark.parametrize('bad', ['box', 'binary', 'nonfinite', 'map', 'frame', 'native', 'budget'])
def test_witness_failure_never_returns_a_promotable_result(bad):
    state, _, _, output, inp, shape, _ = fixture()
    model = _lower_hz_milp(output, project_inactive_cont=False, fix_implied_binary=False)
    x = np.zeros(model.n_var)
    budget, old = pool(), HZSolver._recover_input
    if bad == 'box':
        x[0] = 2.
    elif bad == 'binary':
        x[model.n_cont] = .5
    elif bad == 'nonfinite':
        x[0] = np.nan
    elif bad == 'map':
        model.cont_source[0] = -1
    elif bad == 'frame':
        inp.frame_id += 1
    elif bad == 'native':
        old = lambda *args: None
    else:
        budget = WorkPool(0)
    with pytest.raises((ValueError, MemoryError)):
        recover(state, model, x, inp, shape, 0, old, pool=budget, enabled=True)


def test_default_off():
    assert affine(*([None] * 7), pool=None) is None
    assert recover(*([None] * 7), pool=None) is None


def test_ordinary_nonexact_real_sum_preserves_original_binary64_semantics():
    from fractions import Fraction as F
    from experiments.neural_hz_20260831.c82_native_terminal_v1 import affine as unrounded
    state, net, lid, _, inp, shape, spec = fixture()
    weight = net.layers[1].params['weight']
    weight[:] = np.linspace(.11, 1.73, weight.size).reshape(weight.shape)
    output = sparse_hz_linear(state.hz, sp.csr_matrix(weight), net.layers[1].params['bias'])
    exact = [sum((F(float(w)) * F(float(c)) for w, c in zip(row, state.hz.c)),
                 F(float(b))) for row, b in zip(weight, net.layers[1].params['bias'])]
    assert any(x != F(float(y)) for x, y in zip(exact, output.c))
    with pytest.raises(ValueError, match='center'):
        unrounded(state, net, lid, output, inp, shape, spec, pool=pool(), enabled=True)
    report = affine(state, net, lid, output, inp, shape, spec, pool=pool(), enabled=True)
    assert report['all_final_affine_bits_match_original_binary64']
    assert not report['unrounded_real_affine_exact_claim']
