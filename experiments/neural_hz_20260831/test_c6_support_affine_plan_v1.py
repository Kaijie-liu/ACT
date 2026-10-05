import copy
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import SupportEngine, plan, scalar_test_materialize
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import compare_hz
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


@pytest.mark.parametrize('groups,stride,dilation,padding,batch', [
    (1, 1, 1, 1, 1), (2, 1, 1, 1, 2), (1, 2, 1, 1, 1), (2, 1, 2, 2, 1)])
@pytest.mark.parametrize('transpose', [False, True])
def test_integer_conv_support_equals_complete_expanded_geometry(groups, stride, dilation, padding, batch, transpose):
    rng = np.random.default_rng(330)
    op = ImplicitConv2DOp(rng.integers(-2, 3, size=(4, 4 // groups, 3, 3)) / 8.,
        (batch, 4, 5, 5), stride=stride, dilation=dilation, padding=padding, groups=groups)
    op = ImplicitConv2DOp(op._kernel, op.input_shape, stride=stride, dilation=dilation,
        padding=padding, groups=groups, row_mask=rng.random(op.shape[0]) > .3)
    mask = rng.random(op.shape[0 if transpose else 1]) > .6
    actual = SupportEngine().compute(op, mask, transpose=transpose)
    matrix = op.to_csr_reference()
    matrix.data = (matrix.data != 0.).astype(np.int64)
    expected = (matrix.T if transpose else matrix) @ mask.astype(np.int64)
    assert np.array_equal(actual, expected)


def fixture(dyadic=True):
    rng = np.random.default_rng(42)
    kernel = rng.integers(-3, 4, size=(2, 2, 3, 3)).astype(float)
    kernel = kernel / 8. if dyadic else kernel / 7.3
    op = ImplicitConv2DOp(kernel, (1, 2, 3, 3), padding=1)
    s = source(op.shape[1])
    equal_but_distinct = copy.deepcopy(s)
    zero = copy.deepcopy(s)
    zero.c[:] = 0.
    zero.Gc.data[:] = 0.
    zero.Gb.data[:] = 0.
    zero.ub[:] = .375  # Predicate payload must survive zero-map pruning.
    d = sp.diags(np.where(np.arange(op.shape[0]) % 3, -.25, 0.), format='csr')
    dense = sp.csr_matrix(rng.integers(-3, 4, size=(6, op.shape[0])) / (8. if dyadic else 11.7))
    terms = [cnn.SparseHZAffineTerm(s, (op, d, op, d, dense)),
             cnn.SparseHZAffineTerm(zero, (op, d, op, d, dense)),
             cnn.SparseHZAffineTerm(s, (op, d, dense)),
             cnn.SparseHZAffineTerm(equal_but_distinct, (op, d, dense))]
    return cnn.SparseHZAffineExpr(tuple(terms), np.arange(6, dtype=float) / 8., 6, s.frame_id), op


@pytest.mark.parametrize('dyadic', [False, True])
@pytest.mark.parametrize('mode', ['all', 'partial', 'none'])
def test_complete_mixed_program_byte_matches_native_including_every_predicate(dyadic, mode):
    expr, op = fixture(dyadic)
    keep = np.ones(expr.n_out, dtype=bool)
    if mode == 'partial':
        keep[::2] = False
    elif mode == 'none':
        keep[:] = False
    bound = plan(expr, keep)
    actual = scalar_test_materialize(bound, keep)
    native = cnn._lazy_materialize(expr, keep, 64_000_000)
    assert all(compare_hz(actual, native).values()), compare_hz(actual, native)
    assert bound.report['unique_source_count'] == 3
    assert bound.report['terms'][1]['product_upper_bound'] == 0
    assert actual.n_bin == 2 and actual.n_eq == native.n_eq and actual.n_ineq == native.n_ineq


def test_support_planning_never_expands_conv_or_requests_its_scalar_rows(monkeypatch):
    expr, op = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('planner expanded a Conv row')
    monkeypatch.setattr(ImplicitConv2DOp, '_row', forbidden)
    monkeypatch.setattr(ImplicitConv2DOp, 'to_csr_reference', forbidden)
    bound = plan(expr, np.ones(expr.n_out, dtype=bool))
    assert bound.report['support_integer_visits'] > 0
    assert bound.report['support_cache_hits'] > 0


def test_zero_source_realization_requests_no_conv_rows_but_preserves_bias_and_predicates(monkeypatch):
    expr, op = fixture()
    zero = expr.terms[1]
    expr = cnn.SparseHZAffineExpr((zero,), expr.bias, expr.n_out, expr.frame_id)
    keep = np.ones(expr.n_out, dtype=bool)
    bound = plan(expr, keep)
    def forbidden(*args, **kwargs):
        raise AssertionError('zero-value source requested a Conv row')
    monkeypatch.setattr(ImplicitConv2DOp, '_row', forbidden)
    out = scalar_test_materialize(bound, keep)
    assert out.Gc.nnz == out.Gb.nnz == 0 and out.n_bin == zero.source.n_bin
    assert np.array_equal(out.c, expr.bias) and np.array_equal(out.ub, zero.source.ub)


@pytest.mark.parametrize('mutation', ['source', 'predicate', 'frame', 'kernel', 'diagonal', 'bias', 'support'])
def test_payload_changes_invalidate_source_bound_program(mutation):
    expr, op = fixture()
    keep = np.ones(expr.n_out, dtype=bool)
    bound = plan(expr, keep)
    if mutation == 'source':
        expr.terms[0].source.c[2] += 1.
    elif mutation == 'predicate':
        expr.terms[1].source.ub[0] += 1.
    elif mutation == 'frame':
        expr.terms[0].source.frame_id += 1
    elif mutation == 'kernel':
        op._kernel.flat[0] += .1
    elif mutation == 'diagonal':
        expr.terms[0].operators[1].data[0] += .1
    elif mutation == 'bias':
        expr.bias[0] += .1
    else:
        bound.supports[0][0].flags.writeable = True
        bound.supports[0][0][0] = not bound.supports[0][0][0]
    with pytest.raises(ValueError, match='changed'):
        scalar_test_materialize(bound, keep)


def test_center_and_binary_only_rows_are_not_discarded():
    s = source(18)
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, ()),), np.zeros(18), 18, s.frame_id)
    bound = plan(expr, np.ones(18, dtype=bool))
    assert bound.supports[0][0][0] and bound.supports[0][0][1] and bound.supports[0][0][-2]
    assert all(compare_hz(scalar_test_materialize(bound, np.ones(18, dtype=bool)), s).values())


def test_support_ignores_numerical_cancellation_but_realization_preserves_it():
    s = source(8)
    first = sp.csr_matrix(np.vstack([np.eye(8), np.eye(8)]))
    last = sp.csr_matrix(np.hstack([np.eye(8), -np.eye(8)]))
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, (first, last)),), np.zeros(8), 8, s.frame_id)
    keep = np.ones(8, dtype=bool)
    bound = plan(expr, keep)
    assert bound.supports[0][-1].any()
    actual = scalar_test_materialize(bound, keep)
    assert actual.Gc.nnz == actual.Gb.nnz == 0 and not actual.c.any()
    assert all(compare_hz(actual, cnn._lazy_materialize(expr, keep, 64_000_000)).values())


def test_integer_multiplicity_transport_and_exact_row_set_work_bound():
    expr, op = fixture()
    multiplicities = np.arange(op.shape[0], dtype=np.int64) % 5
    matrix = op.to_csr_reference()
    matrix.data = (matrix.data != 0.).astype(np.int64)
    assert np.array_equal(SupportEngine().compute(op, multiplicities, transpose=True), matrix.T @ multiplicities)
    keep = np.arange(expr.n_out) % 2 == 0
    bound = plan(expr, keep)
    for term, masks, record in zip(expr.terms, bound.supports, bound.report['terms'], strict=True):
        members = [{i} if keep[i] and masks[-1][i] else set() for i in range(expr.n_out)]
        for position in range(len(term.operators) - 1, -1, -1):
            operator = term.operators[position]
            expanded = operator.to_csr_reference() if type(operator) is ImplicitConv2DOp else operator
            new_members, actual_work = [set() for _ in range(expanded.shape[1])], 0
            for row in range(expanded.shape[0]):
                for offset in range(expanded.indptr[row], expanded.indptr[row + 1]):
                    col = expanded.indices[offset]
                    if expanded.data[offset] == 0. or not masks[position][col]:
                        continue
                    actual_work += len(members[row])
                    new_members[col].update(members[row])
            assert actual_work <= record['stages'][position]['product_upper_bound']
            members = new_members


def test_dyadic_latent_witnesses_match_independent_forward_program():
    expr, op = fixture(dyadic=True)
    keep = np.ones(expr.n_out, dtype=bool)
    actual = scalar_test_materialize(plan(expr, keep), keep)
    checked = 0
    for binary in product((-1., 1.), repeat=2):
        z = np.asarray(binary)
        for free in (-1., 0., 1.):
            xi = np.array([z[0], free])
            if any((t.source.Auc @ xi + t.source.Aub @ z > t.source.ub).any() for t in expr.terms):
                continue
            expected = expr.bias.copy()
            for term in expr.terms:
                value = term.source.c + term.source.Gc @ xi + term.source.Gb @ z
                for operator in term.operators:
                    value = operator.matvec(value) if type(operator) is ImplicitConv2DOp else operator @ value
                expected += value
            assert np.array_equal(expected, actual.c + actual.Gc @ xi + actual.Gb @ z)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize('mode', ['cap', 'rows', 'nonfinite', 'zero_nonfinite'])
def test_invalid_or_overbudget_program_fails_closed(mode):
    expr, op = fixture()
    keep, kwargs = np.ones(expr.n_out, dtype=bool), {}
    if mode == 'cap':
        kwargs['max_visits'] = 0
    elif mode == 'rows':
        keep = keep.astype(int)
    elif mode == 'nonfinite':
        op._kernel.flat[0] = np.inf
    else:
        expr = cnn.SparseHZAffineExpr((expr.terms[1],), expr.bias, expr.n_out, expr.frame_id)
        op._kernel.flat[0] = np.inf
    with pytest.raises((MemoryError, ValueError)):
        plan(expr, keep, **kwargs)
