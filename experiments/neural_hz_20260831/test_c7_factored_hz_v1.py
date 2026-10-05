import copy
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831 import c7_factored_hz_v1 as candidate
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


def fraction(value):
    return F.from_float(float(value))


def add_scaled(target, values, scale):
    for i, value in enumerate(values):
        if value:
            target[i] += scale * value


def rational_definitions(lifted):
    hz, nc, nb = lifted.hz, lifted.old_n_cont, lifted.old_n_bin
    width, definitions = nc + nb + 1, []
    for index in range(hz.n_cont - nc):
        row = lifted.old_n_eq + index
        expected_column = nc + index
        assert hz.Ac[row, expected_column] == 1.
        assert hz.Ac[row, expected_column + 1:].nnz == 0
        values = [F(0)] * width
        values[-1] = fraction(hz.b[row])
        start, stop = hz.Ac.indptr[row:row + 2]
        for offset in range(start, stop):
            col, coef = hz.Ac.indices[offset], fraction(hz.Ac.data[offset])
            if col < nc:
                values[col] -= coef
            elif col != expected_column:
                add_scaled(values, definitions[col - nc], -coef)
        start, stop = hz.Ab.indptr[row:row + 2]
        for offset in range(start, stop):
            values[nc + hz.Ab.indices[offset]] -= fraction(hz.Ab.data[offset])
        definitions.append(values)
    outputs = []
    for row in range(hz.n_out):
        values = [F(0)] * width
        values[-1] = fraction(hz.c[row])
        start, stop = hz.Gc.indptr[row:row + 2]
        for offset in range(start, stop):
            col, coef = hz.Gc.indices[offset], fraction(hz.Gc.data[offset])
            if col < nc:
                values[col] += coef
            else:
                add_scaled(values, definitions[col - nc], coef)
        start, stop = hz.Gb.indptr[row:row + 2]
        for offset in range(start, stop):
            values[nc + hz.Gb.indices[offset]] += fraction(hz.Gb.data[offset])
        outputs.append(values)
    return definitions, outputs


def rational_original(expr, nc, nb, keep):
    width = nc + nb + 1
    result = [[F(0)] * width for _ in range(expr.n_out)]
    for row in range(expr.n_out):
        result[row][-1] = fraction(expr.bias[row])
    for term in expr.terms:
        s = term.source
        values = [[F(0)] * width for _ in range(s.n_out)]
        for row in range(s.n_out):
            values[row][-1] = fraction(s.c[row])
            for matrix, shift in ((s.Gc, 0), (s.Gb, nc)):
                start, stop = matrix.indptr[row:row + 2]
                for offset in range(start, stop):
                    values[row][shift + matrix.indices[offset]] += fraction(matrix.data[offset])
        for op in term.operators:
            matrix = op.to_csr_reference() if type(op) is ImplicitConv2DOp else op
            following = [[F(0)] * width for _ in range(matrix.shape[0])]
            for row in range(matrix.shape[0]):
                start, stop = matrix.indptr[row:row + 2]
                for offset in range(start, stop):
                    add_scaled(following[row], values[matrix.indices[offset]], fraction(matrix.data[offset]))
            values = following
        for row in np.flatnonzero(keep):
            add_scaled(result[row], values[row], F(1))
    return result


@pytest.mark.parametrize('dyadic', [False, True])
@pytest.mark.parametrize('selection', ['all', 'partial', 'none'])
def test_complete_rational_elimination_and_unchanged_old_predicates(dyadic, selection):
    expr, op = fixture(dyadic)
    keep = np.ones(expr.n_out, dtype=bool)
    if selection == 'partial':
        keep[::2] = False
    elif selection == 'none':
        keep[:] = False
    lifted = candidate.lift(expr, keep, enabled=True)
    definitions, actual = rational_definitions(lifted)
    assert actual == rational_original(expr, lifted.old_n_cont, lifted.old_n_bin, keep)
    old, hz = zero_transfer_reference(expr), lifted.hz
    for name in ('Ac', 'Ab'):
        expected = getattr(old, name)
        actual = getattr(hz, name)[:old.n_eq, :expected.shape[1]]
        assert equal_payload(actual, expected)
    for name in ('Auc', 'Aub'):
        expected = getattr(old, name)
        assert equal_payload(getattr(hz, name)[:, :expected.shape[1]], expected)
    assert np.array_equal(hz.b[:old.n_eq], old.b) and equal_payload(hz.ub, old.ub)
    assert hz.n_bin == old.n_bin and hz.frame_id == old.frame_id and hz.exact
    assert hz.Ac[:old.n_eq, old.n_cont:].nnz == 0
    assert hz.Auc[:, old.n_cont:].nnz == 0
    assert lifted.report['source_count'] == 3
    assert sum(n['kind'] == 'op' and n['op'] is expr.terms[0].operators[-1] for n in lifted.nodes) == 1


def test_forward_latent_extension_is_within_boxes_and_projects_to_old_coordinates():
    expr, op = fixture(False)
    lifted = candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    definitions, outputs = rational_definitions(lifted)
    checked = 0
    for z in product((-1, 1), repeat=2):
        for free in (-1, 0, 1):
            xi = (z[0], free)
            if any((t.source.Auc @ np.array(xi) + t.source.Aub @ np.array(z) > t.source.ub).any() for t in expr.terms):
                continue
            assignment = list(map(F, (*xi, *z, 1)))
            extension = [sum(v * x for v, x in zip(row, assignment)) for row in definitions]
            assert all(abs(value) <= 1 for value in extension)
            combined = list(map(F, xi)) + extension
            assert combined[:lifted.old_n_cont] == list(map(F, xi))
            for row_id in range(lifted.old_n_eq, lifted.hz.n_eq):
                ac = lifted.hz.Ac.getrow(row_id)
                ab = lifted.hz.Ab.getrow(row_id)
                value = sum(fraction(a) * combined[i] for a, i in zip(ac.data, ac.indices))
                value += sum(fraction(a) * z[i] for a, i in zip(ab.data, ab.indices))
                assert value == fraction(lifted.hz.b[row_id])
            checked += 1
    assert checked > 0


@pytest.mark.parametrize('groups,stride,dilation', [(1, 1, 1), (2, 2, 1), (2, 1, 2)])
def test_grouped_masked_geometry_retains_exact_coefficients(groups, stride, dilation):
    rng = np.random.default_rng(63)
    kernel = rng.integers(-2, 3, size=(2, 2 // groups, 3, 3)) / 7.3
    plain = ImplicitConv2DOp(kernel, (1, 2, 4, 4), groups=groups, padding=dilation, stride=stride, dilation=dilation)
    op = ImplicitConv2DOp(kernel, plain.input_shape, groups=groups, padding=dilation, stride=stride,
        dilation=dilation, row_mask=rng.random(plain.shape[0]) > .3)
    s = source(op.shape[1])
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, (op,)),), np.zeros(op.shape[0]), op.shape[0], s.frame_id)
    keep = np.ones(expr.n_out, dtype=bool)
    lifted = candidate.lift(expr, keep, enabled=True)
    assert rational_definitions(lifted)[1] == rational_original(expr, lifted.old_n_cont, lifted.old_n_bin, keep)


def test_duplicate_terms_keep_multiplicity_not_independent_copies():
    expr, op = fixture(False)
    term = expr.terms[0]
    expr = cnn.SparseHZAffineExpr((term, term, term), expr.bias, expr.n_out, expr.frame_id)
    keep = np.ones(expr.n_out, dtype=bool)
    lifted = candidate.lift(expr, keep, enabled=True)
    assert sum(n['kind'] == 'source' for n in lifted.nodes) == 1
    assert rational_definitions(lifted)[1] == rational_original(expr, lifted.old_n_cont, lifted.old_n_bin, keep)


def test_zero_source_keeps_infeasible_predicate_and_binary_columns():
    s = source(8)
    s.c[:] = 0.
    s.Gc.data[:] = 0.
    s.Gb.data[:] = 0.
    s.Ac.data[:] = 0.
    s.Ab.data[:] = 0.
    s.b[:] = 1.
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, ()),), np.arange(8, dtype=float), 8, s.frame_id)
    lifted = candidate.lift(expr, np.ones(8, dtype=bool), enabled=True)
    assert lifted.hz.n_cont == 2 and lifted.hz.n_bin == 2
    assert np.all(lifted.hz.Ac.data == 0.) and np.all(lifted.hz.Ab.data == 0.)
    assert np.array_equal(lifted.hz.b, [1.]) and np.array_equal(lifted.hz.c, expr.bias)


def test_reserved_global_frame_prefix_is_not_reused():
    expr, op = fixture()
    keep = np.ones(expr.n_out, dtype=bool)
    lifted = candidate.lift(expr, keep, enabled=True, frame_widths=(8, 5))
    assert lifted.old_n_cont == 8 and lifted.hz.n_bin == 5
    assert all(n['slots'][n['needed']].min(initial=8) >= 8 for n in lifted.nodes)
    assert rational_definitions(lifted)[1] == rational_original(expr, 8, 5, keep)


@pytest.mark.parametrize('mutation', ['predicate', 'zero_source', 'kernel', 'bias', 'new_equation', 'slot', 'exponent', 'extra_field'])
def test_bound_source_and_complete_new_numeric_state_mutations_reject(mutation):
    expr, op = fixture()
    lifted = candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    if mutation == 'predicate':
        expr.terms[0].source.ub[0] += 1.
    elif mutation == 'zero_source':
        expr.terms[1].source.c[0] += 1.
    elif mutation == 'kernel':
        op._kernel.flat[0] += 1.
    elif mutation == 'bias':
        expr.bias[0] += 1.
    elif mutation == 'new_equation':
        lifted.hz.Ac.data[-1] += 1.
    elif mutation == 'slot':
        lifted.nodes[0]['slots'][0] += 1
    elif mutation == 'exponent':
        lifted.nodes[0]['exponents'][0] += 1
    else:
        lifted.nodes[0]['unregistered_payload'] = np.ones(2)
    with pytest.raises(ValueError):
        lifted.numeric_roots()


@pytest.mark.parametrize('option', ['work', 'branch', 'entries', 'cap_raise', 'frame', 'nonfinite'])
def test_unproved_or_overbudget_lift_fails_closed(option):
    expr, op = fixture()
    kwargs = {}
    if option == 'work':
        kwargs['max_work'] = 0
    elif option == 'branch':
        kwargs['max_branch_work'] = 0
    elif option == 'entries':
        kwargs['max_entries'] = 0
    elif option == 'cap_raise':
        kwargs['max_work'] = 256_000_001
    elif option == 'frame':
        kwargs['frame_widths'] = (1, 1)
    else:
        op._kernel.flat[0] = np.inf
    with pytest.raises((ValueError, MemoryError)):
        candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True, **kwargs)


def test_exact_scaling_rejects_lost_nonzero_bits():
    with pytest.raises(ValueError):
        candidate.scaled_exact([np.nextafter(0., 1.)], -1)
    values = np.array([.1, -.375, .7])
    assert np.array_equal(candidate.scaled_exact(candidate.scaled_exact(values, -8), 8), values)


def test_default_off_does_not_touch_the_expression(monkeypatch):
    def forbidden(*args):
        raise AssertionError('default-off touched candidate graph')
    monkeypatch.setattr(candidate, 'expression_binding', forbidden)
    assert candidate.lift(None, None) is None
