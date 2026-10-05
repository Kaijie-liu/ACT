from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831 import c9_integrated_suffix_v1 as candidate
from experiments.neural_hz_20260831.c9_integrated_suffix_audit_v1 import audit, proxy
from experiments.neural_hz_20260831.test_c9_radix_predicate_v1 import rational_definitions, rational_rows, fraction
from experiments.neural_hz_20260831.test_c7_factored_hz_v1 import rational_original, add_scaled
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect


def rational_outputs(lifted):
    radix = rational_definitions(proxy(lifted))
    hz, nc, nb = lifted.hz, lifted.old_n_cont, lifted.old_n_bin
    logical = rational_rows(hz.Ac[lifted.eq_roots], hz.Ab[lifted.eq_roots], hz.b[lifted.eq_roots], radix, lifted.logical_n_cont)
    definitions = []
    for index, (coefficients, rhs) in enumerate(logical[lifted.old_n_eq:]):
        slot = nc + index
        pivot = coefficients[slot]
        assert pivot > 0 and all(v == 0 for v in coefficients[slot + 1:lifted.logical_n_cont])
        values = [F(0)] * (nc + nb + 1)
        values[-1] = rhs / pivot
        for col, coefficient in enumerate(coefficients[:slot]):
            if col < nc:
                values[col] -= coefficient / pivot
            else:
                add_scaled(values, definitions[col - nc], -coefficient / pivot)
        for col in range(nb):
            values[nc + col] -= coefficients[lifted.logical_n_cont + col] / pivot
        assert sum(abs(v) for v in values) <= 1
        definitions.append(values)
    outputs = []
    for row in range(hz.n_out):
        values = [F(0)] * (nc + nb + 1)
        values[-1] = fraction(hz.c[row])
        start, stop = hz.Gc.indptr[row:row + 2]
        for offset in range(start, stop):
            slot = hz.Gc.indices[offset]
            assert nc <= slot < lifted.logical_n_cont
            add_scaled(values, definitions[slot - nc], fraction(hz.Gc.data[offset]))
        outputs.append(values)
    return outputs


@pytest.mark.parametrize('dyadic', [False, True])
@pytest.mark.parametrize('selection', ['all', 'partial', 'none'])
def test_two_level_fraction_elimination_matches_original_affine_program(dyadic, selection):
    expr, op = fixture(dyadic)
    keep = np.ones(expr.n_out, dtype=bool)
    if selection == 'partial':
        keep[::2] = False
    elif selection == 'none':
        keep[:] = False
    lifted = candidate.lift(expr, keep, enabled=True)
    assert rational_outputs(lifted) == rational_original(expr, lifted.old_n_cont, lifted.old_n_bin, keep)
    assert audit(lifted)['all_original_coefficients_exact']
    assert inspect(lifted.hz)['passed']


def test_main_and_old_predicate_wide_rows_use_same_radix_rule():
    s = source(8)
    s.Gc.data[0] = 1e-30
    s.c[4] = .1
    s.Ac = sp.csr_matrix([[1., 1e-40]])
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, ()),), np.zeros(8), 8, s.frame_id)
    lifted = candidate.lift(expr, np.ones(8, dtype=bool), enabled=True)
    assert lifted.report['packed_logical_rows'] >= 2 and lifted.def_rows.size > 0
    assert audit(lifted)['all_old_predicates_preserved']
    assert rational_outputs(lifted) == rational_original(expr, s.n_cont, s.n_bin, lifted.keep)
    assert inspect(lifted.hz)['passed']


@pytest.mark.parametrize('groups,stride,dilation', [(1, 1, 1), (2, 2, 1), (2, 1, 2)])
def test_grouped_masked_native_conv_geometry(groups, stride, dilation):
    kernel = (np.arange(2 * (2 // groups) * 9).reshape(2, 2 // groups, 3, 3) % 5 - 2) / 7.3
    plain = ImplicitConv2DOp(kernel, (1, 2, 4, 4), groups=groups, padding=dilation, stride=stride, dilation=dilation)
    op = ImplicitConv2DOp(kernel, plain.input_shape, groups=groups, padding=dilation, stride=stride,
        dilation=dilation, row_mask=np.arange(plain.shape[0]) % 3 != 0)
    s = source(op.shape[1])
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, (op,)),), np.zeros(op.shape[0]), op.shape[0], s.frame_id)
    lifted = candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    assert audit(lifted)['all_original_coefficients_exact']
    assert rational_outputs(lifted) == rational_original(expr, s.n_cont, s.n_bin, lifted.keep)


def test_reserved_prefix_and_duplicate_source_multiplicity():
    expr, op = fixture(False)
    term = expr.terms[0]
    expr = cnn.SparseHZAffineExpr((term, term, term), expr.bias, expr.n_out, expr.frame_id)
    lifted = candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True, frame_widths=(8, 5))
    assert lifted.old_n_cont == 8 and lifted.old_n_bin == 5 and lifted.hz.n_bin == 5
    assert audit(lifted)['exact_original_source_operator_multiset']
    assert rational_outputs(lifted) == rational_original(expr, 8, 5, lifted.keep)


def test_candidate_never_uses_original_conv_row_or_expansion(monkeypatch):
    expr, op = fixture(False)
    def forbidden(*args):
        raise AssertionError('candidate expanded original operator')
    monkeypatch.setattr(ImplicitConv2DOp, '_row', forbidden)
    monkeypatch.setattr(ImplicitConv2DOp, 'to_csr_reference', forbidden)
    assert candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True).hz.exact


@pytest.mark.parametrize('kind', ['coefficient', 'rhs', 'slot', 'scale', 'predicate', 'hidden'])
def test_all_retained_metadata_and_originals_are_sealed(kind):
    expr, op = fixture()
    lifted = candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    if kind == 'coefficient':
        lifted.hz.Ac.data[-1] += .1
    elif kind == 'rhs':
        lifted.hz.b[-1] += .1
    elif kind == 'slot':
        lifted.nodes[0]['slots'][0] += 1
    elif kind == 'scale':
        lifted.eq_scales[0] += 1
    elif kind == 'predicate':
        expr.terms[0].source.ub[0] += 1.
    else:
        lifted.report['hidden'] = np.ones(10)
    with pytest.raises((ValueError, TypeError)):
        lifted.numeric_roots()


@pytest.mark.parametrize('kind', ['coefficient', 'rhs', 'scale'])
def test_independent_audit_rejects_resealed_corruption(kind):
    expr, op = fixture()
    lifted = candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True)
    if kind == 'coefficient':
        lifted.hz.Ac.data[-1] *= 1.01
    elif kind == 'rhs':
        lifted.hz.ub[0] += 1.
    else:
        lifted.eq_scales[-1] += 1
    lifted.seal = lifted.fingerprint()
    with pytest.raises(ValueError):
        audit(lifted)


@pytest.mark.parametrize('keyword,value', [('max_work', 0), ('max_branch_work', 0), ('max_entries', 0),
    ('max_work', 256_000_001), ('frame_widths', (1, 1))])
def test_complete_caps_and_prefix_fail_closed(keyword, value):
    expr, op = fixture()
    with pytest.raises((ValueError, MemoryError)):
        candidate.lift(expr, np.ones(expr.n_out, dtype=bool), enabled=True, **{keyword: value})


def test_default_off_does_not_inspect_input(monkeypatch):
    monkeypatch.setattr(candidate, 'expression_binding', lambda *args: pytest.fail('default off inspected expression'))
    assert candidate.lift(None, None) is None
