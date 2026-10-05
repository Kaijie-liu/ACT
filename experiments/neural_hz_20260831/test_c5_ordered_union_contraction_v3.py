import copy

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr, SparseHZAffineTerm
from act.back_end.solver.solver_hz import sparse_hz_linear
from experiments.neural_hz_20260831.c5_ordered_union_contraction_v3 import contract, materialize
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import (
    reference_matrix, reference_materialize, equal_payload, compare_hz,
)
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import live_value_rows
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


def fixture(divisor=7.3, stride=1, dilation=1, padding=1, batch=1):
    inner = ImplicitConv2DOp((np.arange(36).reshape(2, 2, 3, 3) % 7 - 3) / divisor,
                            (batch, 2, 5, 5), stride=stride, dilation=dilation, padding=padding)
    outer = ImplicitConv2DOp((np.arange(54).reshape(3, 2, 3, 3) % 5 - 2) / divisor,
                            inner.output_shape, padding=1)
    s = source(inner.shape[1])
    # Nontrivial live support in both batches as well as c-only and binary-only rows.
    s.c[::7] = .123
    middle = np.broadcast_to(np.array([.123, -.731])[None, :, None, None], inner.output_shape).copy().reshape(-1)
    output = np.broadcast_to(np.array([-.71, .019, 1.37])[None, :, None, None], outer.output_shape).copy().reshape(-1)
    return s, inner, middle, outer, np.arange(outer.shape[0])[::-1], output


@pytest.mark.parametrize("divisor", [8., 7.3])
@pytest.mark.parametrize("geometry", [(1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 2, 1), (1, 1, 1, 2)])
def test_all_retained_coefficients_and_hz_fields_bitwise_original(divisor, geometry):
    s, a, d, b, rows, e = fixture(divisor, *geometry)
    compiled = contract(s, a, d, b, rows, e)
    full = reference_matrix(s, a, d, b, rows, e, restricted=False)
    reduced = reference_matrix(s, a, d, b, rows, e)
    assert equal_payload(compiled.matrix, reduced)
    live = live_value_rows(s)
    assert equal_payload(compiled.matrix[:, live], full[:, live])
    assert all(compare_hz(compiled.apply(), sparse_hz_linear(s, full)).values())
    assert compiled.stats["actual_channel_products"] <= compiled.stats["channel_product_upper_bound"]


@pytest.mark.parametrize("repeated_source", [False, True])
def test_complete_join_retains_predicates_frame_and_full_unmasked_bias(repeated_source):
    s, a, d, b, rows, e = fixture()
    other = s if repeated_source else copy.deepcopy(s)
    if not repeated_source:
        other.c *= -.73
        other.b[0] = .125
    operators = (a, sp.diags(d, format="csr"), b, sp.diags(e, format="csr"))
    expr = SparseHZAffineExpr((SparseHZAffineTerm(s, operators), SparseHZAffineTerm(other, operators)),
                             np.arange(b.shape[0]) / 9.7, b.shape[0], s.frame_id)
    selected = rows[::3]
    actual, stats = materialize(expr, selected)
    full = reference_materialize(expr, selected, restricted=False)
    reduced = reference_materialize(expr, selected)
    assert all(compare_hz(actual, full).values())
    assert all(compare_hz(actual, reduced).values())
    outside = np.ones(expr.n_out, dtype=bool)
    outside[selected] = False
    assert equal_payload(actual.c[outside], expr.bias[outside])
    assert actual.Gc[outside].nnz == actual.Gb[outside].nnz == 0
    assert len(stats) == 2 and actual.n_bin == 2


def test_stationary_inner_mask_and_nonstationary_output_mask():
    s, a, d, b, rows, e = fixture()
    mask = np.ones(a.output_shape, dtype=bool)
    mask[:, 1] = False
    a = ImplicitConv2DOp(a._kernel, a.input_shape, padding=1, row_mask=mask.reshape(-1))
    outmask = np.arange(b.shape[0]) % 3 == 0
    b = ImplicitConv2DOp(b._kernel, b.input_shape, padding=1, row_mask=outmask)
    assert equal_payload(contract(s, a, d, b, rows, e).matrix, reference_matrix(s, a, d, b, rows, e))


def test_exact_cancellation_zero_source_and_zero_diagonals():
    s = source(8)
    a = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 2, 2))
    b = ImplicitConv2DOp(np.array([1., -1., -0., 1.]).reshape(2, 2, 1, 1), a.output_shape)
    for zero_source in (False, True):
        if zero_source:
            s.c[:] = 0.
            s.Gc.data[:] = 0.
            s.Gb.data[:] = 0.
        out = contract(s, a, np.ones(8), b, np.arange(8), np.ones(8))
        assert equal_payload(out.matrix, reference_matrix(s, a, np.ones(8), b, np.arange(8), np.ones(8)))
        assert out.apply().n_eq == 1 and out.apply().n_ineq == 1 and out.apply().n_bin == 2
    assert out.stats["actual_channel_products"] == 0
    assert contract(source(8), a, np.zeros(8), b, [1, 7], np.ones(8)).matrix.nnz == 0


def test_repeated_spatial_demands_share_final_coefficients():
    a = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 8, 8))
    s = source(a.shape[1])
    s.c[:] = 1.
    result = contract(s, a, np.ones(a.shape[0]), a, np.arange(a.shape[0]), np.ones(a.shape[0]))
    assert result.stats["channel_product_upper_bound"] == 8
    assert result.stats["unrestricted_spatial_channel_products"] == 512
    assert result.stats["quarter_product_gate"]


@pytest.mark.parametrize("mode", ["middle", "output", "mask", "products", "nnz", "rows", "nonfinite"])
def test_fail_closed(mode):
    s, a, d, b, rows, e = fixture()
    kwargs = {}
    if mode == "middle":
        d[1] = np.nextafter(d[1], 1.)
    elif mode == "output":
        e[1] = np.nextafter(e[1], 1.)
    elif mode == "mask":
        mask = np.ones(a.shape[0], dtype=bool)
        mask[1] = False
        a = ImplicitConv2DOp(a._kernel, a.input_shape, padding=1, row_mask=mask)
    elif mode == "products":
        kwargs["max_products"] = 0
    elif mode == "nnz":
        kwargs["max_nnz"] = 0
    elif mode == "rows":
        rows = np.array([0, 0])
    else:
        e[:] = np.inf
    with pytest.raises((ValueError, MemoryError)):
        contract(s, a, d, b, rows, e, **kwargs)


def test_bound_source_mutation_rejected():
    s, a, d, b, rows, e = fixture()
    result = contract(s, a, d, b, rows, e)
    s.ub[0] += .01
    with pytest.raises(ValueError, match="source changed"):
        result.apply()
