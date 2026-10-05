from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import contract, live_value_rows


def source(n):
    c, gc, gb = np.zeros(n), np.zeros((n, 2)), np.zeros((n, 2))
    c[0], gc[n // 2, 0], gc[-1, 1], gb[1, 0], gb[-2, 1] = .5, .25, -.5, .125, .25
    return SparseHZono(c, sp.csr_matrix(gc), sp.csr_matrix(gb), sp.csr_matrix([[1., 0.]]),
        sp.csr_matrix([[-1., 0.]]), np.zeros(1), sp.csr_matrix([[0., 1.]]),
        sp.csr_matrix([[0., 1.]]), np.ones(1), frame_id=7, exact=True)


@pytest.mark.parametrize("stride,dilation,padding", [(1, 1, 1), (2, 1, 1), (1, 2, 2)])
def test_exact_dyadic_map_and_same_latent_witnesses(stride, dilation, padding):
    inner = ImplicitConv2DOp((np.arange(36).reshape(2, 2, 3, 3) % 7 - 3) / 8.,
                            (1, 2, 5, 5), stride=stride, dilation=dilation, padding=padding)
    outer = ImplicitConv2DOp((np.arange(54).reshape(3, 2, 3, 3) % 5 - 2) / 8., inner.output_shape, padding=1)
    s = source(inner.shape[1])
    scale = np.resize(np.array([.5, -.25]), inner.shape[0])
    rows = np.array([0, outer.shape[0] // 2, outer.shape[0] - 1])
    compiled = contract(s, inner, scale, outer, rows)
    reference = (outer.to_csr_reference() @ sp.diags(scale) @ inner.to_csr_reference())[rows]
    bias = np.array([.25, -.5, .125])
    actual = compiled.apply(bias)
    assert np.array_equal(actual.c, reference @ s.c + bias)
    assert np.array_equal(actual.Gc.toarray(), (reference @ s.Gc).toarray())
    assert np.array_equal(actual.Gb.toarray(), (reference @ s.Gb).toarray())
    assert actual.frame_id == s.frame_id and actual.n_cont == s.n_cont and actual.n_bin == s.n_bin
    for name in ("Ac", "Ab", "Auc", "Aub"):
        assert np.array_equal(getattr(actual, name).toarray(), getattr(s, name).toarray())
    for z in product((-1., 1.), repeat=2):
        for xi1 in (-1., 0., 1.):
            xi = np.array([z[0], xi1])
            z = np.asarray(z)
            if (s.Auc @ xi + s.Aub @ z > s.ub).any():
                continue
            before = reference @ (s.c + s.Gc @ xi + s.Gb @ z) + bias
            after = actual.c + actual.Gc @ xi + actual.Gb @ z
            assert np.array_equal(before, after)
    assert compiled.stats["skipped_channel_products"] > 0


def test_zero_value_source_keeps_nontrivial_binary_predicates_and_bias():
    op = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 2, 2))
    s = source(8)
    s.c[:] = 0
    s.Gc.data[:] = 0
    s.Gb.data[:] = 0
    compiled = contract(s, op, np.ones(8), op, [0, 7])
    out = compiled.apply([1., -1.])
    assert compiled.stats["actual_channel_products"] == 0 and compiled.matrix.nnz == 0
    assert out.n_cont == 2 and out.n_bin == 2 and out.n_eq == 1 and out.n_ineq == 1
    assert np.array_equal(out.c, [1., -1.]) and compiled.source is s


def test_center_and_binary_only_rows_are_live():
    live = live_value_rows(source(20))
    assert live[0] and live[1] and live[18] and live[10] and live[19]
    assert live.sum() == 5


@pytest.mark.parametrize("mutation", ["center", "binary", "predicate", "frame"])
def test_source_mutation_rejected_before_application(mutation):
    op = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 2, 2))
    s = source(8)
    compiled = contract(s, op, np.ones(8), op, [0, 7])
    if mutation == "center":
        s.c[0] += 1
    elif mutation == "binary":
        s.Gb.data[0] += 1
    elif mutation == "predicate":
        s.ub[0] += 1
    else:
        s.frame_id += 1
    with pytest.raises(ValueError, match="source changed"):
        compiled.apply()


@pytest.mark.parametrize("mode", ["products", "nnz", "inexact", "rows", "groups"])
def test_invalid_or_overbudget_core_rejects(mode):
    op = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 2, 2))
    s = source(8)
    kwargs, rows = {}, [0, 7]
    if mode == "products":
        kwargs["max_products"] = 0
    elif mode == "nnz":
        kwargs["max_nnz"] = 0
    elif mode == "inexact":
        s.exact = False
    elif mode == "rows":
        rows = [0, 0]
    else:
        op = ImplicitConv2DOp(np.ones((2, 1, 1, 1)), (1, 2, 2, 2), groups=2)
    with pytest.raises((ValueError, MemoryError)):
        contract(s, op, np.ones(8), op, rows, **kwargs)
