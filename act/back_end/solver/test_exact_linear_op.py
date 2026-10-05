import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

sp = pytest.importorskip("scipy.sparse")

from act.back_end.hybridz_tf.exact_linear_op import (
    CSRLinearOp,
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
import act.back_end.hybridz_tf.tf_cnn as tf_cnn_module
from act.back_end.hybridz_tf.tf_cnn import (
    SparseHZAffineExpr,
    SparseHZAffineTerm,
    SparseHZPhaseSelectiveResult,
    _lazy_add_const,
    _lazy_append_linear,
    _lazy_conv2d_operator_and_bias,
    _lazy_identity,
    _lazy_materialize,
    _lazy_operator_entries,
    _lazy_operator_resident_bytes,
    _lazy_operator_resident_entries,
    _try_deferred_expr_conv_relu,
    sparse_hz_apply_affine_expr_layer,
    sparse_hz_apply_layer,
    sparse_conv2d_matrix_from_layer_csr,
)
from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import SparseHZono


def _dyadic_kernel(
    out_channels: int,
    in_per_group: int,
    kernel: tuple[int, int],
) -> np.ndarray:
    count = out_channels * in_per_group * kernel[0] * kernel[1]
    return (
        np.arange(count, dtype=np.float64).reshape(
            out_channels, in_per_group, kernel[0], kernel[1]
        )
        % 17
        - 8
    ) / 16


def _reference_conv(
    kernel,
    input_shape,
    *,
    stride,
    padding,
    dilation,
    groups,
    row_mask=None,
):
    layer = SimpleNamespace(
        params={
            "input_shape": input_shape,
            "weight": torch.from_numpy(np.asarray(kernel)),
            "bias": None,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
    )
    reference, _ = sparse_conv2d_matrix_from_layer_csr(
        layer, keep_rows=row_mask
    )
    return reference


def _assert_csr_exact(left, right):
    left = left.tocsr()
    right = right.tocsr()
    assert left.shape == right.shape
    assert np.array_equal(left.indptr, right.indptr)
    assert np.array_equal(left.indices, right.indices)
    assert np.array_equal(left.data, right.data)


def _exact_sparse_source(size: int, frame_id: int = 71) -> SparseHZono:
    return SparseHZono(
        c=(np.arange(size, dtype=np.float64) % 7 - 3) / 16,
        Gc=sp.eye(size, format="csr", dtype=np.float64),
        Gb=sp.csr_matrix((size, 0), dtype=np.float64),
        Ac=sp.csr_matrix((0, size), dtype=np.float64),
        Ab=sp.csr_matrix((0, 0), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        frame_id=frame_id,
        exact=True,
    )


def _conv_layer(
    kernel,
    input_shape,
    *,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    layer_id=17,
):
    return SimpleNamespace(
        id=layer_id,
        kind="CONV2D",
        params={
            "input_shape": input_shape,
            "weight": torch.as_tensor(kernel, dtype=torch.float64),
            "bias": (
                None
                if bias is None
                else torch.as_tensor(bias, dtype=torch.float64)
            ),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        },
    )


def _tf_stub(*, implicit: bool, limit: int = 100_000):
    return SimpleNamespace(
        _SPARSE_MAX_AFFINE_CELLS=limit,
        _neural_hz_sparse_implicit_conv_dag=implicit,
        _neural_hz_sparse_deferred_relu_materialization=False,
        _neural_hz_sparse_lazy_affine_dag=True,
        _sparse_affine_expr_cache={},
        _sparse_precomputed_relu={},
        _neural_hz_lazy_affine_layers=0,
    )


def _flat_bounds(size: int, lower=-1.0, upper=1.0) -> Bounds:
    return Bounds(
        lb=torch.full((1, size), lower, dtype=torch.float64),
        ub=torch.full((1, size), upper, dtype=torch.float64),
    )


def _assert_sparse_hz_exact(left: SparseHZono, right: SparseHZono):
    assert left.frame_id == right.frame_id
    assert left.exact == right.exact
    for name in ("c", "b", "ub"):
        assert np.array_equal(getattr(left, name), getattr(right, name))
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        _assert_csr_exact(getattr(left, name), getattr(right, name))


def test_csr_and_diagonal_descriptors_are_exact_and_accounted():
    matrix = sp.csr_matrix(
        np.array(
            [
                [0.5, 0.0, -0.25, 0.0],
                [0.0, 2.0, 0.0, 1.0],
                [-1.0, 0.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
    )
    csr_op = CSRLinearOp(matrix)
    assert csr_op.shape == matrix.shape
    assert csr_op.logical_expanded_nnz == matrix.nnz
    assert csr_op.resident_entries == matrix.nnz
    assert csr_op.resident_bytes == (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    )
    _assert_csr_exact(csr_op.to_csr_reference(), matrix)

    vector = np.array([0.5, -0.25, 0.75, 1.0])
    assert np.array_equal(csr_op.matvec(vector), matrix @ vector)
    Q = sp.csr_matrix(
        np.array([[1.0, -0.5, 0.0], [0.0, 0.25, 2.0]])
    )
    expected = (Q @ matrix).tocsr()
    expected.eliminate_zeros()
    expected.sort_indices()
    _assert_csr_exact(
        csr_op.left_compose(Q, max_nnz=expected.nnz), expected
    )

    diagonal = np.array([0.5, 0.0, -2.0, 0.25])
    diagonal_op = DiagonalLinearOp(diagonal)
    diagonal_reference = sp.diags(diagonal, format="csr")
    diagonal_reference.eliminate_zeros()
    assert diagonal_op.shape == (4, 4)
    assert diagonal_op.resident_entries == 4
    assert diagonal_op.resident_bytes == diagonal.nbytes
    assert diagonal_op.logical_expanded_nnz == 3
    _assert_csr_exact(
        diagonal_op.to_csr_reference(), diagonal_reference
    )
    assert np.array_equal(
        diagonal_op.matvec(vector), diagonal_reference @ vector
    )
    Q_diagonal = sp.csr_matrix(
        np.array([[1.0, -0.5, 0.0, 0.25], [0.0, 0.25, 2.0, -1.0]])
    )
    expected_diagonal = (Q_diagonal @ diagonal_reference).tocsr()
    expected_diagonal.eliminate_zeros()
    expected_diagonal.sort_indices()
    _assert_csr_exact(
        diagonal_op.left_compose(Q_diagonal, expected_diagonal.nnz),
        expected_diagonal,
    )


@pytest.mark.parametrize(
    (
        "channels",
        "height",
        "width",
        "out_channels",
        "groups",
        "kernel_shape",
        "stride",
        "padding",
        "dilation",
    ),
    [
        (3, 5, 6, 4, 1, (3, 3), (1, 1), (1, 1), (1, 1)),
        (4, 6, 7, 6, 2, (2, 3), (2, 1), (1, 0), (2, 1)),
        (4, 5, 5, 4, 4, (3, 2), (1, 2), (1, 1), (1, 2)),
    ],
)
def test_implicit_conv_reference_is_array_identical_to_current_builder(
    channels,
    height,
    width,
    out_channels,
    groups,
    kernel_shape,
    stride,
    padding,
    dilation,
):
    kernel = _dyadic_kernel(
        out_channels, channels // groups, kernel_shape
    )
    input_shape = (1, channels, height, width)
    implicit = ImplicitConv2DOp(
        kernel,
        input_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    reference = _reference_conv(
        kernel,
        input_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    _assert_csr_exact(implicit.to_csr_reference(), reference)
    assert implicit.shape == reference.shape
    assert implicit.logical_expanded_nnz == reference.nnz
    assert implicit.resident_entries == kernel.size
    assert implicit.resident_bytes == kernel.nbytes


def test_implicit_conv_full_size_row_mask_matches_current_builder():
    input_shape = (1, 4, 6, 5)
    kernel = _dyadic_kernel(6, 2, (3, 2))
    unmasked = ImplicitConv2DOp(
        kernel,
        input_shape,
        stride=(2, 1),
        padding=(1, 1),
        dilation=(1, 2),
        groups=2,
    )
    row_mask = np.ones(unmasked.shape[0], dtype=bool)
    row_mask[1::3] = False
    implicit = ImplicitConv2DOp(
        kernel,
        input_shape,
        stride=(2, 1),
        padding=(1, 1),
        dilation=(1, 2),
        groups=2,
        row_mask=row_mask,
    )
    reference = _reference_conv(
        kernel,
        input_shape,
        stride=(2, 1),
        padding=(1, 1),
        dilation=(1, 2),
        groups=2,
        row_mask=row_mask,
    )

    _assert_csr_exact(implicit.to_csr_reference(), reference)
    assert implicit.logical_expanded_nnz == reference.nnz
    assert implicit.resident_entries == kernel.size + row_mask.size
    assert implicit.resident_bytes == kernel.nbytes + row_mask.nbytes
    assert np.all(reference.getnnz(axis=1)[~row_mask] == 0)


def test_implicit_conv_nchw_batch_is_exact_block_diagonal():
    kernel = _dyadic_kernel(4, 2, (2, 3))
    kwargs = {
        "stride": (1, 2),
        "padding": (1, 0),
        "dilation": (2, 1),
        "groups": 2,
    }
    implicit = ImplicitConv2DOp(kernel, (2, 4, 5, 6), **kwargs)
    one_sample = _reference_conv(
        kernel, (1, 4, 5, 6), row_mask=None, **kwargs
    )
    reference = sp.block_diag((one_sample, one_sample), format="csr")

    _assert_csr_exact(implicit.to_csr_reference(), reference)
    assert implicit.output_shape[0] == 2
    assert implicit.logical_expanded_nnz == 2 * one_sample.nnz


def test_implicit_conv_matvec_and_left_compose_are_exact_without_expansion(
    monkeypatch,
):
    kernel = _dyadic_kernel(4, 2, (3, 2))
    implicit = ImplicitConv2DOp(
        kernel,
        (1, 4, 5, 6),
        stride=(1, 2),
        padding=(1, 1),
        dilation=(1, 2),
        groups=2,
    )
    reference = implicit.to_csr_reference()
    vector = (np.arange(reference.shape[1], dtype=np.float64) % 13 - 6) / 8
    assert np.array_equal(implicit.matvec(vector), reference @ vector)

    q_dense = np.zeros((3, reference.shape[0]), dtype=np.float64)
    q_dense[0, 0:5] = np.array([0.5, -0.25, 0.0, 1.0, -0.5])
    q_dense[1, 4:9] = np.array([0.25, 0.5, -1.0, 0.5, 0.25])
    q_dense[2, -4:] = np.array([1.0, -0.5, 0.25, -0.125])
    Q = sp.csr_matrix(q_dense)
    expected = (Q @ reference).tocsr()
    expected.eliminate_zeros()
    expected.sort_indices()

    def forbidden_expansion(_self):
        raise AssertionError("left_compose expanded the Conv2D operator")

    monkeypatch.setattr(
        ImplicitConv2DOp, "to_csr_reference", forbidden_expansion
    )
    candidate = implicit.left_compose(Q, max_nnz=expected.nnz)
    _assert_csr_exact(candidate, expected)


def test_implicit_conv_nondyadic_results_match_reference_strictly():
    rng = np.random.default_rng(191)
    kernel = rng.normal(size=(6, 2, 2, 3))
    implicit = ImplicitConv2DOp(
        kernel,
        (1, 4, 5, 6),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
        groups=2,
    )
    reference = implicit.to_csr_reference()
    vector = rng.normal(size=reference.shape[1])
    Q = sp.random(
        4,
        reference.shape[0],
        density=0.25,
        format="csr",
        random_state=rng,
        data_rvs=lambda size: rng.normal(size=size),
    )
    expected = (Q @ reference).tocsr()
    expected.eliminate_zeros()
    candidate = implicit.left_compose(Q, max_nnz=expected.nnz)

    assert np.allclose(
        implicit.matvec(vector), reference @ vector, rtol=1e-13, atol=1e-13
    )
    assert np.allclose(
        candidate.toarray(), expected.toarray(), rtol=1e-13, atol=1e-13
    )


def test_left_compose_fails_closed_at_nnz_cap():
    implicit = ImplicitConv2DOp(
        np.ones((2, 2, 2, 2), dtype=np.float64),
        (1, 2, 4, 4),
        padding=1,
    )
    Q = sp.eye(implicit.shape[0], format="csr")[:3]
    expected = (Q @ implicit.to_csr_reference()).tocsr()
    expected.sort_indices()
    assert expected.nnz > 0
    _assert_csr_exact(
        implicit.left_compose(Q, max_nnz=expected.nnz), expected
    )
    with pytest.raises(MemoryError, match="max_nnz"):
        implicit.left_compose(Q, max_nnz=expected.nnz - 1)
    with pytest.raises(ValueError, match="max_nnz"):
        implicit.left_compose(Q, max_nnz=-1)
    with pytest.raises(ValueError, match="max_nnz"):
        implicit.left_compose(Q, max_nnz=True)


@pytest.mark.parametrize(
    "factory, match",
    [
        (lambda: CSRLinearOp([[np.nan]]), "non-finite"),
        (lambda: DiagonalLinearOp([np.inf]), "non-finite"),
        (
            lambda: ImplicitConv2DOp(
                np.ones((2, 2, 1, 1)), (1, 3, 4, 4), groups=1
            ),
            "grouped",
        ),
        (
            lambda: ImplicitConv2DOp(
                np.ones((3, 1, 1, 1)), (1, 2, 4, 4), groups=2
            ),
            "grouped",
        ),
        (
            lambda: ImplicitConv2DOp(
                np.ones((2, 2, 1, 1)), (1, 2, 4, 4), stride=0
            ),
            "stride",
        ),
        (
            lambda: ImplicitConv2DOp(
                np.ones((2, 2, 1, 1)),
                (1, 2, 4, 4),
                row_mask=np.ones(31, dtype=bool),
            ),
            "row_mask",
        ),
        (
            lambda: ImplicitConv2DOp(
                np.ones((2, 2, 1, 1)),
                (1, 2, 4, 4),
                row_mask=np.ones(32, dtype=np.int64),
            ),
            "boolean",
        ),
    ],
)
def test_descriptors_reject_invalid_or_nonfinite_state(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_execution_rejects_shape_and_nonfinite_results():
    implicit = ImplicitConv2DOp(
        np.array([[[[1e308]]]], dtype=np.float64), (1, 1, 1, 1)
    )
    with pytest.raises(ValueError, match="length mismatch"):
        implicit.matvec(np.ones(2))
    with pytest.raises(ValueError, match="shape mismatch"):
        implicit.left_compose(np.ones((1, 2)), max_nnz=1)
    with pytest.raises(ValueError, match="non-finite"):
        implicit.matvec(np.array([1e308]))
    with pytest.raises(ValueError, match="non-finite"):
        implicit.left_compose(np.array([[1e308]]), max_nnz=1)
    with pytest.raises(ValueError, match="non-finite"):
        implicit.left_compose(np.array([[np.nan]]), max_nnz=1)


def test_lazy_conv_flag_off_is_legacy_csr_and_never_constructs_implicit(
    monkeypatch,
):
    kernel = _dyadic_kernel(2, 2, (1, 1))
    layer = _conv_layer(
        kernel,
        (1, 2, 3, 3),
        bias=np.array([0.25, -0.5]),
    )
    source = _exact_sparse_source(18)
    expr = _lazy_identity(source)
    tf = _tf_stub(implicit=False)

    def forbidden_constructor(*_args, **_kwargs):
        raise AssertionError("flag-off constructed an implicit Conv2D")

    monkeypatch.setattr(
        tf_cnn_module, "ImplicitConv2DOp", forbidden_constructor
    )
    handled, sparse, out_expr, reason = sparse_hz_apply_affine_expr_layer(
        layer,
        expr,
        _flat_bounds(18),
        SimpleNamespace(bounds=_flat_bounds(18)),
        tf,
    )
    reference, reference_bias = sparse_conv2d_matrix_from_layer_csr(layer)

    assert handled and sparse is None and reason is None
    assert sp.issparse(out_expr.terms[0].operators[-1])
    _assert_csr_exact(out_expr.terms[0].operators[-1], reference)
    assert np.array_equal(out_expr.bias, reference_bias)


def test_lazy_conv_flag_on_uses_resident_gate_but_materialization_keeps_cap():
    kernel = _dyadic_kernel(2, 2, (3, 3))
    layer = _conv_layer(
        kernel,
        (1, 2, 5, 5),
        bias=np.array([0.25, -0.5]),
        padding=1,
    )
    source = _exact_sparse_source(50)
    expr = _lazy_identity(source)
    tf = _tf_stub(implicit=True, limit=100)
    bounds = _flat_bounds(50)

    handled, sparse, out_expr, reason = sparse_hz_apply_affine_expr_layer(
        layer, expr, bounds, SimpleNamespace(bounds=bounds), tf
    )
    operator = out_expr.terms[0].operators[-1]

    assert handled and sparse is None and reason is None
    assert isinstance(operator, ImplicitConv2DOp)
    assert operator.resident_entries + out_expr.bias.size <= 100
    assert operator.logical_expanded_nnz + out_expr.bias.size > 100
    assert _lazy_operator_entries(out_expr) == (
        operator.logical_expanded_nnz + out_expr.bias.size
    )
    assert _lazy_operator_resident_entries(out_expr) == (
        operator.resident_entries + out_expr.bias.size
    )
    assert _lazy_operator_resident_bytes(out_expr) == (
        operator.resident_bytes + out_expr.bias.nbytes
    )
    assert tf._neural_hz_implicit_conv_ops == 1
    assert tf._neural_hz_implicit_conv_profile[-1][
        "logical_expanded_nnz"
    ] == operator.logical_expanded_nnz
    with pytest.raises(MemoryError, match="storage limit"):
        _lazy_materialize(
            out_expr,
            np.ones(out_expr.n_out, dtype=bool),
            limit=100,
        )


def test_lazy_conv_flag_on_starts_dag_with_implicit_operator_and_full_bias():
    kernel = _dyadic_kernel(2, 2, (1, 1))
    layer = _conv_layer(
        kernel,
        (1, 2, 3, 3),
        bias=np.array([0.375, -0.625]),
    )
    source = _exact_sparse_source(18)
    tf = _tf_stub(implicit=True)
    bounds = _flat_bounds(18)

    handled, sparse, reason = sparse_hz_apply_layer(
        layer,
        source,
        bounds,
        SimpleNamespace(bounds=bounds),
        tf,
    )
    out_expr = tf._sparse_affine_expr_cache[layer.id]
    _, reference_bias = sparse_conv2d_matrix_from_layer_csr(layer)

    assert handled and sparse is None and reason == "lazy_affine_expr"
    assert isinstance(out_expr.terms[0].operators[-1], ImplicitConv2DOp)
    assert np.array_equal(out_expr.bias, reference_bias)
    assert tf._neural_hz_lazy_affine_layers == 1
    assert tf._neural_hz_implicit_conv_ops == 1


def test_deferred_lazy_conv_uses_full_width_implicit_row_mask_and_bias(
    monkeypatch,
):
    kernel = np.array([[[[0.5]]]], dtype=np.float64)
    layer = _conv_layer(
        kernel,
        (1, 1, 2, 2),
        bias=np.array([0.25]),
    )
    source = _exact_sparse_source(4)
    expr = _lazy_identity(source)
    bounds = Bounds(
        lb=torch.tensor([[-2.0, -1.0, 0.0, 0.25]], dtype=torch.float64),
        ub=torch.tensor([[-0.5, 1.0, 1.0, 2.0]], dtype=torch.float64),
    )
    relu = SimpleNamespace(id=91, kind="RELU")
    tf = _tf_stub(implicit=True)
    captured = {}

    monkeypatch.setattr(
        tf_cnn_module,
        "_deferred_relu_island",
        lambda _layer, _tf: ([], relu),
    )

    def selective(current, input_bounds, _tf, _layer):
        captured["expression"] = current
        return SparseHZPhaseSelectiveResult(
            core=source,
            expression=current,
            output_bounds=input_bounds,
        )

    monkeypatch.setattr(
        tf_cnn_module, "_try_phase_selective_exact_relu", selective
    )
    handled, sparse, out_expr, reason = _try_deferred_expr_conv_relu(
        layer, expr, SimpleNamespace(bounds=bounds), tf
    )
    current = captured["expression"]
    operator = current.terms[0].operators[-1]
    stable_negative = np.array([True, False, False, False])
    reference, reference_bias = sparse_conv2d_matrix_from_layer_csr(
        layer, keep_rows=~stable_negative
    )

    assert handled and sparse is None and out_expr is None
    assert reason == "deferred_lazy_phase_selective_to_relu:91"
    assert isinstance(operator, ImplicitConv2DOp)
    _assert_csr_exact(operator.to_csr_reference(), reference)
    assert np.array_equal(current.bias, reference_bias)
    assert current.bias[0] == 0.25
    assert tf._sparse_precomputed_relu[relu.id][3] is current
    assert tf._neural_hz_implicit_conv_ops == 1
    assert tf._neural_hz_implicit_conv_profile[-1]["masked_rows"] == 1


def test_generic_lazy_materialization_matches_legacy_csr_in_chain(
    monkeypatch,
):
    kernel = _dyadic_kernel(2, 2, (1, 1))
    layer = _conv_layer(
        kernel,
        (1, 2, 3, 3),
        bias=np.array([0.25, -0.5]),
    )
    source = _exact_sparse_source(18)
    first_scale = sp.diags(
        (np.arange(18, dtype=np.float64) % 5 + 1) / 8,
        format="csr",
    )
    final_scale = sp.diags(
        (np.arange(18, dtype=np.float64) % 3 - 1) / 4,
        format="csr",
    )
    first_bias = (np.arange(18, dtype=np.float64) % 7 - 3) / 32
    final_bias = (np.arange(18, dtype=np.float64) % 3) / 64
    tf = _tf_stub(implicit=True)
    implicit, conv_bias = _lazy_conv2d_operator_and_bias(layer, tf)
    reference, reference_bias = sparse_conv2d_matrix_from_layer_csr(layer)

    candidate = _lazy_append_linear(
        _lazy_add_const(_lazy_identity(source), first_bias),
        first_scale,
        None,
        100_000,
    )
    candidate = _lazy_append_linear(
        candidate, implicit, conv_bias, 100_000
    )
    candidate = _lazy_append_linear(
        candidate, final_scale, final_bias, 100_000
    )
    baseline = _lazy_append_linear(
        _lazy_add_const(_lazy_identity(source), first_bias),
        first_scale,
        None,
        100_000,
    )
    baseline = _lazy_append_linear(
        baseline, reference, reference_bias, 100_000
    )
    baseline = _lazy_append_linear(
        baseline, final_scale, final_bias, 100_000
    )
    keep = np.ones(18, dtype=bool)
    keep[1::4] = False

    def forbidden_expansion(_self):
        raise AssertionError("lazy materialization expanded implicit Conv2D")

    monkeypatch.setattr(
        ImplicitConv2DOp, "to_csr_reference", forbidden_expansion
    )
    candidate_hz = _lazy_materialize(candidate, keep, 100_000)
    baseline_hz = _lazy_materialize(baseline, keep, 100_000)

    assert np.array_equal(candidate.bias, baseline.bias)
    _assert_sparse_hz_exact(candidate_hz, baseline_hz)
    expected_map = (
        sp.diags(keep.astype(np.float64), format="csr")
        @ final_scale
        @ reference
        @ first_scale
    ).tocsr()
    expected_map.eliminate_zeros()
    _assert_csr_exact(candidate_hz.Gc, expected_map)


def test_lazy_materialization_memoizes_shared_reverse_operator_prefix(
    monkeypatch,
):
    source = _exact_sparse_source(4)
    shared = ImplicitConv2DOp(
        np.array([[[[0.5]]]], dtype=np.float64), (1, 1, 2, 2)
    )
    first = sp.diags(np.array([0.5, 1.0, -0.5, 0.25]), format="csr")
    second = sp.diags(np.array([1.0, -0.25, 0.5, 0.75]), format="csr")
    expression = SparseHZAffineExpr(
        terms=(
            SparseHZAffineTerm(source, (first, shared)),
            SparseHZAffineTerm(source, (second, shared)),
        ),
        bias=np.zeros(4),
        n_out=4,
        frame_id=source.frame_id,
    )
    reference = shared.to_csr_reference()
    calls = {id(shared): 0}
    original = ImplicitConv2DOp.left_compose

    def counted(self, Q, max_nnz):
        calls[id(self)] += 1
        return original(self, Q, max_nnz)

    monkeypatch.setattr(ImplicitConv2DOp, "left_compose", counted)
    result = _lazy_materialize(
        expression, np.ones(4, dtype=bool), limit=100
    )
    expected = (reference @ first + reference @ second).tocsr()
    expected.eliminate_zeros()
    expected.sort_indices()

    assert calls[id(shared)] == 1
    _assert_csr_exact(result.Gc, expected)


def test_lazy_reverse_prefix_cache_has_strict_total_nnz_budget(monkeypatch):
    source = SparseHZono(
        c=np.zeros(4),
        Gc=sp.csr_matrix((4, 0)),
        Gb=sp.csr_matrix((4, 0)),
        Ac=sp.csr_matrix((0, 0)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=73,
        exact=True,
    )
    first = ImplicitConv2DOp(
        np.array([[[[0.5]]]], dtype=np.float64), (1, 1, 2, 2)
    )
    second = ImplicitConv2DOp(
        np.array([[[[0.5]]]], dtype=np.float64), (1, 1, 2, 2)
    )
    expression = SparseHZAffineExpr(
        terms=(
            SparseHZAffineTerm(source, (first,)),
            SparseHZAffineTerm(source, (first,)),
            SparseHZAffineTerm(source, (second,)),
            SparseHZAffineTerm(source, (second,)),
        ),
        bias=np.zeros(4),
        n_out=4,
        frame_id=source.frame_id,
    )
    calls = {id(first): 0, id(second): 0}
    original = ImplicitConv2DOp.left_compose

    def counted(self, Q, max_nnz):
        calls[id(self)] += 1
        return original(self, Q, max_nnz)

    monkeypatch.setattr(ImplicitConv2DOp, "left_compose", counted)
    result = _lazy_materialize(
        expression, np.ones(4, dtype=bool), limit=4
    )

    assert result.n_out == 4
    assert calls[id(first)] == 1
    assert calls[id(second)] == 2


def test_descriptor_content_arena_is_weak_and_metrics_count_shared_once():
    kernel = _dyadic_kernel(2, 2, (1, 1))
    layer = _conv_layer(kernel, (1, 2, 3, 3))
    source = _exact_sparse_source(18)
    tf = _tf_stub(implicit=True)
    tf._neural_hz_linear_op_arena = weakref.WeakValueDictionary()

    first, _ = _lazy_conv2d_operator_and_bias(layer, tf)
    second, _ = _lazy_conv2d_operator_and_bias(layer, tf)
    assert first is second
    assert first.content_key == second.content_key
    expr = SparseHZAffineExpr(
        terms=(
            SparseHZAffineTerm(source, (first,)),
            SparseHZAffineTerm(source, (second,)),
        ),
        bias=np.zeros(first.shape[0]),
        n_out=first.shape[0],
        frame_id=source.frame_id,
    )
    assert _lazy_operator_entries(expr) == (
        expr.bias.size + first.logical_expanded_nnz
    )
    assert _lazy_operator_resident_entries(expr) == (
        expr.bias.size + first.resident_entries
    )
    assert len(tf._neural_hz_linear_op_arena) == 1

    del expr, first, second
    gc.collect()
    assert len(tf._neural_hz_linear_op_arena) == 0


def test_implicit_lazy_start_failure_is_fail_closed_without_cache_side_effect(
    monkeypatch,
):
    layer = _conv_layer(
        _dyadic_kernel(2, 2, (1, 1)), (1, 2, 3, 3)
    )
    source = _exact_sparse_source(18)
    tf = _tf_stub(implicit=True)
    bounds = _flat_bounds(18)

    class InvalidImplicitConv:
        def __init__(self, *_args, **_kwargs):
            raise ValueError("synthetic invalid descriptor")

    monkeypatch.setattr(
        tf_cnn_module, "ImplicitConv2DOp", InvalidImplicitConv
    )
    handled, sparse, reason = sparse_hz_apply_layer(
        layer,
        source,
        bounds,
        SimpleNamespace(bounds=bounds),
        tf,
    )

    assert handled and sparse is None
    assert reason == "lazy_affine_start:ValueError"
    assert tf._sparse_affine_expr_cache == {}
    assert tf._neural_hz_lazy_affine_layers == 0
