import numpy as np
import pytest
import torch
from types import SimpleNamespace

sp = pytest.importorskip("scipy.sparse")
from scipy.optimize import linprog

from act.back_end.hybridz_tf.tf_mlp import (
    _sparse_exact_duplicate_groups,
    _sparse_exact_positive_proportional_groups,
    _sparse_exact_signed_proportional_groups,
    _sparse_exact_signed_duplicate_groups,
    _sparse_apply_relu,
    _exact_dyadic_sum_is_zero,
    _sparse_drop_canceled_relu_graphs,
    hz_apply_relu,
    hz_apply_relu_compact_exact,
    hz_apply_relu_fill_aware_exact,
    sparse_hz_apply_relu_compact_exact,
    sparse_hz_apply_relu_shared_exact,
    sparse_hz_apply_relu_signed_shared_exact,
    sparse_hz_apply_relu_signed_shared_compact_exact,
)
from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    _lower_hz_milp,
    sparse_hz_add_same_frame,
    sparse_hz_linear,
    sparse_hz_rebase_image_exact,
)
from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
from act.back_end.hybridz_tf.tf_cnn import (
    SparseHZAffineExpr,
    SparseHZAffineTerm,
    _lazy_add,
    _lazy_append_linear,
    _lazy_checkpoint,
    _lazy_from_hz_linear,
    _lazy_identity,
    _lazy_materialize,
    _sparse_hz_storage_entries,
    _try_phase_separate_exact_relu,
    _try_phase_selective_exact_relu,
    sparse_conv2d_matrix_from_layer,
    sparse_conv2d_matrix_from_layer_csr,
)
from act.config.config import HybridZConfig


def _interval_hz(lower: float, upper: float) -> HZono:
    center = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    return HZono(
        c=torch.tensor([[center]], dtype=torch.float64),
        Gc=torch.tensor([[radius]], dtype=torch.float64),
        Gb=torch.zeros((1, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 1), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=torch.zeros(0, dtype=torch.bool),
        col_ids=torch.tensor([7], dtype=torch.long),
        bcol_ids=torch.zeros(0, dtype=torch.long),
    )


@pytest.mark.parametrize(
    ("channels", "height", "width", "out_channels", "groups", "kernel", "stride", "padding", "dilation"),
    [
        (3, 5, 6, 4, 1, (3, 3), (1, 1), (1, 1), (1, 1)),
        (4, 6, 7, 6, 2, (2, 3), (2, 1), (1, 0), (2, 1)),
        (4, 5, 5, 4, 4, (3, 2), (1, 2), (1, 1), (1, 2)),
    ],
)
def test_row_native_conv2d_csr_is_identical_to_loop_builder(
    channels,
    height,
    width,
    out_channels,
    groups,
    kernel,
    stride,
    padding,
    dilation,
):
    in_per_group = channels // groups
    count = out_channels * in_per_group * kernel[0] * kernel[1]
    weight = (
        torch.arange(count, dtype=torch.float64).reshape(
            out_channels, in_per_group, kernel[0], kernel[1]
        )
        % 11
        - 5
    ) / 8
    bias = torch.arange(out_channels, dtype=torch.float64) / 16
    layer = SimpleNamespace(
        params={
            "input_shape": (1, channels, height, width),
            "weight": weight,
            "bias": bias,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
    )

    reference, reference_bias = sparse_conv2d_matrix_from_layer(layer)
    candidate, candidate_bias = sparse_conv2d_matrix_from_layer_csr(layer)

    assert candidate.shape == reference.shape
    assert np.array_equal(candidate.indptr, reference.indptr)
    assert np.array_equal(candidate.indices, reference.indices)
    assert np.array_equal(candidate.data, reference.data)
    assert np.array_equal(candidate_bias, reference_bias)


def test_row_native_conv2d_csr_mask_zeros_only_selected_output_rows():
    layer = SimpleNamespace(
        params={
            "input_shape": (1, 2, 4, 4),
            "weight": torch.arange(1, 37, dtype=torch.float64).reshape(
                2, 2, 3, 3
            ),
            "bias": torch.tensor([0.25, -0.5], dtype=torch.float64),
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
        }
    )
    reference, reference_bias = sparse_conv2d_matrix_from_layer_csr(layer)
    keep = np.ones(reference.shape[0], dtype=bool)
    keep[::3] = False
    candidate, candidate_bias = sparse_conv2d_matrix_from_layer_csr(
        layer, keep_rows=keep
    )
    retained_candidate = candidate[keep].tocsr()
    retained_reference = reference[keep].tocsr()

    assert np.all(candidate.getnnz(axis=1)[~keep] == 0)
    assert np.array_equal(
        retained_candidate.indptr, retained_reference.indptr
    )
    assert np.array_equal(
        retained_candidate.indices, retained_reference.indices
    )
    assert np.array_equal(retained_candidate.data, retained_reference.data)
    assert np.array_equal(candidate_bias, reference_bias)


def test_forced_stable_negative_materialization_matches_full_exact_relu():
    full = SparseHZono(
        c=np.array([-2.0, 0.5]),
        Gc=sp.csr_matrix([[0.5], [1.5]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=19,
        exact=True,
    )
    masked = SparseHZono(
        c=np.array([0.0, 0.5]),
        Gc=sp.csr_matrix([[0.0], [1.5]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=19,
        exact=True,
    )
    bounds = Bounds(
        lb=torch.tensor([[-2.5, -1.0]], dtype=torch.float64),
        ub=torch.tensor([[-1.5, 2.0]], dtype=torch.float64),
    )
    layer = SimpleNamespace(id=23, kind="RELU")
    full_tf = HybridzTF()
    full_tf._sparse_frame_widths[19] = (1, 0)
    masked_tf = HybridzTF()
    masked_tf._sparse_frame_widths[19] = (1, 0)

    expected, expected_reason = _sparse_apply_relu(
        layer, full, bounds, full_tf
    )
    candidate, candidate_reason = _sparse_apply_relu(
        layer,
        masked,
        bounds,
        masked_tf,
        forced_stable_negative=np.array([True, False]),
    )

    assert expected_reason is candidate_reason is None
    assert np.array_equal(candidate.c, expected.c)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        left = getattr(candidate, name)
        right = getattr(expected, name)
        assert np.array_equal(left.indptr, right.indptr)
        assert np.array_equal(left.indices, right.indices)
        assert np.array_equal(left.data, right.data)
    assert np.array_equal(candidate.b, expected.b)
    assert np.array_equal(candidate.ub, expected.ub)
    assert candidate.exact and expected.exact


def test_lazy_residual_affine_dag_matches_eager_exact_relu():
    source = SparseHZono(
        c=np.zeros(2),
        Gc=sp.eye(2, format="csr"),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 2)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=31,
        exact=True,
    )
    first = sp.eye(2, format="csr")
    first_bias = np.array([-3.0, 0.0])
    second = sp.eye(2, format="csr")
    second_bias = np.zeros(2)

    eager_main = sparse_hz_linear(source, first, first_bias)
    eager_residual = sparse_hz_add_same_frame(eager_main, source)
    eager_pre_relu = sparse_hz_linear(eager_residual, second, second_bias)

    lazy_main = _lazy_from_hz_linear(source, first, first_bias, 1000)
    lazy_residual = _lazy_add(lazy_main, _lazy_identity(source), 1000)
    checkpoint = _lazy_checkpoint(lazy_residual, 1000)
    checkpoint_hz = checkpoint.terms[0].source
    assert checkpoint.terms[0].operators == ()
    assert np.array_equal(checkpoint_hz.c, eager_residual.c)
    assert np.array_equal(
        checkpoint_hz.Gc.toarray(), eager_residual.Gc.toarray()
    )
    assert np.array_equal(
        checkpoint_hz.Gb.toarray(), eager_residual.Gb.toarray()
    )
    lazy_output = _lazy_append_linear(
        lazy_residual, second, second_bias, 1000
    )
    stable_negative = np.array([True, False])
    lazy_pre_relu = _lazy_materialize(
        lazy_output, ~stable_negative, 1000
    )
    bounds = Bounds(
        lb=torch.tensor([[-5.0, -2.0]], dtype=torch.float64),
        ub=torch.tensor([[-1.0, 2.0]], dtype=torch.float64),
    )
    layer = SimpleNamespace(id=37, kind="RELU")
    eager_tf = HybridzTF()
    eager_tf._sparse_frame_widths[31] = (2, 0)
    lazy_tf = HybridzTF()
    lazy_tf._sparse_frame_widths[31] = (2, 0)
    expected, expected_reason = _sparse_apply_relu(
        layer, eager_pre_relu, bounds, eager_tf
    )
    candidate, candidate_reason = _sparse_apply_relu(
        layer,
        lazy_pre_relu,
        bounds,
        lazy_tf,
        forced_stable_negative=stable_negative,
    )

    assert expected_reason is candidate_reason is None
    assert np.array_equal(candidate.c, expected.c)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        left = getattr(candidate, name)
        right = getattr(expected, name)
        assert np.array_equal(left.indptr, right.indptr)
        assert np.array_equal(left.indices, right.indices)
        assert np.array_equal(left.data, right.data)
    assert np.array_equal(candidate.b, expected.b)
    assert np.array_equal(candidate.ub, expected.ub)
    assert candidate.exact and expected.exact


def test_lazy_transient_sum_is_scoped_to_phase_consumer():
    first = SparseHZono(
        c=np.zeros(4),
        Gc=sp.eye(4, format="csr"),
        Gb=sp.csr_matrix((4, 0)),
        Ac=sp.csr_matrix((0, 4)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=43,
        exact=True,
    )
    second = SparseHZono(
        c=np.zeros(4),
        Gc=sp.csr_matrix(
            (
                np.ones(4),
                (np.arange(4), np.array([1, 2, 3, 0])),
            ),
            shape=(4, 4),
        ),
        Gb=sp.csr_matrix((4, 0)),
        Ac=sp.csr_matrix((0, 4)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=43,
        exact=True,
    )
    expr = SparseHZAffineExpr(
        terms=(
            SparseHZAffineTerm(first, ()),
            SparseHZAffineTerm(second, ()),
        ),
        bias=np.zeros(4),
        n_out=4,
        frame_id=43,
    )

    with pytest.raises(MemoryError, match="residual materialization"):
        _lazy_materialize(expr, np.ones(4, dtype=bool), 10)
    transient = _lazy_materialize(
        expr,
        np.ones(4, dtype=bool),
        10,
        allow_transient_sum=True,
    )
    assert transient.exact
    assert transient.Gc.nnz == 8


def _phase_separation_fixture(*, dense_positive: bool):
    positive = (
        np.full(10, 0.05, dtype=np.float64)
        if dense_positive
        else np.array([0.5] + [0.0] * 9, dtype=np.float64)
    )
    generators = np.vstack(
        [
            np.array([0.25] + [0.0] * 9, dtype=np.float64),
            positive,
            np.array([1.0] + [0.0] * 9, dtype=np.float64),
        ]
    )
    source = SparseHZono(
        c=np.array([-2.0, 2.0, 0.0]),
        Gc=sp.csr_matrix(generators),
        Gb=sp.csr_matrix([[0.0], [0.1], [0.2]]),
        Ac=sp.csr_matrix(
            ([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, 10)
        ),
        Ab=sp.csr_matrix([[1.0]]),
        b=np.array([0.0]),
        Auc=sp.csr_matrix(
            ([1.0], ([0], [1])), shape=(1, 10)
        ),
        Aub=sp.csr_matrix([[-1.0]]),
        ub=np.array([1.5]),
        frame_id=51,
        exact=True,
    )
    bounds = Bounds(
        lb=torch.tensor([[-3.0, 1.0, -2.0]], dtype=torch.float64),
        ub=torch.tensor([[-1.0, 3.0, 2.0]], dtype=torch.float64),
    )
    layer = SimpleNamespace(id=41, kind="RELU")
    tf = HybridzTF(
        HybridZConfig(sparse_phase_separated_relu=True)
    )
    tf._sparse_frame_widths[51] = (10, 1)
    completed, reason = _sparse_apply_relu(layer, source, bounds, tf)
    assert reason is None
    return source, bounds, layer, tf, completed


def test_phase_separated_lazy_relu_materializes_to_exact_completed_hz():
    source, bounds, layer, tf, completed = _phase_separation_fixture(
        dense_positive=True
    )
    tf._SPARSE_MAX_AFFINE_CELLS = _sparse_hz_storage_entries(completed) + 8
    separated = _try_phase_separate_exact_relu(
        _lazy_identity(source),
        completed,
        source,
        bounds,
        tf,
        layer.id,
    )

    assert separated is not None
    assert len(separated.terms) == 2
    core = separated.terms[-1].source
    assert core.exact and core.frame_id == completed.frame_id
    assert core.n_cont == completed.n_cont
    assert core.n_bin == completed.n_bin
    assert core.n_eq == completed.n_eq
    assert core.n_ineq == completed.n_ineq
    for name in ("Ac", "Ab", "Auc", "Aub"):
        left = getattr(core, name)
        right = getattr(completed, name)
        assert np.array_equal(left.indptr, right.indptr)
        assert np.array_equal(left.indices, right.indices)
        assert np.array_equal(left.data, right.data)
    assert np.array_equal(core.b, completed.b)
    assert np.array_equal(core.ub, completed.ub)

    rematerialized = _lazy_materialize(
        separated,
        np.ones(completed.n_out, dtype=bool),
        10_000,
    )
    assert np.array_equal(rematerialized.c, completed.c)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        left = getattr(rematerialized, name)
        right = getattr(completed, name)
        assert np.array_equal(left.indptr, right.indptr)
        assert np.array_equal(left.indices, right.indices)
        assert np.array_equal(left.data, right.data)
    assert np.array_equal(rematerialized.b, completed.b)
    assert np.array_equal(rematerialized.ub, completed.ub)

    downstream = sp.csr_matrix([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    expected = sparse_hz_linear(completed, downstream)
    candidate = sparse_hz_linear(rematerialized, downstream)
    assert np.array_equal(candidate.c, expected.c)
    assert np.array_equal(candidate.Gc.toarray(), expected.Gc.toarray())
    assert np.array_equal(candidate.Gb.toarray(), expected.Gb.toarray())
    profile = tf._neural_hz_phase_separated_profile[-1]
    assert profile["stable_negative"] == 1
    assert profile["stable_positive"] == 1
    assert profile["unstable"] == 1
    assert profile["removed_value_nnz"] > profile["added_mask_bias_entries"]
    assert profile["core_storage"] <= tf._SPARSE_MAX_AFFINE_CELLS


def test_phase_separated_lazy_relu_is_default_off_and_rejects_no_savings():
    source, bounds, layer, tf, completed = _phase_separation_fixture(
        dense_positive=False
    )
    limit = _sparse_hz_storage_entries(completed) + 8
    tf._SPARSE_MAX_AFFINE_CELLS = limit
    expr = _lazy_identity(source)

    tf._neural_hz_sparse_phase_separated_relu = False
    assert (
        _try_phase_separate_exact_relu(
            expr, completed, source, bounds, tf, layer.id
        )
        is None
    )
    tf._neural_hz_sparse_phase_separated_relu = True
    assert (
        _try_phase_separate_exact_relu(
            expr, completed, source, bounds, tf, layer.id
        )
        is None
    )
    assert tf._neural_hz_phase_separated_relus == 0


def _phase_selective_fixture():
    n_cont = 10
    n_positive = 9
    n_out = n_positive + 2
    generators = np.zeros((n_out, n_cont), dtype=np.float64)
    generators[0, 0] = 0.25
    generators[1 : 1 + n_positive] = 0.05
    generators[-1, 0] = 1.0
    binary = np.zeros((n_out, 1), dtype=np.float64)
    binary[1 : 1 + n_positive, 0] = 0.1
    binary[-1, 0] = 0.2
    source = SparseHZono(
        c=np.array([-2.0] + [2.0] * n_positive + [0.0]),
        Gc=sp.csr_matrix(generators),
        Gb=sp.csr_matrix(binary),
        Ac=sp.csr_matrix(
            ([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, n_cont)
        ),
        Ab=sp.csr_matrix([[1.0]]),
        b=np.array([0.0]),
        Auc=sp.csr_matrix(
            ([1.0], ([0], [1])), shape=(1, n_cont)
        ),
        Aub=sp.csr_matrix([[-1.0]]),
        ub=np.array([1.5]),
        frame_id=59,
        exact=True,
    )
    bias = np.array([0.375] + [0.25] * n_positive + [-0.125])
    expr = SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(source, ()),),
        bias=bias,
        n_out=n_out,
        frame_id=59,
    )
    bounds = Bounds(
        lb=torch.tensor(
            [[-3.0] + [1.0] * n_positive + [-2.0]],
            dtype=torch.float64,
        ),
        ub=torch.tensor(
            [[-1.0] + [3.0] * n_positive + [2.0]],
            dtype=torch.float64,
        ),
    )
    layer = SimpleNamespace(id=73, kind="RELU")
    tf = HybridzTF()
    tf._neural_hz_sparse_phase_selective_materialization = True
    tf._SPARSE_MAX_AFFINE_CELLS = 20_000
    tf._sparse_frame_widths[59] = (n_cont, 1)
    return source, expr, bounds, layer, tf


def test_phase_selective_zeroes_nonzero_bias_before_synthetic_relu(
    monkeypatch,
):
    _, expr, bounds, layer, tf = _phase_selective_fixture()
    seen = {}
    real_apply_relu = hz_cnn._sparse_apply_relu

    def capture_apply_relu(layer_arg, hz, synthetic, tf_arg, **kwargs):
        seen["hz"] = hz
        seen["bounds"] = synthetic
        return real_apply_relu(layer_arg, hz, synthetic, tf_arg, **kwargs)

    monkeypatch.setattr(hz_cnn, "_sparse_apply_relu", capture_apply_relu)
    result = _try_phase_selective_exact_relu(expr, bounds, tf, layer)

    assert result is not None
    unstable_row = expr.n_out - 1
    outside = np.arange(expr.n_out) != unstable_row
    synthetic_input = seen["hz"]
    assert np.all(synthetic_input.c[outside] == 0.0)
    assert synthetic_input.Gc[outside].nnz == 0
    assert synthetic_input.Gb[outside].nnz == 0
    synthetic_lb = seen["bounds"].lb.reshape(-1).numpy()
    synthetic_ub = seen["bounds"].ub.reshape(-1).numpy()
    assert np.all(synthetic_lb[outside] == 0.0)
    assert np.all(synthetic_ub[outside] == 0.0)
    assert synthetic_lb[unstable_row] == bounds.lb.reshape(-1)[unstable_row]
    assert synthetic_ub[unstable_row] == bounds.ub.reshape(-1)[unstable_row]


def test_phase_selective_materialization_is_byte_identical_and_keeps_slots():
    _, expr, bounds, layer, candidate_tf = _phase_selective_fixture()
    candidate = _try_phase_selective_exact_relu(
        expr, bounds, candidate_tf, layer
    )
    assert candidate is not None

    full_tf = HybridzTF()
    full_tf._SPARSE_MAX_AFFINE_CELLS = 20_000
    full_tf._sparse_frame_widths[59] = (10, 1)
    preactivation = _lazy_materialize(
        expr,
        np.ones(expr.n_out, dtype=bool),
        20_000,
    )
    completed, reason = _sparse_apply_relu(
        layer, preactivation, bounds, full_tf
    )
    assert reason is None
    rematerialized = _lazy_materialize(
        candidate.expression,
        np.ones(expr.n_out, dtype=bool),
        20_000,
    )

    assert rematerialized.n_out == completed.n_out == expr.n_out
    assert rematerialized.frame_id == completed.frame_id == expr.frame_id
    assert rematerialized.exact and completed.exact
    for name in ("c", "b", "ub"):
        assert np.array_equal(
            getattr(rematerialized, name), getattr(completed, name)
        )
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        left = getattr(rematerialized, name)
        right = getattr(completed, name)
        assert left.shape == right.shape
        assert np.array_equal(left.indptr, right.indptr)
        assert np.array_equal(left.indices, right.indices)
        assert np.array_equal(left.data, right.data)

    assert candidate_tf._sparse_relu_slots == full_tf._sparse_relu_slots
    assert candidate_tf._sparse_frame_widths == full_tf._sparse_frame_widths
    assert set(candidate_tf._sparse_relu_slots) == {(59, 73, 10)}
    assert candidate.core.n_out == expr.n_out
    assert candidate.core.n_bin == completed.n_bin
    assert candidate.core.n_eq == completed.n_eq
    assert candidate.core.n_ineq == completed.n_ineq

    profile = candidate_tf._neural_hz_phase_selective_profile[-1]
    assert profile["stable_negative"] == 1
    assert profile["stable_positive"] == 9
    assert profile["unstable"] == 1
    assert profile["probe_positive"] == 8
    assert profile["omitted_positive"] == 1
    assert profile["probe_generator_nnz"] > profile[
        "added_mask_bias_entries"
    ]
    hint_lb = candidate.output_bounds.lb.reshape(-1)
    hint_ub = candidate.output_bounds.ub.reshape(-1)
    assert hint_lb[0] == hint_ub[0] == 0.0
    assert hint_lb[1] > bounds.lb.reshape(-1)[1]
    assert hint_ub[1] < bounds.ub.reshape(-1)[1]
    assert hint_lb[9] == bounds.lb.reshape(-1)[9]
    assert hint_ub[9] == bounds.ub.reshape(-1)[9]


def test_phase_selective_equal_savings_rejects_before_relu(monkeypatch):
    source = SparseHZono(
        c=np.array([-2.0, 2.0, 0.0]),
        Gc=sp.csr_matrix(
            [
                [0.25, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
        Gb=sp.csr_matrix([[0.0], [0.0], [0.2]]),
        Ac=sp.csr_matrix((0, 4)),
        Ab=sp.csr_matrix((0, 1)),
        b=np.zeros(0),
        frame_id=61,
        exact=True,
    )
    expr = SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(source, ()),),
        bias=np.array([0.25, 0.25, -0.125]),
        n_out=3,
        frame_id=61,
    )
    bounds = Bounds(
        lb=torch.tensor([[-3.0, 1.0, -2.0]], dtype=torch.float64),
        ub=torch.tensor([[-1.0, 3.0, 2.0]], dtype=torch.float64),
    )
    layer = SimpleNamespace(id=79, kind="RELU")
    tf = HybridzTF()
    tf._neural_hz_sparse_phase_selective_materialization = True
    tf._SPARSE_MAX_AFFINE_CELLS = 1000
    tf._sparse_frame_widths[61] = (4, 1)

    def unexpected_relu(*args, **kwargs):
        raise AssertionError("q == h must reject before exact ReLU")

    monkeypatch.setattr(hz_cnn, "_sparse_apply_relu", unexpected_relu)
    assert _try_phase_selective_exact_relu(expr, bounds, tf, layer) is None
    assert tf._sparse_relu_slots == {}
    assert tf._sparse_frame_widths[61] == (4, 1)
    assert tf._neural_hz_phase_selective_relus == 0


def test_phase_selective_materialization_is_default_off(monkeypatch):
    _, expr, bounds, layer, tf = _phase_selective_fixture()
    tf._neural_hz_sparse_phase_selective_materialization = False

    def unexpected_materialization(*args, **kwargs):
        raise AssertionError("default-off path must not materialize")

    monkeypatch.setattr(hz_cnn, "_lazy_materialize", unexpected_materialization)
    assert _try_phase_selective_exact_relu(expr, bounds, tf, layer) is None
    assert tf._neural_hz_phase_selective_relus == 0


def test_phase_selective_config_flag_is_default_off_and_mirrored():
    default_config = HybridZConfig()
    enabled_config = HybridZConfig(
        sparse_phase_selective_materialization=True
    )

    assert not default_config.sparse_phase_selective_materialization
    assert enabled_config.sparse_phase_selective_materialization
    assert not HybridzTF(
        default_config
    )._neural_hz_sparse_phase_selective_materialization
    assert HybridzTF(
        enabled_config
    )._neural_hz_sparse_phase_selective_materialization


def _phase_selective_interval_output(bounds):
    return Bounds(
        lb=torch.clamp(bounds.lb, min=0.0),
        ub=torch.clamp(bounds.ub, min=0.0),
    )


def test_phase_selective_deferred_tuple5_consumes_phase_bounds_and_cache():
    _, expr, input_bounds, layer, tf = _phase_selective_fixture()
    selective = _try_phase_selective_exact_relu(
        expr, input_bounds, tf, layer
    )
    assert selective is not None
    interval_output = _phase_selective_interval_output(input_bounds)
    result = Fact(bounds=interval_output, cons=ConSet())
    tf._sparse_precomputed_relu[layer.id] = (
        selective.core,
        input_bounds.lb.detach().cpu().clone(),
        input_bounds.ub.detach().cpu().clone(),
        selective.expression,
        selective.output_bounds,
    )

    # The masked core is identically zero on omitted P rows.  Treating it as
    # the whole ReLU output would conflict with their positive interval fact.
    with pytest.raises(ValueError, match="non-numerical conflict"):
        tf._sparse_fact(result, selective.core)
    propagated = tf._propagate_sparse_hz(layer, input_bounds, result)

    assert layer.id not in tf._sparse_precomputed_relu
    assert tf._sparse_affine_expr_cache[layer.id] is selective.expression
    assert tf._sparse_drop_reasons[layer.id] == "lazy_affine_expr"
    assert torch.equal(
        propagated.bounds.lb, selective.output_bounds.lb
    )
    assert torch.equal(
        propagated.bounds.ub, selective.output_bounds.ub
    )
    # Index 9 is P-Q and therefore retains the interval ReLU fact.
    assert propagated.bounds.lb.reshape(-1)[9] == 1.0
    assert propagated.bounds.ub.reshape(-1)[9] == 3.0


def test_phase_selective_direct_lazy_consumes_and_clears_bounds_cache():
    _, expr, input_bounds, layer, tf = _phase_selective_fixture()
    tf._net = SimpleNamespace(preds={layer.id: [layer.id - 1]})
    tf._sparse_affine_expr_cache[layer.id] = expr
    result = Fact(
        bounds=_phase_selective_interval_output(input_bounds),
        cons=ConSet(),
    )

    propagated = tf._propagate_sparse_hz(layer, input_bounds, result)

    assert layer.id not in tf._sparse_phase_output_bounds
    output_expression = tf._sparse_affine_expr_cache[layer.id]
    assert output_expression is not expr
    assert len(output_expression.terms) == 2
    assert tf._sparse_drop_reasons[layer.id] == "lazy_affine_expr"
    core = output_expression.terms[-1].source
    omitted_positive = 9
    assert core.c[omitted_positive] == 0.0
    assert core.Gc[omitted_positive].nnz == 0
    assert core.Gb[omitted_positive].nnz == 0
    # The phase hint, rather than sparse_fact(core), preserves P-Q precision.
    assert propagated.bounds.lb.reshape(-1)[omitted_positive] == 1.0
    assert propagated.bounds.ub.reshape(-1)[omitted_positive] == 3.0
    # Q is tightened by the probe and proves that the cache was actually used.
    assert propagated.bounds.lb.reshape(-1)[1] > 1.0
    assert propagated.bounds.ub.reshape(-1)[1] < 3.0


def test_transient_relu_slots_skip_only_preallocation_guards():
    source = SparseHZono(
        c=np.zeros(1),
        Gc=sp.eye(1, format="csr"),
        Gb=sp.csr_matrix((1, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=47,
        exact=True,
    )
    tf = HybridzTF(HybridZConfig(sparse_relu_nnz_guard=True))
    tf._SPARSE_MAX_AFFINE_CELLS = 1
    tf._sparse_frame_widths[47] = (1, 0)

    assert tf._sparse_relu_slots_for(source, 9, [0]) is None
    tf._neural_hz_transient_relu_input = True
    assert tf._sparse_relu_slots_for(source, 9, [0]) is not None
    tf._neural_hz_transient_relu_input = False
    assert tf._sparse_relu_slots_for(source, 10, [0]) is None


def _continuous_extreme(model, xi: float, z: float, eta_cost: float):
    upper_rows = np.isfinite(model.row_ub)
    lower_rows = np.isfinite(model.row_lb)
    A_ub = np.vstack(
        [model.A[upper_rows].toarray(), -model.A[lower_rows].toarray()]
    )
    b_ub = np.concatenate(
        [model.row_ub[upper_rows], -model.row_lb[lower_rows]]
    )
    objective = np.zeros(model.n_var)
    objective[1] = eta_cost
    return linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(xi, xi), (-1.0, 1.0), (z, z)],
        method="highs",
    )


def test_compact_relu_removes_one_continuous_factor_per_unstable_neuron():
    source = _interval_hz(-2.0, 3.0)
    extended = hz_apply_relu(source)
    compact = hz_apply_relu_compact_exact(source)

    assert extended.Gc.shape[1] == 3
    assert compact.Gc.shape[1] == 2
    assert extended.Gb.shape[1] == compact.Gb.shape[1] == 1
    assert extended.Ac.shape[0] == compact.Ac.shape[0] == 3
    assert compact.eq_mask.tolist() == [False, False, False]
    assert compact.col_ids[0].item() == 7
    assert compact.col_ids.numel() == 2
    assert compact.bcol_ids.numel() == 1


def test_compact_relu_graph_has_exact_pointwise_semantics():
    lower, upper = -2.0, 3.0
    center = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    model = _lower_hz_milp(
        hz_apply_relu_compact_exact(_interval_hz(lower, upper)),
        prune_unused=False,
        coalesce_rows=False,
    )

    assert model.n_cont == 2
    assert model.n_bin == 1
    for xi in np.linspace(-1.0, 1.0, 17):
        x_value = center + radius * xi
        expected = max(x_value, 0.0)
        feasible_outputs = []
        for z in (0.0, 1.0):
            minimum = _continuous_extreme(model, xi, z, 1.0)
            maximum = _continuous_extreme(model, xi, z, -1.0)
            assert minimum.success == maximum.success
            if not minimum.success:
                continue
            for result in (minimum, maximum):
                output = float(
                    model.value_center[0]
                    + np.asarray(model.value_matrix @ result.x).reshape(-1)[0]
                )
                feasible_outputs.append(output)
                assert output == pytest.approx(expected, abs=1e-9)
        assert feasible_outputs


@pytest.mark.parametrize(
    ("lower", "upper", "expected_center", "expected_generator"),
    [(1.0, 3.0, 2.0, 1.0), (-3.0, -1.0, 0.0, 0.0)],
)
def test_compact_relu_preserves_stable_cases(
    lower, upper, expected_center, expected_generator
):
    result = hz_apply_relu_compact_exact(_interval_hz(lower, upper))

    assert result.Gc.shape[1] == 1
    assert result.Gb.shape[1] == 0
    assert result.Ac.shape[0] == 0
    assert result.c.item() == pytest.approx(expected_center)
    assert result.Gc.item() == pytest.approx(expected_generator)


def test_fill_aware_relu_uses_quotient_only_without_predicate_fill():
    sparse_source = _interval_hz(-2.0, 3.0)
    sparse_result = hz_apply_relu_fill_aware_exact(sparse_source)
    assert sparse_result.Gc.shape[1] == 2

    dense_source = HZono(
        c=torch.zeros((1, 1), dtype=torch.float64),
        Gc=torch.ones((1, 3), dtype=torch.float64),
        Gb=torch.zeros((1, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 3), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )
    dense_result = hz_apply_relu_fill_aware_exact(dense_source)
    assert dense_result.Gc.shape[1] == 5
    assert dense_result.eq_mask.tolist() == [True, False, False]

    equal_fill_small = HZono(
        c=torch.zeros((1, 1), dtype=torch.float64),
        Gc=torch.ones((1, 2), dtype=torch.float64),
        Gb=torch.zeros((1, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 2), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )
    equal_fill_small_result = hz_apply_relu_fill_aware_exact(equal_fill_small)
    assert equal_fill_small_result.Gc.shape[1] == 3
    assert equal_fill_small_result.eq_mask.tolist() == [False, False, False]

    wide_generators = torch.zeros((1, 511), dtype=torch.float64)
    wide_generators[0, :2] = 1.0
    equal_fill_large = HZono(
        c=torch.zeros((1, 1), dtype=torch.float64),
        Gc=wide_generators,
        Gb=torch.zeros((1, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 511), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )
    equal_fill_large_result = hz_apply_relu_fill_aware_exact(equal_fill_large)
    assert equal_fill_large_result.Gc.shape[1] == 513
    assert equal_fill_large_result.eq_mask.tolist() == [True, False, False]


def test_sparse_compact_relu_matches_dense_quotient_exactly():
    lower, upper = -2.0, 3.0
    sparse_source = SparseHZono(
        c=np.array([(lower + upper) / 2.0]),
        Gc=sp.csr_matrix([[(upper - lower) / 2.0]]),
        Gb=sp.csr_matrix((1, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=9,
        exact=True,
    )
    sparse_result = sparse_hz_apply_relu_compact_exact(
        sparse_source,
        np.array([lower]),
        np.array([upper]),
        [(1, 1, 0)],
        n_cont=2,
        n_bin=1,
    )
    dense_result = hz_apply_relu_compact_exact(_interval_hz(lower, upper))
    sparse_model = _lower_hz_milp(
        sparse_result, prune_unused=False, coalesce_rows=False
    )
    dense_model = _lower_hz_milp(
        dense_result, prune_unused=False, coalesce_rows=False
    )

    np.testing.assert_allclose(sparse_model.value_center, dense_model.value_center)
    np.testing.assert_allclose(
        sparse_model.value_matrix.toarray(), dense_model.value_matrix.toarray()
    )
    np.testing.assert_allclose(sparse_model.A.toarray(), dense_model.A.toarray())
    np.testing.assert_allclose(sparse_model.row_lb, dense_model.row_lb)
    np.testing.assert_allclose(sparse_model.row_ub, dense_model.row_ub)


def test_sparse_shared_relu_uses_one_exact_graph_for_duplicate_affine_rows():
    lower, upper = -2.0, 3.0
    center = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    source = SparseHZono(
        c=np.array([center, center]),
        Gc=sp.csr_matrix([[radius], [radius]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=10,
        exact=True,
    )
    groups = _sparse_exact_duplicate_groups(source, np.array([0, 1]))
    assert [group.tolist() for group in groups] == [[0, 1]]
    shared = sparse_hz_apply_relu_shared_exact(
        source,
        np.array([lower, -1.5]),
        np.array([upper, 2.5]),
        groups,
        [(1, 2, 0)],
        n_cont=3,
        n_bin=1,
    )
    model = _lower_hz_milp(shared, prune_unused=False, coalesce_rows=False)

    assert model.n_cont == 3
    assert model.n_bin == 1
    assert model.A.shape[0] == 3
    np.testing.assert_allclose(model.value_center[0], model.value_center[1])
    np.testing.assert_allclose(
        model.value_matrix.toarray()[0], model.value_matrix.toarray()[1]
    )

    upper_rows = np.isfinite(model.row_ub)
    lower_rows = np.isfinite(model.row_lb)
    A_ub = np.vstack(
        [model.A[upper_rows].toarray(), -model.A[lower_rows].toarray()]
    )
    b_ub = np.concatenate(
        [model.row_ub[upper_rows], -model.row_lb[lower_rows]]
    )
    output_row = model.value_matrix.toarray()[0]
    for xi in np.linspace(-1.0, 1.0, 13):
        expected = max(center + radius * xi, 0.0)
        outputs = []
        for z in (0.0, 1.0):
            bounds = [(xi, xi), (-1.0, 1.0), (-1.0, 1.0), (z, z)]
            for direction in (1.0, -1.0):
                result = linprog(
                    direction * output_row,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method="highs",
                )
                if result.success:
                    output = float(model.value_center[0] + output_row @ result.x)
                    outputs.append(output)
                    assert output == pytest.approx(expected, abs=1e-9)
        assert outputs


def test_sparse_duplicate_groups_never_merge_near_equal_rows():
    one_up = np.nextafter(1.0, np.inf)
    source = SparseHZono(
        c=np.array([0.0, 0.0, np.nextafter(0.0, 1.0)]),
        Gc=sp.csr_matrix([[1.0], [one_up], [1.0]]),
        Gb=sp.csr_matrix((3, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=11,
        exact=True,
    )

    groups = _sparse_exact_duplicate_groups(source, np.array([0, 1, 2]))

    assert [group.tolist() for group in groups] == [[0], [1], [2]]


def test_sparse_positive_proportional_groups_use_exact_real_relation():
    six_up = np.nextafter(6.0, np.inf)
    source = SparseHZono(
        c=np.array([1.0, 2.0, -2.0, 2.0]),
        Gc=sp.csr_matrix([[3.0], [6.0], [-6.0], [six_up]]),
        Gb=sp.csr_matrix((4, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=12,
        exact=True,
    )

    groups = _sparse_exact_positive_proportional_groups(
        source, np.array([0, 1, 2, 3])
    )

    assert [group.tolist() for group in groups] == [[0, 1], [2], [3]]

    signed_groups = _sparse_exact_signed_proportional_groups(
        source, np.array([0, 1, 2, 3])
    )
    assert [group.tolist() for group in signed_groups] == [[0, 1, 2], [3]]

    signed_duplicate_groups = _sparse_exact_signed_duplicate_groups(
        source, np.array([0, 1, 2, 3])
    )
    assert [group.tolist() for group in signed_duplicate_groups] == [
        [0],
        [1, 2],
        [3],
    ]


def test_sparse_signed_shared_relu_is_exact_for_x_and_negative_x():
    center, radius = 0.5, 2.5
    source = SparseHZono(
        c=np.array([center, -center]),
        Gc=sp.csr_matrix([[radius], [-radius]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=13,
        exact=True,
    )
    groups = _sparse_exact_signed_duplicate_groups(source, np.array([0, 1]))
    assert [group.tolist() for group in groups] == [[0, 1]]
    shared = sparse_hz_apply_relu_signed_shared_exact(
        source,
        np.array([-2.0, -3.0]),
        np.array([3.0, 2.0]),
        groups,
        [(1, 2, 0)],
        n_cont=3,
        n_bin=1,
    )
    model = _lower_hz_milp(shared, prune_unused=False, coalesce_rows=False)

    assert model.n_cont == 3
    assert model.n_bin == 1
    assert model.A.shape[0] == 3
    upper_rows = np.isfinite(model.row_ub)
    lower_rows = np.isfinite(model.row_lb)
    A_ub = np.vstack(
        [model.A[upper_rows].toarray(), -model.A[lower_rows].toarray()]
    )
    b_ub = np.concatenate(
        [model.row_ub[upper_rows], -model.row_lb[lower_rows]]
    )
    output_rows = model.value_matrix.toarray()
    for xi in np.linspace(-1.0, 1.0, 13):
        x_value = center + radius * xi
        expected = (max(x_value, 0.0), max(-x_value, 0.0))
        outputs = [[], []]
        for z in (0.0, 1.0):
            bounds = [(xi, xi), (-1.0, 1.0), (-1.0, 1.0), (z, z)]
            for output_index in (0, 1):
                for direction in (1.0, -1.0):
                    result = linprog(
                        direction * output_rows[output_index],
                        A_ub=A_ub,
                        b_ub=b_ub,
                        bounds=bounds,
                        method="highs",
                    )
                    if result.success:
                        value = float(
                            model.value_center[output_index]
                            + output_rows[output_index] @ result.x
                        )
                        outputs[output_index].append(value)
                        assert value == pytest.approx(
                            expected[output_index], abs=1e-9
                        )
        assert outputs[0]
        assert outputs[1]


def test_sparse_compact_signed_shared_relu_is_exact_for_x_and_negative_x():
    center, radius = 0.5, 2.5
    source = SparseHZono(
        c=np.array([center, -center]),
        Gc=sp.csr_matrix([[radius], [-radius]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=14,
        exact=True,
    )
    groups = _sparse_exact_signed_duplicate_groups(source, np.array([0, 1]))
    shared = sparse_hz_apply_relu_signed_shared_compact_exact(
        source,
        np.array([-2.0, -3.0]),
        np.array([3.0, 2.0]),
        groups,
        [(1, 1, 0)],
        n_cont=2,
        n_bin=1,
    )
    model = _lower_hz_milp(shared, prune_unused=False, coalesce_rows=False)

    assert model.n_cont == 2
    assert model.n_bin == 1
    assert model.A.shape[0] == 3
    upper_rows = np.isfinite(model.row_ub)
    lower_rows = np.isfinite(model.row_lb)
    A_ub = np.vstack(
        [model.A[upper_rows].toarray(), -model.A[lower_rows].toarray()]
    )
    b_ub = np.concatenate(
        [model.row_ub[upper_rows], -model.row_lb[lower_rows]]
    )
    output_rows = model.value_matrix.toarray()
    for xi in np.linspace(-1.0, 1.0, 13):
        x_value = center + radius * xi
        expected = (max(x_value, 0.0), max(-x_value, 0.0))
        outputs = [[], []]
        for z in (0.0, 1.0):
            bounds = [(xi, xi), (-1.0, 1.0), (z, z)]
            for output_index in (0, 1):
                for direction in (1.0, -1.0):
                    result = linprog(
                        direction * output_rows[output_index],
                        A_ub=A_ub,
                        b_ub=b_ub,
                        bounds=bounds,
                        method="highs",
                    )
                    if result.success:
                        value = float(
                            model.value_center[output_index]
                            + output_rows[output_index] @ result.x
                        )
                        outputs[output_index].append(value)
                        assert value == pytest.approx(
                            expected[output_index], abs=1e-9
                        )
        assert outputs[0]
        assert outputs[1]


def test_signed_relu_sharing_is_explicit_opt_in_and_selects_one_graph_form():
    baseline = HybridzTF()
    extended = HybridzTF(
        HybridZConfig(signed_relu_sharing=True, signed_relu_compact=False)
    )
    compact = HybridzTF(
        HybridZConfig(signed_relu_sharing=True, signed_relu_compact=True)
    )
    cancellation_without_sharing = HybridzTF(
        HybridZConfig(signed_relu_cancellation=True)
    )
    cancellation = HybridzTF(
        HybridZConfig(
            signed_relu_sharing=True,
            signed_relu_compact=True,
            signed_relu_cancellation=True,
        )
    )
    mixed_cancellation = HybridzTF(
        HybridZConfig(
            signed_relu_sharing=True,
            signed_relu_compact=True,
            signed_relu_cancellation=True,
            signed_relu_cancellation_mixed_only=True,
        )
    )
    pair_cancellation = HybridzTF(
        HybridZConfig(
            signed_relu_sharing=True,
            signed_relu_compact=True,
            signed_relu_cancellation=True,
            signed_relu_cancellation_pairs_only=True,
        )
    )
    bounded_cancellation = HybridzTF(
        HybridZConfig(
            signed_relu_sharing=True,
            signed_relu_compact=True,
            signed_relu_cancellation=True,
            signed_relu_cancellation_elimination_max_cardinality=4,
        )
    )
    adaptive_cancellation = HybridzTF(
        HybridZConfig(
            signed_relu_sharing=True,
            signed_relu_compact=True,
            signed_relu_cancellation=True,
            signed_relu_cancellation_two_pair_min_outputs=8192,
        )
    )
    sparse_affine = HybridzTF(HybridZConfig(sparse_affine_nnz_guard=True))
    sparse_relu = HybridzTF(HybridZConfig(sparse_relu_nnz_guard=True))
    sparse_conv_csr = HybridzTF(HybridZConfig(sparse_conv_csr_builder=True))
    sparse_deferred = HybridzTF(
        HybridZConfig(sparse_deferred_relu_materialization=True)
    )
    sparse_lazy = HybridzTF(HybridZConfig(sparse_lazy_affine_dag=True))
    sparse_frontier_rebase = HybridzTF(
        HybridZConfig(sparse_frontier_image_rebase=True)
    )

    assert not baseline._neural_hz_share_signed_relu
    assert not baseline._neural_hz_share_signed_compact_relu
    assert not baseline._neural_hz_signed_cancellation
    assert extended._neural_hz_share_signed_relu
    assert not extended._neural_hz_share_signed_compact_relu
    assert not extended._neural_hz_signed_cancellation
    assert not compact._neural_hz_share_signed_relu
    assert compact._neural_hz_share_signed_compact_relu
    assert not compact._neural_hz_signed_cancellation
    assert not cancellation_without_sharing._neural_hz_signed_cancellation
    assert cancellation._neural_hz_share_signed_compact_relu
    assert cancellation._neural_hz_signed_cancellation
    assert not cancellation._neural_hz_signed_cancellation_mixed_only
    assert mixed_cancellation._neural_hz_signed_cancellation
    assert mixed_cancellation._neural_hz_signed_cancellation_mixed_only
    assert pair_cancellation._neural_hz_signed_cancellation
    assert pair_cancellation._neural_hz_signed_cancellation_pairs_only
    assert bounded_cancellation._neural_hz_signed_cancellation
    assert (
        bounded_cancellation
        ._neural_hz_signed_cancellation_elimination_max_cardinality
        == 4
    )
    assert (
        adaptive_cancellation
        ._neural_hz_signed_cancellation_two_pair_min_outputs
        == 8192
    )
    assert not baseline._neural_hz_sparse_affine_nnz_guard
    assert sparse_affine._neural_hz_sparse_affine_nnz_guard
    assert not baseline._neural_hz_sparse_relu_nnz_guard
    assert sparse_relu._neural_hz_sparse_relu_nnz_guard
    assert not baseline._neural_hz_sparse_conv_csr_builder
    assert sparse_conv_csr._neural_hz_sparse_conv_csr_builder
    assert not baseline._neural_hz_sparse_deferred_relu_materialization
    assert sparse_deferred._neural_hz_sparse_deferred_relu_materialization
    assert not baseline._neural_hz_sparse_lazy_affine_dag
    assert sparse_lazy._neural_hz_sparse_lazy_affine_dag
    assert not baseline._neural_hz_sparse_frontier_image_rebase
    assert sparse_frontier_rebase._neural_hz_sparse_frontier_image_rebase

    with pytest.raises(ValueError, match="must be nonnegative"):
        HybridzTF(
            HybridZConfig(
                signed_relu_sharing=True,
                signed_relu_compact=True,
                signed_relu_cancellation=True,
                signed_relu_cancellation_elimination_max_cardinality=-1,
            )
        )

    with pytest.raises(ValueError, match="must be nonnegative"):
        HybridzTF(
            HybridZConfig(
                signed_relu_sharing=True,
                signed_relu_compact=True,
                signed_relu_cancellation=True,
                signed_relu_cancellation_two_pair_min_outputs=-1,
            )
        )


def test_compact_signed_mode_keeps_singleton_only_layer_in_extended_form():
    source = SparseHZono(
        c=np.zeros(2),
        Gc=sp.eye(2, format="csr"),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 2)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=15,
        exact=True,
    )
    tf = HybridzTF(HybridZConfig(signed_relu_sharing=True))
    tf._sparse_frame_widths[15] = (2, 0)

    result, reason = _sparse_apply_relu(
        SimpleNamespace(id=100),
        source,
        Bounds(
            torch.full((1, 2), -1.0, dtype=torch.float64),
            torch.full((1, 2), 1.0, dtype=torch.float64),
        ),
        tf,
    )

    assert reason is None
    assert result.n_cont == 6
    assert result.n_bin == 2
    assert getattr(tf, "_neural_hz_signed_compact_layers", 0) == 0


def test_exact_dyadic_zero_sum_rejects_one_ulp_near_cancellation():
    value = np.float64(0.1)

    assert _exact_dyadic_sum_is_zero([value, -value])
    assert not _exact_dyadic_sum_is_zero(
        [value, -np.nextafter(value, np.inf)]
    )


def test_dense_cancellation_drops_only_local_compact_signed_graph():
    source = SparseHZono(
        c=np.array([0.5, -0.5]),
        Gc=sp.csr_matrix([[2.5], [-2.5]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=16,
        exact=True,
    )
    groups = _sparse_exact_signed_duplicate_groups(source, np.array([0, 1]))
    relu = sparse_hz_apply_relu_signed_shared_compact_exact(
        source,
        np.array([-2.0, -3.0]),
        np.array([3.0, 2.0]),
        groups,
        [(1, 1, 0)],
        n_cont=2,
        n_bin=1,
    )
    operator = sp.csr_matrix([[1.0, -1.0]])
    dense = SparseHZono(
        c=np.asarray(operator @ relu.c).reshape(-1),
        Gc=(operator @ relu.Gc).tocsr(),
        Gb=(operator @ relu.Gb).tocsr(),
        Ac=relu.Ac.copy(),
        Ab=relu.Ab.copy(),
        b=relu.b.copy(),
        Auc=relu.Auc.copy(),
        Aub=relu.Aub.copy(),
        ub=relu.ub.copy(),
        frame_id=relu.frame_id,
        exact=True,
    )

    reduced, eliminated = _sparse_drop_canceled_relu_graphs(
        dense,
        {
            "frame_id": 16,
            "base_ineq": 0,
            "n_groups": 1,
            "group_indices": [0],
            "eta_cols": [1],
            "z_cols": [0],
        },
    )

    assert eliminated == 1
    assert reduced.n_ineq == 0
    assert reduced.Gc[:, 1].nnz == 0
    assert reduced.Gb[:, 0].nnz == 0
    np.testing.assert_allclose(reduced.c, [0.5])
    np.testing.assert_allclose(reduced.Gc[:, :1].toarray(), [[2.5]])


def test_dense_cancellation_fails_closed_on_retained_graph_coupling():
    source = SparseHZono(
        c=np.array([0.5, -0.5]),
        Gc=sp.csr_matrix([[2.5], [-2.5]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=17,
        exact=True,
    )
    groups = _sparse_exact_signed_duplicate_groups(source, np.array([0, 1]))
    relu = sparse_hz_apply_relu_signed_shared_compact_exact(
        source,
        np.array([-2.0, -3.0]),
        np.array([3.0, 2.0]),
        groups,
        [(1, 1, 0)],
        n_cont=2,
        n_bin=1,
    )
    operator = sp.csr_matrix([[1.0, -1.0]])
    dense = SparseHZono(
        c=np.asarray(operator @ relu.c).reshape(-1),
        Gc=(operator @ relu.Gc).tocsr(),
        Gb=(operator @ relu.Gb).tocsr(),
        Ac=relu.Ac.copy(),
        Ab=relu.Ab.copy(),
        b=relu.b.copy(),
        Auc=sp.vstack(
            [
                relu.Auc,
                sp.csr_matrix(([1.0], ([0], [1])), shape=(1, relu.n_cont)),
            ],
            format="csr",
        ),
        Aub=sp.vstack(
            [relu.Aub, sp.csr_matrix((1, relu.n_bin))], format="csr"
        ),
        ub=np.concatenate([relu.ub, [1.0]]),
        frame_id=relu.frame_id,
        exact=True,
    )

    retained, eliminated = _sparse_drop_canceled_relu_graphs(
        dense,
        {
            "frame_id": 17,
            "base_ineq": 0,
            "n_groups": 1,
            "group_indices": [0],
            "eta_cols": [1],
            "z_cols": [0],
        },
    )

    assert eliminated == 0
    assert retained is dense


def test_sparse_image_rebase_is_exact_and_preserves_nonconvex_predicates():
    source = SparseHZono(
        c=np.array([0.2, -0.1, 0.0]),
        Gc=sp.csr_matrix([[0.5, 0.0], [0.0, 0.25], [0.0, 0.0]]),
        Gb=sp.csr_matrix([[0.1], [0.0], [0.0]]),
        Ac=sp.csr_matrix([[1.0, -1.0]]),
        Ab=sp.csr_matrix([[0.0]]),
        b=np.array([0.0]),
        Auc=sp.csr_matrix([[1.0, 0.0]]),
        Aub=sp.csr_matrix([[1.0]]),
        ub=np.array([2.0]),
        frame_id=71,
        exact=True,
    )
    bounds = Bounds(
        torch.tensor([[-1.0, -1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64),
    )

    rebased = sparse_hz_rebase_image_exact(source, bounds)

    assert rebased.exact
    assert rebased.frame_id == source.frame_id
    assert rebased.n_bin == source.n_bin == 1
    assert rebased.n_cont == source.n_cont + 2
    assert rebased.Gc.nnz == 2
    assert rebased.Gb.nnz == 0
    np.testing.assert_array_equal(
        rebased.Ac[: source.n_eq, : source.n_cont].toarray(),
        source.Ac.toarray(),
    )
    np.testing.assert_array_equal(
        rebased.Ab[: source.n_eq].toarray(), source.Ab.toarray()
    )
    np.testing.assert_array_equal(rebased.Aub.toarray(), source.Aub.toarray())

    # Two feasible old assignments, including both binary phases, lift to the
    # new interface by eta=(y-m)/r and reproduce exactly the same image.
    for xi, beta in ((np.array([0.25, 0.25]), 1.0), (np.array([-0.5, -0.5]), -1.0)):
        old_value = (
            source.c
            + np.asarray(source.Gc @ xi).reshape(-1)
            + np.asarray(source.Gb @ np.array([beta])).reshape(-1)
        )
        eta = old_value[:2]
        lifted_cont = np.concatenate([xi, eta])
        np.testing.assert_allclose(
            rebased.Ac @ lifted_cont + rebased.Ab @ np.array([beta]),
            rebased.b,
        )
        assert np.all(rebased.Auc @ lifted_cont + rebased.Aub @ np.array([beta]) <= rebased.ub)
        new_value = (
            rebased.c
            + np.asarray(rebased.Gc @ lifted_cont).reshape(-1)
            + np.asarray(rebased.Gb @ np.array([beta])).reshape(-1)
        )
        np.testing.assert_allclose(new_value, old_value)


def test_sparse_image_rebase_rejects_nonfinite_or_inexact_inputs():
    source = SparseHZono(
        c=np.array([0.0]),
        Gc=sp.csr_matrix([[1.0]]),
        Gb=sp.csr_matrix((1, 0)),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=72,
        exact=False,
    )
    finite = Bounds(
        torch.tensor([[-1.0]], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="exact sparse HZ"):
        sparse_hz_rebase_image_exact(source, finite)

    source.exact = True
    nonfinite = Bounds(
        torch.tensor([[-1.0]], dtype=torch.float64),
        torch.tensor([[float("inf")]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="finite forward bounds"):
        sparse_hz_rebase_image_exact(source, nonfinite)


def test_sparse_image_rebase_merges_exactly_with_an_older_shared_frame_skip():
    branch = SparseHZono(
        c=np.array([0.0]),
        Gc=sp.csr_matrix([[0.5, -0.25]]),
        Gb=sp.csr_matrix([[0.2]]),
        Ac=sp.csr_matrix((0, 2)),
        Ab=sp.csr_matrix((0, 1)),
        b=np.zeros(0),
        frame_id=73,
        exact=True,
    )
    skip = SparseHZono(
        c=np.array([0.1]),
        Gc=sp.csr_matrix([[0.3, 0.4]]),
        Gb=sp.csr_matrix([[-0.1]]),
        Ac=sp.csr_matrix((0, 2)),
        Ab=sp.csr_matrix((0, 1)),
        b=np.zeros(0),
        frame_id=73,
        exact=True,
    )
    bounds = Bounds(
        torch.tensor([[-1.0]], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    rebased = sparse_hz_rebase_image_exact(branch, bounds)
    merged = sparse_hz_add_same_frame(rebased, skip)

    assert merged.frame_id == 73
    assert merged.n_cont == 3
    assert merged.n_bin == 1
    for xi, beta in ((np.array([0.2, -0.4]), 1.0), (np.array([-0.3, 0.1]), -1.0)):
        branch_value = (
            branch.c + branch.Gc @ xi + branch.Gb @ np.array([beta])
        ).item()
        eta = branch_value
        continuous = np.array([xi[0], xi[1], eta])
        np.testing.assert_allclose(
            merged.Ac @ continuous + merged.Ab @ np.array([beta]),
            merged.b,
        )
        merged_value = (
            merged.c
            + merged.Gc @ continuous
            + merged.Gb @ np.array([beta])
        ).item()
        skip_value = (
            skip.c + skip.Gc @ xi + skip.Gb @ np.array([beta])
        ).item()
        np.testing.assert_allclose(merged_value, branch_value + skip_value)


def test_sparse_affine_images_share_immutable_predicate_blocks_by_identity():
    source = SparseHZono(
        c=np.array([0.0, 0.0]),
        Gc=sp.eye(2, format="csr"),
        Gb=sp.csr_matrix((2, 1)),
        Ac=sp.csr_matrix([[1.0, -1.0]]),
        Ab=sp.csr_matrix([[1.0]]),
        b=np.array([0.5]),
        Auc=sp.csr_matrix([[0.0, 1.0]]),
        Aub=sp.csr_matrix([[-1.0]]),
        ub=np.array([0.25]),
        frame_id=74,
        exact=True,
    )
    image = sparse_hz_linear(source, sp.eye(2, format="csr"))
    merged = sparse_hz_add_same_frame(image, image)

    for name in ("Ac", "Ab", "Auc", "Aub"):
        assert getattr(image, name) is getattr(source, name)
        assert getattr(merged, name) is getattr(source, name)
    assert np.shares_memory(image.b, source.b)
    assert np.shares_memory(merged.b, source.b)
    assert np.shares_memory(image.ub, source.ub)
    assert np.shares_memory(merged.ub, source.ub)
    np.testing.assert_array_equal(merged.Ac.toarray(), source.Ac.toarray())
    np.testing.assert_array_equal(merged.Ab.toarray(), source.Ab.toarray())


def test_sparse_hz_constructor_still_removes_explicit_zero_entries():
    with_zero = sp.csr_matrix(
        (
            np.array([1.0, 0.0]),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        ),
        shape=(1, 2),
    )
    source = SparseHZono(
        c=np.array([0.0]),
        Gc=with_zero,
        Gb=sp.csr_matrix((1, 0)),
        Ac=sp.csr_matrix((0, 2)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=75,
        exact=True,
    )

    assert source.Gc.nnz == 1
    np.testing.assert_array_equal(source.Gc.toarray(), [[1.0, 0.0]])


@pytest.mark.parametrize(("prior_terms", "expected"), [(1, 2), (3, 4), (4, 5), (5, 6)])
def test_residual_rebase_pressure_counts_sources_through_future_affine_chain(
    prior_terms,
    expected,
):
    """The trigger is structural and can see through an unexecuted skip arm."""
    kinds = {
        0: "RELU",
        1: "CONV2D",
        2: "BIAS",
        3: "CONV2D",
        4: "BIAS",
        5: "ADD",
        9: "ADD",
    }
    net = SimpleNamespace(
        by_id={
            lid: SimpleNamespace(id=lid, kind=kind)
            for lid, kind in kinds.items()
        },
        preds={1: [0], 2: [1], 3: [9], 4: [3], 5: [2, 4]},
        succs={0: [1], 1: [2], 2: [5], 9: [3], 3: [4], 4: [5]},
    )
    tf = HybridzTF()
    tf._net = net
    tf._sparse_affine_expr_cache[9] = SimpleNamespace(
        terms=tuple(range(prior_terms))
    )

    assert tf._relu_next_residual_terms(0) == expected


def test_residual_rebase_pressure_fails_closed_for_two_nearest_adds():
    net = SimpleNamespace(
        by_id={
            0: SimpleNamespace(id=0, kind="RELU"),
            1: SimpleNamespace(id=1, kind="BIAS"),
            2: SimpleNamespace(id=2, kind="BIAS"),
            3: SimpleNamespace(id=3, kind="ADD"),
            4: SimpleNamespace(id=4, kind="ADD"),
        },
        preds={1: [0], 2: [0], 3: [1, 8], 4: [2, 9]},
        succs={0: [1, 2], 1: [3], 2: [4]},
    )
    tf = HybridzTF()
    tf._net = net

    assert tf._relu_next_residual_terms(0) == 0
