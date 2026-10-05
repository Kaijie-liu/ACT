"""Focused tests for the isolated S0-C1 composed-stencil prototype."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.composed_stencil_prototype import (
    ComposedStencilPrototype,
    FrozenV1Limits,
    GateContext,
    PrototypeReject,
)


def _dyadic_kernel(out_channels, in_per_group, kernel_shape, offset=0):
    size = out_channels * in_per_group * kernel_shape[0] * kernel_shape[1]
    values = ((np.arange(size, dtype=np.int64) + offset) % 9 - 4) / 16.0
    return values.reshape(
        out_channels, in_per_group, kernel_shape[0], kernel_shape[1]
    )


def _stationary_full(scale, shape):
    scale = np.asarray(scale, dtype=np.float64)
    return np.broadcast_to(
        scale.reshape(1, shape[1], 1, 1), shape
    ).copy().reshape(-1)


def _gate(outer, *, selected_rows=None, limits=None, **kwargs):
    if selected_rows is None:
        selected_rows = np.arange(outer.shape[0], dtype=np.int64)
    return GateContext(
        selected_rows=selected_rows,
        reachable_before_bytes=kwargs.pop("reachable_before_bytes", 10**9),
        reachable_after_other_bytes=kwargs.pop(
            "reachable_after_other_bytes", 0
        ),
        limits=FrozenV1Limits() if limits is None else limits,
        **kwargs,
    )


def _canonical(matrix):
    result = matrix.tocsr().astype(np.float64, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    return result


def _assert_csr_dyadic_equal(candidate, reference):
    candidate = _canonical(candidate)
    reference = _canonical(reference)
    assert candidate.shape == reference.shape
    assert np.array_equal(candidate.indptr, reference.indptr)
    assert np.array_equal(candidate.indices, reference.indices)
    assert np.array_equal(candidate.data, reference.data)


def _reference(inner, scales, outer):
    middle = np.ones(inner.shape[0], dtype=np.float64)
    for scale in scales:
        raw = np.asarray(scale, dtype=np.float64).reshape(-1)
        if raw.size == inner.output_shape[1]:
            raw = _stationary_full(raw, inner.output_shape)
        middle *= raw
    result = (
        outer.to_csr_reference()
        @ sp.diags(middle, format="csr")
        @ inner.to_csr_reference()
    )
    return _canonical(result)


def _build(inner, scales, outer, *, gate=None):
    decision = ComposedStencilPrototype.try_build(
        inner,
        tuple(scales),
        outer,
        _gate(outer) if gate is None else gate,
    )
    assert decision.triggered, (decision.reason, decision.estimate)
    assert decision.operator is not None
    return decision.operator, decision.estimate


@pytest.mark.parametrize(
    "case",
    [
        "ordinary_odd_even",
        "asymmetric_geometry",
        "grouped",
        "depthwise_multiplier",
        "batch_masked",
    ],
)
def test_composed_stencil_matches_small_reference_csr_exactly(case):
    if case == "ordinary_odd_even":
        inner = ImplicitConv2DOp(
            _dyadic_kernel(8, 2, (3, 3)),
            (1, 2, 7, 8),
            padding=(1, 1),
        )
        scale = np.array([1, -0.5, 0, 0.25, -1, 2, 0.5, -0.25])
        outer = ImplicitConv2DOp(
            _dyadic_kernel(3, 8, (3, 3), offset=2),
            inner.output_shape,
            padding=(1, 1),
        )
    elif case == "asymmetric_geometry":
        inner = ImplicitConv2DOp(
            _dyadic_kernel(8, 2, (3, 2)),
            (1, 2, 9, 10),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(1, 2),
        )
        scale = np.array([1, -0.5, 0.25, -0.25, 2, 0, 0.5, -1])
        outer = ImplicitConv2DOp(
            _dyadic_kernel(4, 8, (2, 3), offset=3),
            inner.output_shape,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
        )
    elif case == "grouped":
        inner = ImplicitConv2DOp(
            _dyadic_kernel(16, 2, (3, 3)),
            (1, 4, 7, 7),
            padding=1,
            groups=2,
        )
        scale = np.array(
            [1, -1, 0.5, -0.5, 0.25, -0.25, 2, -2] * 2,
            dtype=np.float64,
        )
        outer = ImplicitConv2DOp(
            _dyadic_kernel(8, 8, (3, 3), offset=1),
            inner.output_shape,
            padding=1,
            groups=2,
        )
    elif case == "depthwise_multiplier":
        inner = ImplicitConv2DOp(
            _dyadic_kernel(16, 1, (3, 3)),
            (1, 2, 7, 7),
            padding=1,
            groups=2,
        )
        scale = np.array(
            [1, -1, 0.5, -0.5, 0.25, -0.25, 2, -2] * 2,
            dtype=np.float64,
        )
        outer = ImplicitConv2DOp(
            _dyadic_kernel(4, 16, (3, 3), offset=4),
            inner.output_shape,
            padding=1,
        )
    else:
        inner_mask = np.ones((2, 8, 6, 7), dtype=bool)
        inner_mask[:, 2] = False
        inner_mask[:, 6] = False
        inner = ImplicitConv2DOp(
            _dyadic_kernel(8, 2, (3, 3)),
            (2, 2, 6, 7),
            padding=1,
            row_mask=inner_mask,
        )
        scale = _stationary_full(
            np.array([1, -0.5, 0, 0.25, -1, 2, 0.5, -0.25]),
            inner.output_shape,
        )
        provisional = ImplicitConv2DOp(
            _dyadic_kernel(3, 8, (3, 3), offset=2),
            inner.output_shape,
            padding=1,
        )
        outer_mask = np.ones(provisional.shape[0], dtype=bool)
        outer_mask[::5] = False
        outer_mask[7::11] = False
        outer = ImplicitConv2DOp(
            _dyadic_kernel(3, 8, (3, 3), offset=2),
            inner.output_shape,
            padding=1,
            row_mask=outer_mask,
        )

    reference = _reference(inner, (scale,), outer)
    candidate, estimate = _build(inner, (scale,), outer)
    gathered = candidate.gather_rows(
        np.arange(candidate.shape[0]), max_nnz=reference.nnz
    )

    _assert_csr_dyadic_equal(gathered, reference)
    assert estimate.resident_bytes == candidate.resident_bytes
    assert estimate.coefficient_entries == candidate._coefficients.size
    assert estimate.fused_total_work * 4 <= estimate.unfused_path_products
    assert candidate.logical_expanded_nnz >= reference.nnz


def test_matvec_bias_monomial_and_support_sliced_left_compose_without_expansion(
    monkeypatch,
):
    inner = ImplicitConv2DOp(
        _dyadic_kernel(8, 2, (3, 3)),
        (1, 2, 7, 8),
        padding=1,
    )
    scale = np.array([1, -0.5, 0, 0.25, -1, 2, 0.5, -0.25])
    outer = ImplicitConv2DOp(
        _dyadic_kernel(3, 8, (3, 3), offset=2),
        inner.output_shape,
        padding=1,
    )
    reference = _reference(inner, (scale,), outer)
    candidate, _ = _build(inner, (scale,), outer)

    def forbidden_expansion(_self):
        raise AssertionError("prototype expanded an input Conv descriptor")

    monkeypatch.setattr(
        ImplicitConv2DOp, "to_csr_reference", forbidden_expansion
    )
    x = ((np.arange(candidate.shape[1]) % 7) - 3) / 8.0
    assert np.array_equal(candidate.matvec(x), np.asarray(reference @ x))

    middle_scale = _stationary_full(scale, inner.output_shape)
    inner_bias = ((np.arange(inner.shape[0]) % 5) - 2) / 16.0
    outer_bias = ((np.arange(outer.shape[0]) % 3) - 1) / 8.0
    propagated_bias = outer.matvec(middle_scale * inner_bias) + outer_bias
    expected_affine = reference @ x + propagated_bias
    assert np.array_equal(candidate.matvec(x) + propagated_bias, expected_affine)

    chosen = np.array([0, 5, 17, candidate.shape[0] - 1])
    monomial = sp.csr_matrix(
        (
            np.array([1.0, -0.5, 0.25, 2.0]),
            (np.arange(chosen.size), chosen),
        ),
        shape=(chosen.size, candidate.shape[0]),
    )
    monomial_expected = _canonical(monomial @ reference)
    monomial_result = candidate.left_compose(
        monomial, max_nnz=monomial_expected.nnz
    )
    _assert_csr_dyadic_equal(monomial_result, monomial_expected)

    general = sp.csr_matrix(
        (
            np.array([1.0, -0.5, 0.25, 0.5]),
            (
                np.array([0, 0, 1, 1]),
                np.array([0, 5, 17, 19]),
            ),
        ),
        shape=(2, candidate.shape[0]),
    )
    general_expected = _canonical(general @ reference)
    support_nnz = reference[np.array([0, 5, 17, 19])].nnz
    general_result = candidate.left_compose(
        general, max_nnz=max(general_expected.nnz, support_nnz)
    )
    _assert_csr_dyadic_equal(general_result, general_expected)


def test_non_dyadic_compilation_is_deterministic_and_numerically_equivalent():
    rng = np.random.default_rng(20260831)
    inner = ImplicitConv2DOp(
        rng.normal(size=(8, 2, 3, 3)), (1, 2, 7, 8), padding=1
    )
    scale = rng.normal(size=8)
    outer = ImplicitConv2DOp(
        rng.normal(size=(3, 8, 3, 3)), inner.output_shape, padding=1
    )
    reference = _reference(inner, (scale,), outer)
    first, _ = _build(inner, (scale,), outer)
    second, _ = _build(inner, (scale,), outer)
    first_csr = first.gather_rows(
        np.arange(first.shape[0]), max_nnz=64_000_000
    )
    second_csr = second.gather_rows(
        np.arange(second.shape[0]), max_nnz=64_000_000
    )

    assert first.content_key == second.content_key
    assert np.array_equal(first._coefficients, second._coefficients)
    _assert_csr_dyadic_equal(first_csr, second_csr)
    assert np.array_equal(first_csr.indptr, reference.indptr)
    assert np.array_equal(first_csr.indices, reference.indices)
    assert np.allclose(first_csr.data, reference.data, rtol=2e-13, atol=2e-13)


def test_general_left_compose_rejects_full_support_and_nnz_caps():
    inner = ImplicitConv2DOp(
        _dyadic_kernel(8, 2, (3, 3)), (1, 2, 7, 7), padding=1
    )
    scale = np.ones(8)
    outer = ImplicitConv2DOp(
        _dyadic_kernel(3, 8, (3, 3), offset=1),
        inner.output_shape,
        padding=1,
    )
    candidate, _ = _build(inner, (scale,), outer)
    reference = _reference(inner, (scale,), outer)
    with pytest.raises(PrototypeReject, match="gather_result_nnz_limit"):
        candidate.gather_rows(np.arange(candidate.shape[0]), reference.nnz - 1)
    diagonal = np.arange(candidate.shape[0])
    full_general = sp.csr_matrix(
        (
            np.concatenate((np.ones(candidate.shape[0]), np.array([0.5]))),
            (
                np.concatenate((diagonal, np.array([0]))),
                np.concatenate((diagonal, np.array([1]))),
            ),
        ),
        shape=(candidate.shape[0], candidate.shape[0]),
    )
    with pytest.raises(
        PrototypeReject, match="support_slice_would_expand_full_operator"
    ):
        candidate.left_compose(full_general, max_nnz=64_000_000)


def test_spatial_scale_and_spatial_inner_mask_reject_without_compilation():
    inner = ImplicitConv2DOp(
        _dyadic_kernel(8, 2, (3, 3)), (1, 2, 7, 7), padding=1
    )
    outer = ImplicitConv2DOp(
        _dyadic_kernel(3, 8, (3, 3)), inner.output_shape, padding=1
    )
    spatial_scale = _stationary_full(np.ones(8), inner.output_shape)
    spatial_scale[1] = 0.5
    decision = ComposedStencilPrototype.try_build(
        inner, (spatial_scale,), outer, _gate(outer)
    )
    assert not decision.triggered
    assert decision.reason == "middle_0_not_channel_stationary"
    assert decision.estimate is None

    bad_mask = np.ones(inner.shape[0], dtype=bool)
    bad_mask[1] = False
    masked_inner = ImplicitConv2DOp(
        _dyadic_kernel(8, 2, (3, 3)),
        (1, 2, 7, 7),
        padding=1,
        row_mask=bad_mask,
    )
    decision = ComposedStencilPrototype.try_build(
        masked_inner, (np.ones(8),), outer, _gate(outer)
    )
    assert not decision.triggered
    assert decision.reason == "inner_row_mask_not_channel_stationary"


def test_unknown_middle_operator_and_shape_mismatch_reject():
    inner = ImplicitConv2DOp(
        _dyadic_kernel(8, 2, (3, 3)), (1, 2, 7, 7), padding=1
    )
    outer = ImplicitConv2DOp(
        _dyadic_kernel(3, 8, (3, 3)), inner.output_shape, padding=1
    )
    diagonal = np.arange(inner.shape[0])
    non_diagonal = sp.csr_matrix(
        (
            np.concatenate((np.ones(inner.shape[0]), np.array([1.0]))),
            (
                np.concatenate((diagonal, np.array([0]))),
                np.concatenate((diagonal, np.array([1]))),
            ),
        ),
        shape=(inner.shape[0], inner.shape[0]),
    )
    decision = ComposedStencilPrototype.try_build(
        inner, (non_diagonal,), outer, _gate(outer)
    )
    assert not decision.triggered
    assert decision.reason == "middle_0_not_diagonal"

    wrong_outer = ImplicitConv2DOp(
        _dyadic_kernel(3, 8, (3, 3)), (1, 8, 8, 7), padding=1
    )
    decision = ComposedStencilPrototype.try_build(
        inner, (np.ones(8),), wrong_outer, _gate(wrong_outer)
    )
    assert not decision.triggered
    assert decision.reason == "intermediate_shape_mismatch"


def test_every_frozen_resource_gate_has_stable_rejection_reason():
    inner = ImplicitConv2DOp(
        _dyadic_kernel(8, 2, (3, 3)), (1, 2, 7, 7), padding=1
    )
    scale = np.ones(8)
    outer = ImplicitConv2DOp(
        _dyadic_kernel(3, 8, (3, 3)), inner.output_shape, padding=1
    )
    baseline = ComposedStencilPrototype.try_build(
        inner, (scale,), outer, _gate(outer)
    )
    assert baseline.triggered
    estimate = baseline.estimate
    default = FrozenV1Limits()
    cases = [
        (
            replace(
                default,
                max_descriptor_contraction_products=(
                    estimate.contraction_products - 1
                ),
            ),
            {},
            "contraction_product_limit",
        ),
        (
            replace(default, max_transaction_work=estimate.fused_total_work - 1),
            {},
            "transaction_work_limit",
        ),
        (
            replace(default, max_coefficient_entries=estimate.coefficient_entries - 1),
            {},
            "coefficient_entry_limit",
        ),
        (
            replace(default, max_resident_bytes=estimate.resident_bytes - 1),
            {},
            "resident_payload_limit",
        ),
        (
            replace(
                default,
                max_transient_bytes=estimate.controlled_transient_bytes - 1,
            ),
            {},
            "controlled_transient_limit",
        ),
        (
            replace(default, max_result_nnz=estimate.result_nnz_upper - 1),
            {},
            "result_nnz_limit",
        ),
        (
            default,
            {
                "reachable_before_bytes": estimate.resident_bytes,
                "reachable_after_other_bytes": 0,
            },
            "physical_metric_not_reduced",
        ),
    ]
    for limits, gate_kwargs, expected in cases:
        decision = ComposedStencilPrototype.try_build(
            inner,
            (scale,),
            outer,
            _gate(outer, limits=limits, **gate_kwargs),
        )
        assert not decision.triggered
        assert decision.reason == expected
        assert decision.operator is None

    cumulative_cases = [
        (
            {
                "transaction_total_work": (
                    default.max_transaction_work
                    - estimate.fused_total_work
                    + 1
                )
            },
            "transaction_work_limit",
        ),
        (
            {
                "transaction_transient_live_bytes": (
                    default.max_transient_bytes
                    - estimate.controlled_transient_bytes
                    + 1
                )
            },
            "controlled_transient_limit",
        ),
    ]
    for gate_kwargs, expected in cumulative_cases:
        decision = ComposedStencilPrototype.try_build(
            inner,
            (scale,),
            outer,
            _gate(outer, **gate_kwargs),
        )
        assert not decision.triggered
        assert decision.reason == expected

    decision = ComposedStencilPrototype.try_build(
        inner,
        (scale,),
        outer,
        _gate(
            outer,
            reachable_before_bytes=None,
            reachable_after_other_bytes=None,
        ),
    )
    assert decision.reason == "physical_metric_unproven"


def test_pure_depthwise_is_soundly_rejected_by_uniform_quarter_work_gate():
    inner = ImplicitConv2DOp(
        _dyadic_kernel(4, 1, (3, 3)),
        (1, 4, 8, 8),
        padding=1,
        groups=4,
    )
    outer = ImplicitConv2DOp(
        _dyadic_kernel(4, 1, (3, 3), offset=2),
        inner.output_shape,
        padding=1,
        groups=4,
    )
    decision = ComposedStencilPrototype.try_build(
        inner, (np.ones(4),), outer, _gate(outer)
    )
    assert not decision.triggered
    assert decision.reason == "insufficient_work_reduction"
    assert decision.estimate.fused_total_work > (
        decision.estimate.unfused_path_products // 4
    )


def test_tiny36_registered_shape_estimate_without_large_compilation():
    kernel = np.zeros((128, 128, 3, 3), dtype=np.float64)
    inner = ImplicitConv2DOp(kernel, (1, 128, 14, 14), padding=1)
    outer = ImplicitConv2DOp(kernel, inner.output_shape, padding=1)
    selected = np.arange(2_180, dtype=np.int64)
    decision = ComposedStencilPrototype.try_build(
        inner,
        (np.ones(128),),
        outer,
        _gate(
            outer,
            selected_rows=selected,
            reachable_before_bytes=0,
            reachable_after_other_bytes=0,
        ),
    )
    assert not decision.triggered
    assert decision.reason == "physical_metric_not_reduced"
    estimate = decision.estimate
    assert estimate.contraction_products == 169_869_312
    assert estimate.coefficient_entries == 1_327_104
    assert estimate.contraction_products <= 200_000_000
    assert estimate.fused_total_work <= 256_000_000
    assert estimate.coefficient_entries <= 2_000_000
    assert estimate.resident_bytes <= 64 * 1024 * 1024
    assert estimate.controlled_transient_bytes <= 1024**3
    assert estimate.fused_total_work * 4 <= estimate.unfused_path_products


def test_prototype_is_not_imported_by_production_runtime():
    root = Path(__file__).resolve().parents[2]
    production = (
        root / "act" / "back_end" / "hybridz_tf" / "tf_cnn.py"
    ).read_text(encoding="utf-8")
    assert "composed_stencil_prototype" not in production
