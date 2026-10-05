"""Focused gates for the isolated composed Conv2D stencil candidate V2."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
import experiments.neural_hz_20260831.composed_conv2d_stencil_candidate_v2 as v2
from experiments.neural_hz_20260831.composed_conv2d_stencil_candidate_v2 import (
    CandidateV2Reject,
    ComposedConv2DStencilCandidateV2,
    FrozenV1Limits,
    GateRequestV2,
    V2Transaction,
)


def _dyadic_kernel(out_channels, in_per_group, shape, *, offset=0):
    size = out_channels * in_per_group * shape[0] * shape[1]
    values = ((np.arange(size, dtype=np.int64) + offset) % 11 - 5) / 16.0
    return values.reshape(out_channels, in_per_group, shape[0], shape[1])


def _positive_dyadic_kernel(out_channels, in_per_group, shape, *, offset=0):
    size = out_channels * in_per_group * shape[0] * shape[1]
    values = ((np.arange(size, dtype=np.int64) + offset) % 7 + 1) / 16.0
    return values.reshape(out_channels, in_per_group, shape[0], shape[1])


def _stationary_full(channel, shape):
    channel = np.asarray(channel, dtype=np.float64)
    return np.broadcast_to(
        channel.reshape(1, shape[1], 1, 1), shape
    ).copy().reshape(-1)


def _gate(
    outer,
    *,
    transaction=None,
    selected_rows=None,
    limits=None,
    reachable_before_bytes=10**12,
    reachable_after_other_bytes=0,
):
    if transaction is None:
        transaction = V2Transaction()
    if selected_rows is None:
        selected_rows = np.arange(outer.shape[0], dtype=np.int64)
    return GateRequestV2(
        selected_rows=selected_rows,
        reachable_before_bytes=reachable_before_bytes,
        reachable_after_other_bytes=reachable_after_other_bytes,
        transaction=transaction,
        limits=FrozenV1Limits() if limits is None else limits,
    )


def _canonical(matrix):
    result = matrix.tocsr().astype(np.float64, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    return result


def _assert_csr_exact(candidate, reference):
    candidate = _canonical(candidate)
    reference = _canonical(reference)
    assert candidate.shape == reference.shape
    assert np.array_equal(candidate.indptr, reference.indptr)
    assert np.array_equal(candidate.indices, reference.indices)
    assert np.array_equal(candidate.data, reference.data)


def _scale_payload(scale, middle_shape):
    total = int(np.prod(middle_shape))
    if isinstance(scale, DiagonalLinearOp):
        return np.asarray(scale._diagonal, dtype=np.float64)
    if sp.issparse(scale):
        return np.asarray(scale.diagonal(), dtype=np.float64)
    raw = np.asarray(scale, dtype=np.float64)
    if raw.ndim == 0:
        return np.full(total, float(raw), dtype=np.float64)
    if raw.ndim == 2:
        return np.diag(raw)
    raw = raw.reshape(-1)
    if raw.size == middle_shape[1]:
        return _stationary_full(raw, middle_shape)
    return raw


def _reference(inner, middle_ops, outer):
    scale = np.ones(inner.shape[0], dtype=np.float64)
    for middle in middle_ops:
        scale *= _scale_payload(middle, inner.output_shape)
    return _canonical(
        outer.to_csr_reference()
        @ sp.diags(scale, format="csr")
        @ inner.to_csr_reference()
    )


def _build(inner, middle_ops, outer, *, gate=None):
    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        tuple(middle_ops),
        outer,
        _gate(outer) if gate is None else gate,
    )
    assert decision.triggered, (decision.reason, decision.estimate)
    assert decision.operator is not None
    return decision.operator, decision


def _case(name):
    if name == "ordinary":
        inner = ImplicitConv2DOp(
            _dyadic_kernel(8, 2, (3, 3)),
            (1, 2, 7, 8),
            padding=1,
        )
        scales = (
            np.array([1, -0.5, 0, 0.25, -1, 2, 0.5, -0.25]),
            np.array([0.5, 1, -1, 2, 0.25, -0.5, 1, 2]),
        )
        outer = ImplicitConv2DOp(
            _dyadic_kernel(3, 8, (3, 3), offset=2),
            inner.output_shape,
            padding=1,
        )
        limits = None
    elif name == "asymmetric_geometry":
        inner = ImplicitConv2DOp(
            _dyadic_kernel(8, 2, (3, 2)),
            (1, 2, 9, 10),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(1, 2),
        )
        scales = (
            np.array([1, -0.5, 0.25, -0.25, 2, 0, 0.5, -1]),
        )
        outer = ImplicitConv2DOp(
            _dyadic_kernel(4, 8, (2, 3), offset=3),
            inner.output_shape,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
        )
        limits = None
    elif name == "aligned_groups":
        inner = ImplicitConv2DOp(
            _dyadic_kernel(16, 4, (3, 3)),
            (1, 8, 8, 8),
            padding=1,
            groups=2,
        )
        scales = (np.resize(np.array([1, -1, 0.5, -0.5]), 16),)
        outer = ImplicitConv2DOp(
            _dyadic_kernel(8, 8, (3, 3), offset=1),
            inner.output_shape,
            padding=1,
            groups=2,
        )
        limits = None
    elif name == "misaligned_groups":
        # Inner middle blocks have width 12; outer blocks have width 8.
        inner = ImplicitConv2DOp(
            _dyadic_kernel(24, 4, (3, 3)),
            (1, 8, 9, 9),
            padding=1,
            groups=2,
        )
        scales = (np.resize(np.array([1, -1, 0.5, -0.5, 2, 0]), 24),)
        outer = ImplicitConv2DOp(
            _dyadic_kernel(9, 8, (3, 3), offset=4),
            inner.output_shape,
            padding=1,
            groups=3,
        )
        limits = None
    elif name == "inner_depthwise_multiplier":
        # Depthwise inner Conv with multiplier eight; the v1 work gate passes.
        inner = ImplicitConv2DOp(
            _dyadic_kernel(16, 1, (3, 3)),
            (1, 2, 8, 8),
            padding=1,
            groups=2,
        )
        scales = (np.resize(np.array([1, -1, 0.5, -0.5]), 16),)
        outer = ImplicitConv2DOp(
            _dyadic_kernel(4, 16, (3, 3), offset=4),
            inner.output_shape,
            padding=1,
        )
        limits = None
    elif name == "outer_depthwise_multiplier":
        # A depthwise outer Conv has no 4x work reduction; relax only that
        # synthetic semantic test.  A separate test proves frozen-v1 rejects.
        inner = ImplicitConv2DOp(
            _dyadic_kernel(8, 8, (3, 3)),
            (1, 8, 8, 8),
            padding=1,
        )
        scales = (np.resize(np.array([1, -1, 0.5, -0.5]), 8),)
        outer = ImplicitConv2DOp(
            _dyadic_kernel(16, 1, (3, 3), offset=4),
            inner.output_shape,
            padding=1,
            groups=8,
        )
        limits = replace(
            FrozenV1Limits(),
            max_work_numerator=4,
            max_work_denominator=1,
        )
    elif name == "batch_channel_and_outer_masks":
        middle_shape = (2, 8, 6, 7)
        inner_mask = np.ones(middle_shape, dtype=bool)
        inner_mask[:, 2] = False
        inner_mask[:, 6] = False
        inner = ImplicitConv2DOp(
            _dyadic_kernel(8, 2, (3, 3)),
            (2, 2, 6, 7),
            padding=1,
            row_mask=inner_mask,
        )
        diagonal = _stationary_full(
            np.array([1, -0.5, 0, 0.25, -1, 2, 0.5, -0.25]),
            inner.output_shape,
        )
        scales = (DiagonalLinearOp(diagonal),)
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
        limits = None
    else:  # pragma: no cover - test helper defense
        raise AssertionError(name)
    return inner, scales, outer, limits


EXACT_CASES = (
    "ordinary",
    "asymmetric_geometry",
    "aligned_groups",
    "misaligned_groups",
    "inner_depthwise_multiplier",
    "outer_depthwise_multiplier",
    "batch_channel_and_outer_masks",
)


@pytest.mark.parametrize("name", EXACT_CASES)
def test_small_dyadic_csr_oracles_are_elementwise_exact(name):
    inner, scales, outer, limits = _case(name)
    reference = _reference(inner, scales, outer)
    candidate, decision = _build(
        inner,
        scales,
        outer,
        gate=_gate(outer, limits=limits),
    )
    gathered = candidate.gather_rows(
        np.arange(candidate.shape[0]), max_nnz=reference.nnz
    )

    _assert_csr_exact(gathered, reference)
    ledger = candidate.compilation_ledger
    assert ledger.actual_contraction_products == (
        ledger.intersection_formula_products
    )
    assert ledger.actual_contraction_products == ledger.gate_formula_products
    assert ledger.gate_formula_products == decision.estimate.contraction_products
    assert ledger.sigma_fold_products == ledger.expected_sigma_fold_products
    assert ledger.expected_sigma_fold_products == outer._kernel.size


def test_misaligned_group_intersections_have_canonical_metadata_order():
    inner, scales, outer, _ = _case("misaligned_groups")
    candidate, _ = _build(inner, scales, outer)
    metadata = candidate._block_metadata
    assert [tuple(int(v) for v in row[:4]) for row in metadata] == [
        (0, 0, 0, 8),
        (1, 0, 8, 12),
        (1, 1, 12, 16),
        (2, 1, 16, 24),
    ]
    assert np.array_equal(candidate._outer_group_indptr, [0, 1, 3, 4])


def test_exact_logical_nnz_counts_unique_spatial_columns_after_collision():
    inner = ImplicitConv2DOp(
        _positive_dyadic_kernel(8, 4, (3, 3)),
        (1, 4, 7, 8),
        padding=1,
    )
    outer = ImplicitConv2DOp(
        _positive_dyadic_kernel(3, 8, (3, 3), offset=2),
        inner.output_shape,
        padding=1,
    )
    scales = (np.ones(8),)
    reference = _reference(inner, scales, outer)
    candidate, decision = _build(inner, scales, outer)

    assert candidate.logical_expanded_nnz == reference.nnz
    assert decision.estimate.exact_full_logical_nnz == reference.nnz
    assert decision.estimate.exact_selected_logical_nnz == reference.nnz
    assert decision.estimate.emission_contributions > reference.nnz


def test_resident_breakdown_matches_every_owned_numeric_buffer():
    inner, scales, outer, _ = _case("misaligned_groups")
    candidate, decision = _build(inner, scales, outer)
    resident = candidate.resident_breakdown
    assert resident.coefficient_bytes == sum(
        array.nbytes for array in candidate._coefficients
    )
    assert resident.block_metadata_bytes == candidate._block_metadata.nbytes
    assert (
        resident.outer_group_indptr_bytes
        == candidate._outer_group_indptr.nbytes
    )
    assert (
        resident.reachable_input_count_bytes
        == candidate._reachable_input_counts.nbytes
    )
    assert resident.outer_row_mask_bytes == 0
    assert resident.total_bytes == candidate.resident_bytes
    assert resident.total_bytes == decision.estimate.resident_bytes
    assert candidate.resident_entries == (
        sum(array.size for array in candidate._coefficients)
        + candidate._block_metadata.size
        + candidate._outer_group_indptr.size
        + candidate._reachable_input_counts.size
    )


def test_typed_content_key_is_binary_canonical_and_compilation_deterministic():
    inner, scales, outer, _ = _case("misaligned_groups")
    first, _ = _build(inner, scales, outer)
    second, _ = _build(inner, scales, outer)

    assert first.content_key == second.content_key
    assert first.content_key[0] == "composed_conv2d_stencil_candidate_v2"
    middle_key = first.content_key[3]
    assert middle_key[0] == "middle_scale"
    assert middle_key[1][0:3] == (
        "canonical_ndarray_v1",
        "float64_le",
        (24,),
    )
    assert isinstance(middle_key[1][3], bytes)
    assert len(middle_key[1][3]) == 32
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            first._coefficients, second._coefficients, strict=True
        )
    )
    assert "repr(" not in inspect.getsource(v2)


def test_descriptor_transaction_deduplicates_only_compilation_and_defers_emission():
    inner, scales, outer, _ = _case("ordinary")
    transaction = V2Transaction()
    rows = np.arange(outer.shape[0], dtype=np.int64)
    first, first_decision = _build(
        inner,
        scales,
        outer,
        gate=_gate(outer, transaction=transaction, selected_rows=rows),
    )
    after_first = transaction.snapshot()
    assert after_first.contraction_used == (
        first_decision.estimate.contraction_products
    )
    assert first_decision.estimate.prospective_emission_products == (
        first_decision.estimate.emission_contributions
    )
    assert first_decision.estimate.emission_accounting_deferred
    assert after_first.descriptor_resident_used_bytes == first.resident_bytes
    assert after_first.transient_live_bytes == 0
    assert after_first.transient_peak_bytes > 0
    assert len(after_first.compiled_content_keys) == 1
    assert first_decision.estimate.transaction_resident_delta_bytes == (
        first.resident_bytes
    )
    assert first_decision.estimate.transaction_transient_delta_bytes == (
        first_decision.estimate.compilation_transient_bytes
        + first_decision.estimate.input_snapshot_bytes
    )

    second, second_decision = _build(
        inner,
        scales,
        outer,
        gate=_gate(outer, transaction=transaction, selected_rows=rows),
    )
    assert second is first
    assert second_decision.reused_descriptor
    assert second_decision.estimate.transaction_resident_delta_bytes == 0
    assert second_decision.estimate.transaction_transient_delta_bytes == 0
    assert transaction.snapshot() == after_first

    reversed_rows = rows[::-1].copy()
    tight_limits = replace(
        FrozenV1Limits(),
        max_transaction_work=(
            after_first.contraction_used
            + first_decision.estimate.emission_contributions
            - 1
        ),
    )
    tight_reject = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        scales,
        outer,
        _gate(
            outer,
            transaction=transaction,
            selected_rows=reversed_rows,
            limits=tight_limits,
        ),
    )
    assert not tight_reject.triggered
    assert tight_reject.reason == "transaction_work_limit"
    assert transaction.snapshot() == after_first

    third, third_decision = _build(
        inner,
        scales,
        outer,
        gate=_gate(
            outer,
            transaction=transaction,
            selected_rows=reversed_rows,
        ),
    )
    after_third = transaction.snapshot()
    assert third is first
    assert third_decision.reused_descriptor
    assert third_decision.estimate.transaction_resident_delta_bytes == 0
    assert third_decision.estimate.transaction_transient_delta_bytes == 0
    assert after_third == after_first
    assert third_decision.estimate.emission_accounting_deferred
    assert third_decision.estimate.prospective_emission_products == (
        third_decision.estimate.emission_contributions
    )


def test_failed_post_reservation_compilation_rolls_transaction_back_exactly():
    inner = ImplicitConv2DOp(
        np.full((8, 2, 3, 3), 1e308),
        (1, 2, 7, 7),
        padding=1,
    )
    outer = ImplicitConv2DOp(
        np.full((3, 8, 3, 3), 1e308),
        inner.output_shape,
        padding=1,
    )
    scale = np.ones(8)
    transaction = V2Transaction()
    before = transaction.snapshot()
    inner_bytes = inner._kernel.tobytes()
    outer_bytes = outer._kernel.tobytes()
    scale_bytes = scale.tobytes()

    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        (scale,),
        outer,
        _gate(outer, transaction=transaction),
    )
    assert not decision.triggered
    assert decision.reason == "compiled_coefficients_nonfinite"
    assert decision.operator is None
    assert transaction.snapshot() == before
    assert inner._kernel.tobytes() == inner_bytes
    assert outer._kernel.tobytes() == outer_bytes
    assert scale.tobytes() == scale_bytes


def test_matvec_bias_and_both_left_compose_paths_never_expand_input_convs(
    monkeypatch,
):
    inner, scales, outer, _ = _case("ordinary")
    reference = _reference(inner, scales, outer)
    candidate, _ = _build(inner, scales, outer)

    def forbidden(_self):
        raise AssertionError("input Conv CSR expansion")

    monkeypatch.setattr(ImplicitConv2DOp, "to_csr_reference", forbidden)
    rows = np.array([0, 5, 17, candidate.shape[0] - 1])
    _assert_csr_exact(
        candidate.gather_rows(rows, max_nnz=reference[rows].nnz),
        reference[rows],
    )

    x = ((np.arange(candidate.shape[1]) % 7) - 3) / 8.0
    assert np.array_equal(candidate.matvec(x), np.asarray(reference @ x))
    total_scale = np.ones(inner.shape[0])
    for scale in scales:
        total_scale *= _scale_payload(scale, inner.output_shape)
    inner_bias = ((np.arange(inner.shape[0]) % 5) - 2) / 16.0
    outer_bias = ((np.arange(outer.shape[0]) % 3) - 1) / 8.0
    propagated_bias = outer.matvec(total_scale * inner_bias) + outer_bias
    assert np.array_equal(
        candidate.matvec(x) + propagated_bias,
        np.asarray(reference @ x) + propagated_bias,
    )

    monomial = sp.csr_matrix(
        (
            np.array([1.0, -0.5, 0.25, 2.0]),
            (np.arange(rows.size), rows),
        ),
        shape=(rows.size, candidate.shape[0]),
    )
    expected = _canonical(monomial @ reference)
    _assert_csr_exact(
        candidate.left_compose(monomial, max_nnz=expected.nnz), expected
    )

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
    expected = _canonical(general @ reference)
    support_nnz = reference[np.array([0, 5, 17, 19])].nnz
    _assert_csr_exact(
        candidate.left_compose(
            general, max_nnz=max(expected.nnz, support_nnz)
        ),
        expected,
    )


def test_left_compose_and_gather_caps_fail_closed():
    inner, scales, outer, _ = _case("ordinary")
    reference = _reference(inner, scales, outer)
    candidate, _ = _build(inner, scales, outer)
    with pytest.raises(CandidateV2Reject, match="gather_result_nnz_limit"):
        candidate.gather_rows(
            np.arange(candidate.shape[0]), max_nnz=reference.nnz - 1
        )

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
        CandidateV2Reject, match="support_slice_would_expand_full_operator"
    ):
        candidate.left_compose(full_general, max_nnz=64_000_000)


def test_non_dyadic_compilation_is_bitwise_deterministic_and_equivalent():
    rng = np.random.default_rng(20260831)
    inner = ImplicitConv2DOp(
        rng.normal(size=(12, 3, 3, 2)),
        (1, 6, 8, 9),
        padding=(1, 0),
        groups=2,
    )
    scale = rng.normal(size=12)
    outer = ImplicitConv2DOp(
        rng.normal(size=(6, 4, 2, 3)),
        inner.output_shape,
        padding=1,
        groups=3,
    )
    relaxed = replace(
        FrozenV1Limits(), max_work_numerator=2, max_work_denominator=1
    )
    first, _ = _build(
        inner, (scale,), outer, gate=_gate(outer, limits=relaxed)
    )
    second, _ = _build(
        inner, (scale,), outer, gate=_gate(outer, limits=relaxed)
    )
    assert first.content_key == second.content_key
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            first._coefficients, second._coefficients, strict=True
        )
    )
    reference = _reference(inner, (scale,), outer)
    candidate = first.gather_rows(
        np.arange(first.shape[0]), max_nnz=64_000_000
    )
    candidate = _canonical(candidate)
    assert np.array_equal(candidate.indptr, reference.indptr)
    assert np.array_equal(candidate.indices, reference.indices)
    assert np.allclose(candidate.data, reference.data, rtol=2e-13, atol=2e-13)


def test_every_frozen_v1_resource_gate_has_a_stable_reason():
    inner, scales, outer, _ = _case("ordinary")
    _, baseline = _build(inner, scales, outer)
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
            replace(
                default,
                max_transaction_work=estimate.fused_total_work - 1,
            ),
            {},
            "transaction_work_limit",
        ),
        (
            replace(
                default,
                max_coefficient_entries=estimate.coefficient_entries - 1,
            ),
            {},
            "coefficient_entry_limit",
        ),
        (
            replace(
                default,
                max_resident_bytes=estimate.resident_bytes - 1,
            ),
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
            replace(
                default,
                max_result_nnz=estimate.result_nnz_upper - 1,
            ),
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
        transaction = V2Transaction()
        before = transaction.snapshot()
        decision = ComposedConv2DStencilCandidateV2.try_build(
            inner,
            scales,
            outer,
            _gate(
                outer,
                transaction=transaction,
                limits=limits,
                **gate_kwargs,
            ),
        )
        assert not decision.triggered
        assert decision.reason == expected
        assert decision.operator is None
        assert transaction.snapshot() == before

    transaction = V2Transaction()
    before = transaction.snapshot()
    unproven = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        scales,
        outer,
        _gate(
            outer,
            transaction=transaction,
            reachable_before_bytes=None,
            reachable_after_other_bytes=None,
        ),
    )
    assert unproven.reason == "physical_metric_unproven"
    assert transaction.snapshot() == before


def test_empty_selection_and_pure_depthwise_close_under_uniform_v1_gate():
    inner, scales, outer, _ = _case("ordinary")
    empty = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        scales,
        outer,
        _gate(outer, selected_rows=np.empty(0, dtype=np.int64)),
    )
    assert not empty.triggered
    assert empty.reason == "no_unfused_work"

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
    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner, (np.ones(4),), outer, _gate(outer)
    )
    assert not decision.triggered
    assert decision.reason == "insufficient_work_reduction"


def test_inadmissible_middle_and_masks_reject_without_mutation_or_budget():
    inner, scales, outer, _ = _case("ordinary")
    transaction = V2Transaction()
    snapshot = transaction.snapshot()
    inner_before = inner._kernel.tobytes()
    outer_before = outer._kernel.tobytes()

    spatial_scale = _stationary_full(np.ones(8), inner.output_shape)
    spatial_scale[1] = 0.5
    scale_before = spatial_scale.tobytes()
    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        (spatial_scale,),
        outer,
        _gate(outer, transaction=transaction),
    )
    assert decision.reason == "middle_0_not_channel_stationary"
    assert transaction.snapshot() == snapshot
    assert spatial_scale.tobytes() == scale_before

    bad_mask = np.ones(inner.shape[0], dtype=bool)
    bad_mask[1] = False
    masked_inner = ImplicitConv2DOp(
        inner._kernel,
        inner.input_shape,
        padding=1,
        row_mask=bad_mask,
    )
    decision = ComposedConv2DStencilCandidateV2.try_build(
        masked_inner,
        scales,
        outer,
        _gate(outer, transaction=transaction),
    )
    assert decision.reason == "inner_row_mask_not_channel_stationary"
    assert transaction.snapshot() == snapshot

    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        (object(),),
        outer,
        _gate(outer, transaction=transaction),
    )
    assert decision.reason == "middle_0_not_real"
    assert transaction.snapshot() == snapshot
    assert inner._kernel.tobytes() == inner_before
    assert outer._kernel.tobytes() == outer_before


def test_shape_mismatch_and_non_diagonal_operator_reject():
    inner, _, outer, _ = _case("ordinary")
    wrong_outer = ImplicitConv2DOp(
        _dyadic_kernel(3, 8, (3, 3)),
        (1, 8, 8, 8),
        padding=1,
    )
    mismatch = ComposedConv2DStencilCandidateV2.try_build(
        inner, (np.ones(8),), wrong_outer, _gate(wrong_outer)
    )
    assert mismatch.reason == "intermediate_shape_mismatch"

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
    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner, (non_diagonal,), outer, _gate(outer)
    )
    assert decision.reason == "middle_0_not_diagonal"


def test_tiny36_registered_arithmetic_stays_inside_v1_before_physical_gate():
    kernel = np.zeros((128, 128, 3, 3), dtype=np.float64)
    inner = ImplicitConv2DOp(kernel, (1, 128, 14, 14), padding=1)
    outer = ImplicitConv2DOp(kernel, inner.output_shape, padding=1)
    selected = np.arange(2_180, dtype=np.int64)
    decision = ComposedConv2DStencilCandidateV2.try_build(
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


def test_selected_boolean_mask_preserves_ordered_gather_semantics():
    inner, scales, outer, _ = _case("ordinary")
    # Keep enough uniformly selected work for the frozen 4x gate while still
    # exercising boolean-mask normalization rather than an integer row list.
    mask = np.ones(outer.shape[0], dtype=bool)
    mask[::13] = False
    candidate, decision = _build(
        inner, scales, outer, gate=_gate(outer, selected_rows=mask)
    )
    assert decision.estimate.selected_rows == int(np.count_nonzero(mask))
    reference = _reference(inner, scales, outer)
    rows = np.flatnonzero(mask)
    _assert_csr_exact(
        candidate.gather_rows(rows, max_nnz=reference[rows].nnz),
        reference[rows],
    )


def test_candidate_remains_default_off_and_absent_from_production_runtime():
    root = Path(__file__).resolve().parents[2]
    module_name = "composed_conv2d_stencil_candidate_v2"
    production_files = (
        root / "act" / "back_end" / "hybridz_tf" / "exact_linear_op.py",
        root / "act" / "back_end" / "hybridz_tf" / "tf_cnn.py",
        root / "act" / "back_end" / "hybridz_tf" / "tf_mlp.py",
    )
    for path in production_files:
        assert module_name not in path.read_text(encoding="utf-8")
