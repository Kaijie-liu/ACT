"""Adversarial main-agent audit gates for the isolated V2 candidate."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
import experiments.neural_hz_20260831.composed_conv2d_stencil_candidate_v2 as v2
from experiments.neural_hz_20260831.composed_conv2d_stencil_candidate_v2 import (
    CandidateV2Reject,
    ComposedConv2DStencilCandidateV2,
    FrozenV1Limits,
    GateRequestV2,
    V2Transaction,
)


def _relaxed_limits() -> FrozenV1Limits:
    return replace(
        FrozenV1Limits(),
        max_work_numerator=10**9,
        max_work_denominator=1,
    )


def _gate(outer, transaction, *, rows=None):
    if rows is None:
        rows = np.arange(outer.shape[0], dtype=np.int64)
    return GateRequestV2(
        selected_rows=rows,
        reachable_before_bytes=10**12,
        reachable_after_other_bytes=0,
        transaction=transaction,
        limits=_relaxed_limits(),
    )


def _positive_case():
    inner_values = (
        np.arange(8 * 2 * 3 * 3, dtype=np.int64) % 7 + 1
    ).reshape(8, 2, 3, 3) / 16.0
    inner = ImplicitConv2DOp(inner_values, (1, 2, 7, 8), padding=1)
    scale = np.array([1, 0.5, 0.25, 2, 1.5, 0.75, 1.25, 0.125])
    outer_values = (
        np.arange(3 * 8 * 3 * 3, dtype=np.int64) % 5 + 1
    ).reshape(3, 8, 3, 3) / 16.0
    outer = ImplicitConv2DOp(outer_values, inner.output_shape, padding=1)
    return inner, scale, outer


def _build(inner, scale, outer, transaction=None):
    if transaction is None:
        transaction = V2Transaction()
    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        (scale,),
        outer,
        _gate(outer, transaction),
    )
    assert decision.triggered, (decision.reason, decision.estimate)
    return decision.operator, decision


def _canonical(matrix):
    result = matrix.tocsr().astype(np.float64, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    return result


def _reserve(transaction, **kwargs):
    request = transaction.new_reservation_request()
    return transaction.reserve_descriptor(request=request, **kwargs)


def _assert_same(left, right):
    left = _canonical(left)
    right = _canonical(right)
    assert left.shape == right.shape
    assert np.array_equal(left.indptr, right.indptr)
    assert np.array_equal(left.indices, right.indices)
    assert np.allclose(left.data, right.data, rtol=1e-13, atol=1e-13)


def test_transaction_serializes_open_reservations_and_rejects_foreign_owner():
    transaction = V2Transaction()
    before = transaction.snapshot()
    reservation, reason, quote = _reserve(transaction,
        content_key=("descriptor-a",),
        contraction_products=3,
        descriptor_resident_bytes=11,
        prospective_emission_products=5,
        transient_bytes=7,
        reachable_before_bytes=10**6,
        reachable_after_nontransaction_bytes=0,
        limits=FrozenV1Limits(),
    )
    assert reason is None and reservation is not None
    assert quote.needs_compile
    during = transaction.snapshot()
    assert during.reservation_open
    assert during.contraction_used == 3
    assert during.descriptor_resident_used_bytes == 0
    assert (during.transient_live_bytes, during.transient_peak_bytes) == (7, 7)

    other_request = transaction.new_reservation_request()
    second, reason, _ = transaction.reserve_descriptor(
        request=other_request,
        content_key=("descriptor-b",),
        contraction_products=11,
        descriptor_resident_bytes=19,
        prospective_emission_products=13,
        transient_bytes=17,
        reachable_before_bytes=10**6,
        reachable_after_nontransaction_bytes=0,
        limits=FrozenV1Limits(),
    )
    assert second is None
    assert reason == "transaction_reservation_conflict"
    assert transaction.snapshot() == during
    transaction.rollback_if_pending(other_request)
    assert transaction.snapshot() == during

    foreign = V2Transaction()
    foreign_before = foreign.snapshot()
    with pytest.raises(RuntimeError, match="foreign_reservation"):
        foreign.rollback(reservation)
    assert foreign.snapshot() == foreign_before
    assert transaction.snapshot() == during

    transaction.rollback(reservation)
    assert transaction.snapshot() == before
    transaction.rollback(reservation)
    assert transaction.snapshot() == before
    assert not hasattr(transaction, "_closed_reservation_tokens")


def test_reserve_allocation_failure_restores_snapshot_before_return():
    class FailingSet(set):
        def add(self, _value):
            raise MemoryError("injected reserve allocation failure")

    transaction = V2Transaction()
    transaction._reserved_content_keys = FailingSet()
    before = transaction.snapshot()
    with pytest.raises(MemoryError, match="injected reserve allocation failure"):
        _reserve(transaction,
            content_key=("descriptor",),
            contraction_products=3,
            descriptor_resident_bytes=11,
            prospective_emission_products=5,
            transient_bytes=7,
            reachable_before_bytes=10**6,
            reachable_after_nontransaction_bytes=0,
            limits=FrozenV1Limits(),
        )
    assert transaction.snapshot() == before
    assert not transaction._reserved_content_keys


def test_descriptor_reservation_is_opaque_and_rejects_invalid_budgets():
    transaction = V2Transaction()
    before = transaction.snapshot()
    common = dict(
        content_key=("descriptor",),
        contraction_products=0,
        descriptor_resident_bytes=0,
        prospective_emission_products=0,
        transient_bytes=0,
        reachable_before_bytes=1,
        reachable_after_nontransaction_bytes=0,
        limits=FrozenV1Limits(),
    )
    reservation, reason, quote = _reserve(transaction, **common)
    assert reason is None and reservation is not None
    assert quote.needs_compile  # Zero work is not a cache-hit signal.
    with pytest.raises(FrozenInstanceError):
        reservation.reservation_token = object()
    needs_compile, cached = transaction.reservation_descriptor_state(
        reservation
    )
    assert needs_compile and cached is None
    with pytest.raises(RuntimeError, match="descriptor_content_mismatch"):
        transaction.commit(reservation, None)
    transaction.rollback(reservation)
    assert transaction.snapshot() == before

    for field, value in (
        ("contraction_products", -1),
        ("descriptor_resident_bytes", -1),
        ("prospective_emission_products", -1),
        ("transient_bytes", -1),
        ("contraction_products", True),
    ):
        invalid = dict(common)
        invalid[field] = value
        with pytest.raises(CandidateV2Reject):
            _reserve(transaction, **invalid)
        assert transaction.snapshot() == before


def test_partial_descriptor_publication_failure_restores_exact_snapshot():
    class PublishThenFail(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise MemoryError("injected partial descriptor publication")

    inner, scale, outer = _positive_case()
    candidate, decision = _build(inner, scale, outer)
    transaction = V2Transaction()
    transaction._compiled = PublishThenFail()
    before = transaction.snapshot()
    reservation, reason, _ = _reserve(transaction,
        content_key=candidate.content_key,
        contraction_products=decision.estimate.contraction_products,
        descriptor_resident_bytes=candidate.resident_bytes,
        prospective_emission_products=0,
        transient_bytes=7,
        reachable_before_bytes=10**12,
        reachable_after_nontransaction_bytes=0,
        limits=FrozenV1Limits(),
    )
    assert reason is None and reservation is not None
    with pytest.raises(MemoryError, match="partial descriptor publication"):
        transaction.commit(reservation, candidate)
    assert transaction.snapshot() == before
    assert not transaction._compiled


@pytest.mark.parametrize("offset", [-1, 1])
def test_commit_rejects_under_or_overcharged_contraction_ledger(offset):
    inner, scale, outer = _positive_case()
    candidate, decision = _build(inner, scale, outer)
    actual = decision.estimate.contraction_products
    charged = actual + offset
    assert charged >= 0
    transaction = V2Transaction()
    before = transaction.snapshot()
    reservation, reason, _ = _reserve(transaction,
        content_key=candidate.content_key,
        contraction_products=charged,
        descriptor_resident_bytes=candidate.resident_bytes,
        prospective_emission_products=0,
        transient_bytes=7,
        reachable_before_bytes=10**12,
        reachable_after_nontransaction_bytes=0,
        limits=FrozenV1Limits(),
    )
    assert reason is None and reservation is not None
    with pytest.raises(RuntimeError, match="contraction_ledger_mismatch"):
        transaction.commit(reservation, candidate)
    transaction.rollback(reservation)
    assert transaction.snapshot() == before


def test_public_descriptor_reserve_cannot_bypass_local_limits():
    transaction = V2Transaction()
    before = transaction.snapshot()
    limits = FrozenV1Limits()
    common = dict(
        content_key=("descriptor",),
        prospective_emission_products=0,
        transient_bytes=0,
        reachable_before_bytes=10**12,
        reachable_after_nontransaction_bytes=0,
        limits=limits,
    )
    reservation, reason, _ = _reserve(transaction,
        contraction_products=(
            limits.max_descriptor_contraction_products + 1
        ),
        descriptor_resident_bytes=0,
        **common,
    )
    assert reservation is None and reason == "contraction_product_limit"
    assert transaction.snapshot() == before

    reservation, reason, _ = _reserve(transaction,
        contraction_products=0,
        descriptor_resident_bytes=limits.max_resident_bytes + 1,
        **common,
    )
    assert reservation is None and reason == "resident_payload_limit"
    assert transaction.snapshot() == before


def test_cached_descriptor_quote_must_match_authoritative_ledger():
    inner, scale, outer = _positive_case()
    transaction = V2Transaction()
    candidate, decision = _build(inner, scale, outer, transaction)
    before = transaction.snapshot()
    with pytest.raises(
        CandidateV2Reject,
        match="cached_descriptor_contraction_quote_mismatch",
    ):
        transaction.preview_descriptor(
            content_key=candidate.content_key,
            contraction_products=(
                decision.estimate.contraction_products - 1
            ),
            descriptor_resident_bytes=candidate.resident_bytes,
            prospective_emission_products=0,
            transient_bytes=0,
            reachable_before_bytes=10**12,
            reachable_after_nontransaction_bytes=0,
        )
    assert transaction.snapshot() == before

    reservation, reason, _ = _reserve(transaction,
        content_key=candidate.content_key,
        contraction_products=decision.estimate.contraction_products,
        descriptor_resident_bytes=candidate.resident_bytes,
        prospective_emission_products=0,
        transient_bytes=0,
        reachable_before_bytes=10**12,
        reachable_after_nontransaction_bytes=0,
        limits=replace(
            FrozenV1Limits(),
            max_descriptor_contraction_products=(
                decision.estimate.contraction_products - 1
            ),
        ),
    )
    assert reservation is None and reason == "contraction_product_limit"
    assert transaction.snapshot() == before


def test_success_decision_allocation_failure_never_commits_descriptor(
    monkeypatch,
):
    inner, scale, outer = _positive_case()
    transaction = V2Transaction()
    before = transaction.snapshot()
    original_decision_type = v2.BuildDecisionV2

    def injected_decision(*args, **kwargs):
        triggered = args[0] if args else kwargs["triggered"]
        if triggered:
            raise MemoryError("injected success decision allocation")
        return original_decision_type(*args, **kwargs)

    monkeypatch.setattr(v2, "BuildDecisionV2", injected_decision)
    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        (scale,),
        outer,
        _gate(outer, transaction),
    )
    assert not decision.triggered
    assert decision.reason == "controlled_allocation_failed"
    assert transaction.snapshot() == before


def test_cached_success_decision_allocation_failure_preserves_snapshot(
    monkeypatch,
):
    inner, scale, outer = _positive_case()
    transaction = V2Transaction()
    _build(inner, scale, outer, transaction)
    before = transaction.snapshot()
    original_decision_type = v2.BuildDecisionV2

    def injected_decision(*args, **kwargs):
        triggered = args[0] if args else kwargs["triggered"]
        if triggered:
            raise MemoryError("injected cached success decision allocation")
        return original_decision_type(*args, **kwargs)

    monkeypatch.setattr(v2, "BuildDecisionV2", injected_decision)
    decision = ComposedConv2DStencilCandidateV2.try_build(
        inner,
        (scale,),
        outer,
        _gate(outer, transaction),
    )
    assert not decision.triggered
    assert decision.reason == "controlled_allocation_failed"
    assert transaction.snapshot() == before


def test_reserve_rechecks_cumulative_physical_state_after_preview(monkeypatch):
    inner_a, scale_a, outer_a = _positive_case()
    inner_b, scale_b, outer_b = _positive_case()
    inner_b._kernel[0, 0, 0, 0] += 0.125

    candidate_a, _ = _build(inner_a, scale_a, outer_a)
    candidate_b, _ = _build(inner_b, scale_b, outer_b)
    assert candidate_a.content_key != candidate_b.content_key
    assert candidate_a.resident_bytes == candidate_b.resident_bytes
    resident = candidate_a.resident_bytes

    transaction = V2Transaction()
    physical_before = resident + 1

    def local_gate(outer):
        return GateRequestV2(
            selected_rows=np.arange(outer.shape[0], dtype=np.int64),
            reachable_before_bytes=physical_before,
            reachable_after_other_bytes=0,
            transaction=transaction,
            limits=_relaxed_limits(),
        )

    original_reserve = transaction.reserve_descriptor
    inserted = []

    def insert_b_then_reserve(**kwargs):
        if not inserted:
            inserted.append(True)
            b_decision = ComposedConv2DStencilCandidateV2.try_build(
                inner_b,
                (scale_b,),
                outer_b,
                local_gate(outer_b),
            )
            assert b_decision.triggered, b_decision.reason
        return original_reserve(**kwargs)

    monkeypatch.setattr(
        transaction, "reserve_descriptor", insert_b_then_reserve
    )
    a_decision = ComposedConv2DStencilCandidateV2.try_build(
        inner_a,
        (scale_a,),
        outer_a,
        local_gate(outer_a),
    )
    assert not a_decision.triggered
    assert a_decision.reason == "physical_metric_not_reduced"
    after = transaction.snapshot()
    assert len(after.compiled_content_keys) == 1
    assert candidate_b.content_key in after.compiled_content_keys
    assert after.descriptor_resident_used_bytes == resident


def test_baseexception_after_reservation_always_rolls_back(monkeypatch):
    inner, scale, outer = _positive_case()
    transaction = V2Transaction()
    before = transaction.snapshot()

    def interrupt(_cls, **_kwargs):
        raise KeyboardInterrupt("audit interrupt")

    monkeypatch.setattr(
        ComposedConv2DStencilCandidateV2,
        "_compile_group_intersections",
        classmethod(interrupt),
    )
    with pytest.raises(KeyboardInterrupt, match="audit interrupt"):
        ComposedConv2DStencilCandidateV2.try_build(
            inner,
            (scale,),
            outer,
            _gate(outer, transaction),
        )
    assert transaction.snapshot() == before


def test_baseexception_between_reserve_return_and_assignment_rolls_back(
    monkeypatch,
):
    inner, scale, outer = _positive_case()
    transaction = V2Transaction()
    before = transaction.snapshot()
    original_reserve = transaction.reserve_descriptor

    def reserve_then_interrupt(**kwargs):
        result = original_reserve(**kwargs)
        assert result[0] is kwargs["request"]
        raise KeyboardInterrupt("audit reserve handoff interrupt")

    monkeypatch.setattr(
        transaction, "reserve_descriptor", reserve_then_interrupt
    )
    with pytest.raises(
        KeyboardInterrupt, match="audit reserve handoff interrupt"
    ):
        ComposedConv2DStencilCandidateV2.try_build(
            inner,
            (scale,),
            outer,
            _gate(outer, transaction),
        )
    assert transaction.snapshot() == before


def test_mutated_implicit_payload_gets_a_new_snapshot_key_not_stale_cache():
    inner, scale, outer = _positive_case()
    transaction = V2Transaction()
    first, _ = _build(inner, scale, outer, transaction)
    constructor_key = inner.content_key
    probe = np.linspace(-0.5, 0.5, first.shape[1])
    first_value = first.matvec(probe)

    inner._kernel[0, 0, 0, 0] += 0.125
    assert inner.content_key == constructor_key  # Demonstrates the stale source key.
    second, decision = _build(inner, scale, outer, transaction)
    assert not decision.reused_descriptor
    assert second is not first
    assert second.content_key != first.content_key
    assert np.array_equal(first.matvec(probe), first_value)

    scale_full = np.broadcast_to(
        scale.reshape(1, scale.size, 1, 1), inner.output_shape
    ).reshape(-1)
    reference = (
        outer.to_csr_reference()
        @ sp.diags(scale_full, format="csr")
        @ inner.to_csr_reference()
    )
    rows = np.arange(second.shape[0], dtype=np.int64)
    _assert_same(second.gather_rows(rows, max_nnz=10**7), reference)


def test_general_left_product_cap_is_checked_before_sparse_multiply(monkeypatch):
    inner, scale, outer = _positive_case()
    candidate, _ = _build(inner, scale, outer)
    support = np.array([0, 1], dtype=np.int64)
    gathered_nnz = candidate.gather_rows(support, max_nnz=10**7).nnz
    output_rows = 4
    q = sp.csr_matrix(
        (
            np.ones(output_rows * support.size),
            (
                np.repeat(np.arange(output_rows), support.size),
                np.tile(support, output_rows),
            ),
        ),
        shape=(output_rows, candidate.shape[0]),
    )

    def forbidden_product(_self, _other):
        raise AssertionError("sparse product allocated before cap check")

    monkeypatch.setattr(sp.csr_matrix, "__matmul__", forbidden_product)
    with pytest.raises(
        CandidateV2Reject, match="left_product_contribution_limit"
    ):
        candidate.left_compose(q, max_nnz=gathered_nnz)


def test_seeded_group_geometry_sweep_matches_explicit_csr_oracle():
    rng = np.random.default_rng(8312026)
    configurations = []
    for input_channels in (1, 2, 4, 6):
        for middle_channels in (2, 4, 6, 8, 12):
            for output_channels in (2, 3, 4, 6, 8):
                for inner_groups in range(
                    1, min(input_channels, middle_channels) + 1
                ):
                    if input_channels % inner_groups or middle_channels % inner_groups:
                        continue
                    for outer_groups in range(
                        1, min(middle_channels, output_channels) + 1
                    ):
                        if (
                            middle_channels % outer_groups
                            or output_channels % outer_groups
                        ):
                            continue
                        configurations.append(
                            (
                                input_channels,
                                middle_channels,
                                output_channels,
                                inner_groups,
                                outer_groups,
                            )
                        )
    rng.shuffle(configurations)

    checked = 0
    for case, (cin, middle, cout, inner_groups, outer_groups) in enumerate(
        configurations[:64]
    ):
        batch = 1 + (case % 5 == 0)
        height, width = 4 + case % 3, 5 + (case // 3) % 3
        inner_kernel_shape = (1 + case % 3, 1 + (case // 2) % 3)
        outer_kernel_shape = (1 + (case // 3) % 3, 1 + (case // 5) % 3)
        inner_kernel = rng.integers(
            -4,
            5,
            size=(middle, cin // inner_groups, *inner_kernel_shape),
        ).astype(np.float64) / 8.0
        try:
            inner = ImplicitConv2DOp(
                inner_kernel,
                (batch, cin, height, width),
                stride=(1 + case % 2, 1 + (case // 7) % 2),
                padding=(case % 2, (case // 4) % 2),
                dilation=(1 + (case // 11) % 2, 1),
                groups=inner_groups,
            )
        except ValueError:
            continue
        if min(inner.output_shape[2:]) <= 0:
            continue

        outer_kernel = rng.integers(
            -4,
            5,
            size=(cout, middle // outer_groups, *outer_kernel_shape),
        ).astype(np.float64) / 8.0
        try:
            outer = ImplicitConv2DOp(
                outer_kernel,
                inner.output_shape,
                stride=(1 + (case // 13) % 2, 1),
                padding=(case % 2, (case // 6) % 2),
                dilation=(1, 1 + (case // 17) % 2),
                groups=outer_groups,
            )
        except ValueError:
            continue
        if min(outer.output_shape[2:]) <= 0:
            continue

        scale = rng.integers(-3, 4, size=middle).astype(np.float64) / 4.0
        candidate, _ = _build(inner, scale, outer)
        rows = np.arange(candidate.shape[0], dtype=np.int64)
        actual = candidate.gather_rows(rows, max_nnz=10**8)
        scale_full = np.broadcast_to(
            scale.reshape(1, middle, 1, 1), inner.output_shape
        ).reshape(-1)
        reference = (
            outer.to_csr_reference()
            @ sp.diags(scale_full, format="csr")
            @ inner.to_csr_reference()
        )
        _assert_same(actual, reference)
        checked += 1

    assert checked == 58
