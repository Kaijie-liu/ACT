"""Adversarial tests for the isolated S0-C2 CSR artifact transaction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

import experiments.neural_hz_20260831.s0_c2_emission_artifact_transaction_prototype as artifact_module
from experiments.neural_hz_20260831.s0_c2_emission_artifact_transaction_prototype import (
    ArtifactLimits,
    C2_ARTIFACT_NO_CLAIMS,
    C2ArtifactReject,
    C2EmissionArtifactTransaction,
)


class _TrustedTestAdapter:
    """Private stand-in; deliberately not evidence of V2 instrumentation."""

    def __init__(self, descriptor, *, work=3, peak=0):
        self.descriptor = sp.csr_matrix(descriptor, dtype=np.float64)
        self.work = work
        self.peak = peak
        self.calls = 0
        self.fail_on_call: dict[int, BaseException] = {}

    def __call__(self, q, recorder):
        self.calls += 1
        failure = self.fail_on_call.get(self.calls)
        if failure is not None:
            raise failure
        result = q @ self.descriptor
        return recorder.handoff(
            result,
            instrumented_work=self.work,
            peak_transient_bytes=self.peak,
        )


def _q_first():
    # Duplicates and unsorted coordinates canonicalize deterministically.
    return sp.coo_matrix(
        (
            np.array([2.0, 1.0, -0.5, 0.5]),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 1, 1])),
        ),
        shape=(2, 2),
    )


def _q_second():
    return sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))


def _keys(*, descriptor=("current-descriptor", "a"), prefix=("reverse", 0)):
    return dict(
        descriptor_current_key=descriptor,
        support_key=("rows", 0, 1),
        reverse_prefix_semantic_key=prefix,
    )


def _new_transaction(adapter, **kwargs):
    transaction = C2EmissionArtifactTransaction(adapter, **kwargs)
    token = transaction.begin_request()
    return transaction, token


def test_real_csr_is_owned_canonical_readonly_and_actual_footprint_recorded():
    adapter = _TrustedTestAdapter([[1.0, 2.0], [0.0, -1.0]], work=7)
    transaction, token = _new_transaction(adapter)

    artifact = transaction.materialize_left(token, q=_q_first(), **_keys())

    expected = _q_first().tocsr() @ adapter.descriptor
    assert isinstance(artifact.matrix, sp.csr_matrix)
    assert artifact.matrix.has_canonical_format
    np.testing.assert_allclose(artifact.matrix.toarray(), expected.toarray())
    for matrix in (artifact.q, artifact.matrix):
        assert matrix.dtype == np.float64
        for buffer in (matrix.data, matrix.indices, matrix.indptr):
            assert buffer.flags.owndata
            assert not buffer.flags.writeable
    assert artifact.actual_nnz == artifact.matrix.nnz
    assert artifact.actual_buffer_entries == (
        artifact.matrix.data.size
        + artifact.matrix.indices.size
        + artifact.matrix.indptr.size
    )
    assert artifact.actual_buffer_bytes == (
        artifact.matrix.data.nbytes
        + artifact.matrix.indices.nbytes
        + artifact.matrix.indptr.nbytes
    )
    assert artifact.q_digest == artifact.key[3]
    assert artifact.no_claims == C2_ARTIFACT_NO_CLAIMS
    assert "not_connected_to_v2" in " ".join(artifact.no_claims)


def test_same_complete_key_and_equivalent_q_returns_identical_strong_artifact():
    adapter = _TrustedTestAdapter(np.eye(2), work=11)
    transaction, token = _new_transaction(adapter)
    first = transaction.materialize_left(token, q=_q_first(), **_keys())
    before = transaction.snapshot()

    # A separately allocated canonical equivalent has the same full digest.
    equivalent = sp.csr_matrix(_q_first())
    second = transaction.materialize_left(token, q=equivalent, **_keys())
    after = transaction.snapshot()

    assert second is first
    assert adapter.calls == 1
    assert after.cumulative_work == before.cumulative_work == 11
    assert after.artifact_count == 1
    assert after.version == before.version


def test_same_support_different_q_never_aliases():
    adapter = _TrustedTestAdapter(np.eye(2), work=5)
    transaction, token = _new_transaction(adapter)
    first = transaction.materialize_left(token, q=_q_first(), **_keys())
    second = transaction.materialize_left(token, q=_q_second(), **_keys())

    assert second is not first
    assert first.key[:3] == second.key[:3]
    assert first.q_digest != second.q_digest
    snapshot = transaction.snapshot()
    assert snapshot.artifact_count == 2
    assert snapshot.cumulative_work == 10


def test_descriptor_and_reverse_prefix_are_independent_key_components():
    adapter = _TrustedTestAdapter(np.eye(2), work=2)
    transaction, token = _new_transaction(adapter)
    base = transaction.materialize_left(token, q=_q_first(), **_keys())
    descriptor_changed = transaction.materialize_left(
        token,
        q=_q_first(),
        **_keys(descriptor=("current-descriptor", "b")),
    )
    prefix_changed = transaction.materialize_left(
        token,
        q=_q_first(),
        **_keys(prefix=("reverse", 1)),
    )
    support_keys = _keys()
    support_keys["support_key"] = ("rows", 1)
    support_changed = transaction.materialize_left(
        token, q=_q_first(), **support_keys
    )

    assert len(
        {
            id(base),
            id(descriptor_changed),
            id(prefix_changed),
            id(support_changed),
        }
    ) == 4
    assert base.key[0] != descriptor_changed.key[0]
    assert base.key[2] != prefix_changed.key[2]
    assert base.key[1] != support_changed.key[1]
    assert transaction.snapshot().cumulative_work == 8


def test_readonly_buffer_and_digest_revalidation_detect_forced_corruption():
    adapter = _TrustedTestAdapter(np.eye(2))
    transaction, token = _new_transaction(adapter)
    artifact = transaction.materialize_left(token, q=_q_first(), **_keys())

    with pytest.raises(ValueError):
        artifact.matrix.data[0] += 1.0
    artifact.matrix.data.flags.writeable = True
    artifact.matrix.data[0] += 1.0
    artifact.matrix.data.flags.writeable = False

    with pytest.raises(C2ArtifactReject, match="digest_corrupt"):
        transaction.snapshot()
    with pytest.raises(C2ArtifactReject, match="digest_corrupt"):
        transaction.materialize_left(token, q=_q_first(), **_keys())


def test_cumulative_work_accepts_exact_limit_then_rejects_next_artifact():
    adapter = _TrustedTestAdapter(np.eye(2), work=5)
    transaction, token = _new_transaction(
        adapter, limits=ArtifactLimits(max_transaction_work=10)
    )
    transaction.materialize_left(token, q=_q_first(), **_keys(prefix=("p", 0)))
    transaction.materialize_left(token, q=_q_first(), **_keys(prefix=("p", 1)))
    at_boundary = transaction.snapshot()
    assert at_boundary.cumulative_work == 10
    assert at_boundary.artifact_count == 2

    with pytest.raises(C2ArtifactReject, match="transaction_work_limit"):
        transaction.materialize_left(
            token, q=_q_first(), **_keys(prefix=("p", 2))
        )
    after = transaction.snapshot()
    assert after == at_boundary
    assert adapter.calls == 3


def test_second_artifact_memoryerror_rolls_back_without_phantom_key_credit():
    adapter = _TrustedTestAdapter(np.eye(2), work=4)
    adapter.fail_on_call[2] = MemoryError("injected")
    transaction, token = _new_transaction(adapter)
    first = transaction.materialize_left(
        token, q=_q_first(), **_keys(prefix=("first",))
    )
    before = transaction.snapshot()

    with pytest.raises(MemoryError, match="injected"):
        transaction.materialize_left(
            token, q=_q_first(), **_keys(prefix=("second",))
        )
    after_failure = transaction.snapshot()
    assert after_failure == before
    assert after_failure.artifact_count == 1
    assert transaction.materialize_left(
        token, q=_q_first(), **_keys(prefix=("first",))
    ) is first

    # Removing the injection makes the same missing key call the materializer
    # again.  A key from a failed call earned no accounting exemption.
    adapter.fail_on_call.clear()
    second = transaction.materialize_left(
        token, q=_q_first(), **_keys(prefix=("second",))
    )
    assert second is not first
    assert adapter.calls == 3
    assert transaction.snapshot().cumulative_work == 8


def test_baseexception_at_post_commit_handoff_cleans_publication_and_reraises():
    class HandoffAbort(BaseException):
        pass

    calls = 0

    def abort_once(_artifact):
        nonlocal calls
        calls += 1
        raise HandoffAbort("handoff")

    adapter = _TrustedTestAdapter(np.eye(2), work=9)
    transaction, token = _new_transaction(
        adapter, _post_commit_handoff_hook=abort_once
    )
    before = transaction.snapshot()

    with pytest.raises(HandoffAbort, match="handoff"):
        transaction.materialize_left(token, q=_q_first(), **_keys())

    after = transaction.snapshot()
    assert after == before
    assert after.artifact_count == 0
    assert after.cumulative_work == 0
    assert not after.build_open
    assert calls == 1


@pytest.mark.parametrize("exception_type", [MemoryError, KeyboardInterrupt])
def test_invocation_nonce_failure_never_leaves_build_open(
    monkeypatch, exception_type
):
    adapter = _TrustedTestAdapter(np.eye(2), work=9)
    transaction, token = _new_transaction(adapter)
    before = transaction.snapshot()

    def fail_nonce(_size):
        raise exception_type("nonce allocation interrupt")

    monkeypatch.setattr(artifact_module.secrets, "token_bytes", fail_nonce)
    with pytest.raises(exception_type, match="nonce allocation interrupt"):
        transaction.materialize_left(token, q=_q_first(), **_keys())

    assert transaction.snapshot() == before
    assert adapter.calls == 0


def test_foreign_token_is_rejected_before_callback_or_state_change():
    first_adapter = _TrustedTestAdapter(np.eye(2))
    second_adapter = _TrustedTestAdapter(np.eye(2))
    first, first_token = _new_transaction(first_adapter)
    second, second_token = _new_transaction(second_adapter)

    with pytest.raises(C2ArtifactReject, match="foreign_request_token"):
        first.materialize_left(second_token, q=_q_first(), **_keys())
    assert first_adapter.calls == 0
    assert first.snapshot().artifact_count == 0
    assert first_token is not second_token


def test_forged_or_callback_mismatched_ledger_is_rejected_atomically():
    class ForgingAdapter(_TrustedTestAdapter):
        def __call__(self, q, recorder):
            self.calls += 1
            result = q @ self.descriptor
            valid = recorder.handoff(
                result, instrumented_work=3, peak_transient_bytes=0
            )
            forged = replace(valid.ledger, instrumented_work=0)
            return replace(valid, ledger=forged)

    adapter = ForgingAdapter(np.eye(2))
    transaction, token = _new_transaction(adapter)

    with pytest.raises(C2ArtifactReject, match="authentication_failed"):
        transaction.materialize_left(token, q=_q_first(), **_keys())
    snapshot = transaction.snapshot()
    assert snapshot.artifact_count == 0
    assert snapshot.cumulative_work == 0


def test_nnz_and_transient_limits_are_per_artifact_and_inclusive():
    # This 2x2 identity has two nnz.  Exact nnz and exact conservative
    # transient bounds are accepted; one-less limits reject atomically.
    adapter = _TrustedTestAdapter(np.eye(2), work=1)
    generous, token = _new_transaction(adapter)
    probe = generous.materialize_left(token, q=_q_second(), **_keys())
    nnz = probe.actual_nnz
    transient = probe.controlled_transient_upper_bytes

    equal_adapter = _TrustedTestAdapter(np.eye(2), work=1)
    equal, equal_token = _new_transaction(
        equal_adapter,
        limits=ArtifactLimits(
            max_result_nnz=nnz,
            max_transient_bytes=transient,
        ),
    )
    equal.materialize_left(equal_token, q=_q_second(), **_keys())
    assert equal.snapshot().artifact_count == 1

    nnz_adapter = _TrustedTestAdapter(np.eye(2), work=1)
    nnz_tx, nnz_token = _new_transaction(
        nnz_adapter, limits=ArtifactLimits(max_result_nnz=nnz - 1)
    )
    with pytest.raises(C2ArtifactReject, match="artifact_nnz_limit"):
        nnz_tx.materialize_left(nnz_token, q=_q_second(), **_keys())
    assert nnz_tx.snapshot().artifact_count == 0

    transient_adapter = _TrustedTestAdapter(np.eye(2), work=1)
    transient_tx, transient_token = _new_transaction(
        transient_adapter,
        limits=ArtifactLimits(max_transient_bytes=transient - 1),
    )
    with pytest.raises(C2ArtifactReject, match="artifact_transient_limit"):
        transient_tx.materialize_left(
            transient_token, q=_q_second(), **_keys()
        )
    assert transient_tx.snapshot().artifact_count == 0


def test_seal_revalidates_then_closes_request_without_losing_artifacts():
    adapter = _TrustedTestAdapter(np.eye(2), work=6)
    transaction, token = _new_transaction(adapter)
    artifact = transaction.materialize_left(token, q=_q_first(), **_keys())

    sealed = transaction.seal(token)
    assert sealed.sealed
    assert not sealed.request_open
    assert sealed.artifact_count == 1
    assert sealed.cumulative_work == 6
    assert transaction.snapshot() == sealed
    assert artifact is not None
    with pytest.raises(C2ArtifactReject, match="transaction_sealed"):
        transaction.materialize_left(token, q=_q_first(), **_keys())
    with pytest.raises(C2ArtifactReject, match="already_sealed"):
        transaction.seal(token)


def test_callback_cannot_return_plain_matrix_or_caller_ledger():
    calls = 0

    def plain_callback(q, _recorder):
        nonlocal calls
        calls += 1
        return q.copy()

    transaction, token = _new_transaction(plain_callback)
    with pytest.raises(C2ArtifactReject, match="callback_handoff_type"):
        transaction.materialize_left(token, q=_q_first(), **_keys())
    assert calls == 1
    assert transaction.snapshot().artifact_count == 0
