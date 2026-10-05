"""Request-local real-CSR artifact transaction for isolated S0-C2 work.

This module is deliberately disconnected from ACT production and from the V2
descriptor compiler.  It closes only the first ownership/accounting layer for
future materialization: a request owns canonical input ``Q`` and output CSR
buffers, keys a strong-reference cache by every semantic input, and charges
instrumented emission work only when a real artifact is committed.

The descriptor callback is a trust boundary.  It is bound to the transaction,
receives an invocation-local recorder, and must return the recorder's
authenticated handoff.  This prevents a ``materialize_left`` caller from
supplying a ledger, but it does *not* prove that the callback measured work
correctly.  The isolated tests use a private trusted adapter.  Until an audited
V2 materializer performs instrumented emission through this protocol, its
ledger is not formal evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import operator
import secrets
import threading
from typing import Callable, Final, TypeAlias

import numpy as np
import scipy.sparse as sp


GIB: Final = 1024 * 1024 * 1024

C2_ARTIFACT_NO_CLAIMS: Final = (
    "isolated_prototype_is_not_imported_by_act_production",
    "trusted_callback_is_not_connected_to_v2_instrumented_emission",
    "callback_instrumented_work_is_not_independent_formal_evidence",
    "descriptor_current_key_is_supplied_not_revalidated_against_live_v2",
    "support_and_reverse_prefix_keys_are_supplied_semantic_snapshots",
    "no_whole_state_bytes_entries_rss_or_four_concurrent_gate_is_proven",
    "no_tiny_iid143_family_full2413_or_formal_gain_is_claimed",
)


class C2ArtifactReject(ValueError):
    """Stable fail-closed rejection by the isolated artifact layer."""


@dataclass(frozen=True)
class ArtifactLimits:
    """Frozen default request and per-artifact gates (equality is allowed)."""

    max_transaction_work: int = 256_000_000
    max_result_nnz: int = 64_000_000
    max_transient_bytes: int = GIB


# Each semantic value is encoded rather than retained as a mutable/hash-
# surprising Python object.  The four tuple positions remain independently
# tagged and therefore cannot alias one another.
ArtifactKey: TypeAlias = tuple[bytes, bytes, bytes, str]


@dataclass(frozen=True)
class TrustedEmissionLedger:
    """Authenticated callback handoff, valid for exactly one invocation.

    Instances accepted by the transaction can only be minted by the recorder
    passed to its bound callback.  The MAC is intentionally not a claim that
    the callback's instrumentation is correct; that requires the future
    audited V2 emission integration.
    """

    invocation_id: bytes
    q_digest: str
    result_object_id: int
    instrumented_work: int
    callback_peak_transient_bytes: int
    authentication_tag: bytes


@dataclass(frozen=True)
class TrustedCallbackHandoff:
    """The only callback result accepted by ``materialize_left``."""

    matrix: object
    ledger: TrustedEmissionLedger


@dataclass(frozen=True)
class EmissionArtifact:
    """Strongly owned immutable-buffer CSR artifact."""

    key: ArtifactKey
    q: sp.csr_matrix
    matrix: sp.csr_matrix
    q_digest: str
    matrix_digest: str
    ledger: TrustedEmissionLedger
    actual_nnz: int
    actual_buffer_entries: int
    actual_buffer_bytes: int
    retained_buffer_bytes: int
    controlled_transient_upper_bytes: int
    charged_work: int
    no_claims: tuple[str, ...] = C2_ARTIFACT_NO_CLAIMS


@dataclass(frozen=True)
class ArtifactTransactionSnapshot:
    """Revalidated committed state; it never contains key-only cache credit."""

    artifact_keys: frozenset[ArtifactKey]
    artifact_count: int
    cumulative_work: int
    cumulative_nnz: int
    cumulative_buffer_entries: int
    cumulative_buffer_bytes: int
    cumulative_retained_buffer_bytes: int
    transient_live_bytes: int
    transient_peak_bytes: int
    request_open: bool
    build_open: bool
    sealed: bool
    version: int
    no_claims: tuple[str, ...] = C2_ARTIFACT_NO_CLAIMS


class _RequestToken:
    """Identity-only request capability; intentionally has no public fields."""

    __slots__ = ()


def _strict_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise C2ArtifactReject(f"{name}_not_integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise C2ArtifactReject(f"{name}_not_integer") from exc
    if result < 0:
        raise C2ArtifactReject(f"{name}_negative")
    return result


def _stable_key_payload(value, *, depth: int = 0) -> bytes:
    """Encode a closed recursively immutable key vocabulary."""

    if depth > 32:
        raise C2ArtifactReject("semantic_key_too_deep")
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise C2ArtifactReject("semantic_key_not_stable")
    if isinstance(value, (int, np.integer)):
        payload = str(int(value)).encode("ascii")
        return b"I" + len(payload).to_bytes(8, "big") + payload
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"S" + len(payload).to_bytes(8, "big") + payload
    if isinstance(value, bytes):
        return b"Y" + len(value).to_bytes(8, "big") + value
    if isinstance(value, tuple):
        pieces = [_stable_key_payload(item, depth=depth + 1) for item in value]
        return b"T" + len(pieces).to_bytes(8, "big") + b"".join(
            len(piece).to_bytes(8, "big") + piece for piece in pieces
        )
    raise C2ArtifactReject("semantic_key_not_stable")


def _raw_csr_buffer_bytes(matrix: sp.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _csr_buffer_entries(matrix: sp.csr_matrix) -> int:
    return int(matrix.data.size + matrix.indices.size + matrix.indptr.size)


def _canonical_owned_float64_csr(value, *, name: str) -> sp.csr_matrix:
    """Return a canonical CSR whose three buffers are owned by this call."""

    if not sp.issparse(value):
        raise C2ArtifactReject(f"{name}_not_sparse")
    if getattr(value, "ndim", None) != 2:
        raise C2ArtifactReject(f"{name}_rank")
    raw_dtype = np.dtype(value.dtype)
    if raw_dtype.kind not in "fiu":
        raise C2ArtifactReject(f"{name}_not_real_numeric")

    try:
        temporary = sp.csr_matrix(value, dtype=np.float64, copy=True)
        temporary.sum_duplicates()
        temporary.eliminate_zeros()
        temporary.sort_indices()
    except (MemoryError, C2ArtifactReject):
        raise
    except Exception as exc:
        raise C2ArtifactReject(f"{name}_canonicalization_failed") from exc

    if temporary.data.size and not np.all(np.isfinite(temporary.data)):
        raise C2ArtifactReject(f"{name}_nonfinite")

    # Rebuild instead of trusting SciPy's copy flag.  Assignment to a fresh CSR
    # makes the ownership assertion observable and stable for later audits.
    data = np.array(temporary.data, dtype=np.float64, order="C", copy=True)
    indices = np.array(temporary.indices, dtype=np.int64, order="C", copy=True)
    indptr = np.array(temporary.indptr, dtype=np.int64, order="C", copy=True)
    owned = sp.csr_matrix(temporary.shape, dtype=np.float64)
    owned.data = data
    owned.indices = indices
    owned.indptr = indptr
    if not owned.has_canonical_format:
        raise C2ArtifactReject(f"{name}_not_canonical")
    if not all(
        array.flags.c_contiguous and array.flags.owndata
        for array in (owned.data, owned.indices, owned.indptr)
    ):
        raise RuntimeError(f"{name}_ownership_failure")
    for array in (owned.data, owned.indices, owned.indptr):
        array.flags.writeable = False
    return owned


def _csr_digest(matrix: sp.csr_matrix) -> str:
    """Stable digest of shape and normalized CSR indptr/indices/data."""

    shape = np.asarray(matrix.shape, dtype="<i8").tobytes(order="C")
    indptr = np.asarray(matrix.indptr, dtype="<i8", order="C").tobytes(
        order="C"
    )
    indices = np.asarray(matrix.indices, dtype="<i8", order="C").tobytes(
        order="C"
    )
    data = np.asarray(matrix.data, dtype="<f8", order="C").tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(b"s0-c2-owned-csr-v1")
    for payload in (shape, indptr, indices, data):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_frozen_csr(
    matrix: sp.csr_matrix, *, expected_digest: str, name: str
) -> None:
    if not isinstance(matrix, sp.csr_matrix):
        raise C2ArtifactReject(f"{name}_object_replaced")
    if matrix.dtype != np.dtype(np.float64):
        raise C2ArtifactReject(f"{name}_dtype_corrupt")
    if any(
        array.ndim != 1
        for array in (matrix.data, matrix.indices, matrix.indptr)
    ):
        raise C2ArtifactReject(f"{name}_format_corrupt")
    rows, columns = matrix.shape
    if (
        matrix.indptr.size != rows + 1
        or matrix.data.size != matrix.indices.size
        or matrix.indptr.size == 0
        or int(matrix.indptr[0]) != 0
        or int(matrix.indptr[-1]) != matrix.data.size
        or np.any(matrix.indptr[1:] < matrix.indptr[:-1])
        or np.any(matrix.indices < 0)
        or np.any(matrix.indices >= columns)
    ):
        raise C2ArtifactReject(f"{name}_format_corrupt")
    if matrix.indices.dtype != np.dtype(np.int64) or matrix.indptr.dtype != np.dtype(
        np.int64
    ):
        raise C2ArtifactReject(f"{name}_index_dtype_corrupt")
    if not matrix.has_canonical_format:
        raise C2ArtifactReject(f"{name}_canonical_corrupt")
    for array_name, array in (
        ("data", matrix.data),
        ("indices", matrix.indices),
        ("indptr", matrix.indptr),
    ):
        if array.flags.writeable:
            raise C2ArtifactReject(f"{name}_{array_name}_writeable")
        if not array.flags.c_contiguous or not array.flags.owndata:
            raise C2ArtifactReject(f"{name}_{array_name}_ownership_corrupt")
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise C2ArtifactReject(f"{name}_nonfinite_corrupt")
    if not hmac.compare_digest(_csr_digest(matrix), expected_digest):
        raise C2ArtifactReject(f"{name}_digest_corrupt")


def _ledger_message(
    *,
    invocation_id: bytes,
    q_digest: str,
    result_object_id: int,
    work: int,
    transient: int,
) -> bytes:
    pieces = (
        invocation_id,
        q_digest.encode("ascii"),
        str(result_object_id).encode("ascii"),
        str(work).encode("ascii"),
        str(transient).encode("ascii"),
    )
    return b"".join(len(piece).to_bytes(8, "big") + piece for piece in pieces)


class _TrustedLedgerRecorder:
    """Invocation-local one-shot authenticated ledger factory."""

    __slots__ = (
        "_authentication_secret",
        "_invocation_id",
        "_q",
        "_q_digest",
        "_sealed",
    )

    def __init__(
        self,
        *,
        authentication_secret: bytes,
        invocation_id: bytes,
        q: sp.csr_matrix,
        q_digest: str,
    ):
        self._authentication_secret = authentication_secret
        self._invocation_id = invocation_id
        self._q = q
        self._q_digest = q_digest
        self._sealed = False

    @property
    def q(self) -> sp.csr_matrix:
        """Exact owned/read-only Q that the callback must consume."""

        return self._q

    def handoff(
        self,
        matrix,
        *,
        instrumented_work: int,
        peak_transient_bytes: int,
    ) -> TrustedCallbackHandoff:
        if self._sealed:
            raise C2ArtifactReject("callback_ledger_already_sealed")
        work = _strict_nonnegative_int(
            instrumented_work, name="instrumented_work"
        )
        transient = _strict_nonnegative_int(
            peak_transient_bytes, name="callback_peak_transient_bytes"
        )
        result_object_id = id(matrix)
        message = _ledger_message(
            invocation_id=self._invocation_id,
            q_digest=self._q_digest,
            result_object_id=result_object_id,
            work=work,
            transient=transient,
        )
        tag = hmac.new(
            self._authentication_secret, message, hashlib.sha256
        ).digest()
        self._sealed = True
        ledger = TrustedEmissionLedger(
            invocation_id=self._invocation_id,
            q_digest=self._q_digest,
            result_object_id=result_object_id,
            instrumented_work=work,
            callback_peak_transient_bytes=transient,
            authentication_tag=tag,
        )
        return TrustedCallbackHandoff(matrix=matrix, ledger=ledger)


class C2EmissionArtifactTransaction:
    """One request-local atomic strong-reference artifact cache.

    ``trusted_descriptor_callback`` is bound once and called as
    ``callback(canonical_q, recorder)``.  It must build the real left
    materialization and return ``recorder.handoff(...)``.  Ordinary
    ``materialize_left`` callers cannot submit a matrix or ledger directly.
    """

    def __init__(
        self,
        trusted_descriptor_callback: Callable[
            [sp.csr_matrix, _TrustedLedgerRecorder], TrustedCallbackHandoff
        ],
        *,
        limits: ArtifactLimits = ArtifactLimits(),
        _post_commit_handoff_hook: Callable[[EmissionArtifact], None]
        | None = None,
    ):
        if not callable(trusted_descriptor_callback):
            raise C2ArtifactReject("trusted_descriptor_callback_not_callable")
        self._limits = ArtifactLimits(
            max_transaction_work=_strict_nonnegative_int(
                limits.max_transaction_work, name="max_transaction_work"
            ),
            max_result_nnz=_strict_nonnegative_int(
                limits.max_result_nnz, name="max_result_nnz"
            ),
            max_transient_bytes=_strict_nonnegative_int(
                limits.max_transient_bytes, name="max_transient_bytes"
            ),
        )
        self._callback = trusted_descriptor_callback
        self._post_commit_handoff_hook = _post_commit_handoff_hook
        self._lock = threading.RLock()
        self._authentication_secret = secrets.token_bytes(32)
        self._token: _RequestToken | None = None
        self._cache: dict[ArtifactKey, EmissionArtifact] = {}
        self._work_used = 0
        self._nnz_used = 0
        self._entries_used = 0
        self._buffer_bytes_used = 0
        self._retained_bytes_used = 0
        self._transient_peak_bytes = 0
        self._build_open = False
        self._sealed = False
        self._version = 0

    def begin_request(self) -> _RequestToken:
        with self._lock:
            if self._token is not None:
                raise C2ArtifactReject("request_already_begun")
            if self._sealed:
                raise C2ArtifactReject("transaction_sealed")
            self._token = _RequestToken()
            self._version += 1
            return self._token

    def _check_token_unlocked(self, token: object) -> None:
        if self._token is None:
            raise C2ArtifactReject("request_not_begun")
        if token is not self._token:
            raise C2ArtifactReject("foreign_request_token")

    def _validate_artifact_unlocked(self, artifact: EmissionArtifact) -> None:
        if self._cache.get(artifact.key) is not artifact:
            raise C2ArtifactReject("artifact_cache_identity_corrupt")
        _validate_frozen_csr(
            artifact.q, expected_digest=artifact.q_digest, name="artifact_q"
        )
        _validate_frozen_csr(
            artifact.matrix,
            expected_digest=artifact.matrix_digest,
            name="artifact_matrix",
        )
        if artifact.key[3] != artifact.q_digest:
            raise C2ArtifactReject("artifact_key_q_digest_corrupt")
        if artifact.actual_nnz != int(artifact.matrix.nnz):
            raise C2ArtifactReject("artifact_nnz_ledger_corrupt")
        if artifact.actual_buffer_entries != _csr_buffer_entries(
            artifact.matrix
        ):
            raise C2ArtifactReject("artifact_entries_ledger_corrupt")
        if artifact.actual_buffer_bytes != _raw_csr_buffer_bytes(
            artifact.matrix
        ):
            raise C2ArtifactReject("artifact_bytes_ledger_corrupt")
        retained = artifact.actual_buffer_bytes + _raw_csr_buffer_bytes(
            artifact.q
        )
        if artifact.retained_buffer_bytes != retained:
            raise C2ArtifactReject("artifact_retained_bytes_ledger_corrupt")
        if artifact.charged_work != artifact.ledger.instrumented_work:
            raise C2ArtifactReject("artifact_work_ledger_corrupt")
        self._verify_ledger_unlocked(
            artifact.ledger,
            q_digest=artifact.q_digest,
            result_object_id=artifact.ledger.result_object_id,
        )

    def _verify_ledger_unlocked(
        self,
        ledger: TrustedEmissionLedger,
        *,
        q_digest: str,
        result_object_id: int,
        invocation_id: bytes | None = None,
    ) -> None:
        if not isinstance(ledger, TrustedEmissionLedger):
            raise C2ArtifactReject("callback_ledger_type")
        if invocation_id is not None and ledger.invocation_id != invocation_id:
            raise C2ArtifactReject("callback_ledger_invocation_mismatch")
        if ledger.q_digest != q_digest:
            raise C2ArtifactReject("callback_ledger_q_mismatch")
        if ledger.result_object_id != result_object_id:
            raise C2ArtifactReject("callback_ledger_result_mismatch")
        work = _strict_nonnegative_int(
            ledger.instrumented_work, name="instrumented_work"
        )
        transient = _strict_nonnegative_int(
            ledger.callback_peak_transient_bytes,
            name="callback_peak_transient_bytes",
        )
        message = _ledger_message(
            invocation_id=ledger.invocation_id,
            q_digest=q_digest,
            result_object_id=result_object_id,
            work=work,
            transient=transient,
        )
        expected = hmac.new(
            self._authentication_secret, message, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, ledger.authentication_tag):
            raise C2ArtifactReject("callback_ledger_authentication_failed")

    def _snapshot_unlocked(self) -> ArtifactTransactionSnapshot:
        work = 0
        nnz = 0
        entries = 0
        buffer_bytes = 0
        retained_bytes = 0
        for key, artifact in self._cache.items():
            if artifact.key != key:
                raise C2ArtifactReject("artifact_key_corrupt")
            self._validate_artifact_unlocked(artifact)
            work += artifact.charged_work
            nnz += artifact.actual_nnz
            entries += artifact.actual_buffer_entries
            buffer_bytes += artifact.actual_buffer_bytes
            retained_bytes += artifact.retained_buffer_bytes
        if work != self._work_used:
            raise C2ArtifactReject("transaction_work_ledger_corrupt")
        if nnz != self._nnz_used:
            raise C2ArtifactReject("transaction_nnz_ledger_corrupt")
        if entries != self._entries_used:
            raise C2ArtifactReject("transaction_entries_ledger_corrupt")
        if buffer_bytes != self._buffer_bytes_used:
            raise C2ArtifactReject("transaction_bytes_ledger_corrupt")
        if retained_bytes != self._retained_bytes_used:
            raise C2ArtifactReject("transaction_retained_ledger_corrupt")
        return ArtifactTransactionSnapshot(
            artifact_keys=frozenset(self._cache),
            artifact_count=len(self._cache),
            cumulative_work=self._work_used,
            cumulative_nnz=self._nnz_used,
            cumulative_buffer_entries=self._entries_used,
            cumulative_buffer_bytes=self._buffer_bytes_used,
            cumulative_retained_buffer_bytes=self._retained_bytes_used,
            transient_live_bytes=0,
            transient_peak_bytes=self._transient_peak_bytes,
            request_open=self._token is not None and not self._sealed,
            build_open=self._build_open,
            sealed=self._sealed,
            version=self._version,
        )

    def snapshot(self) -> ArtifactTransactionSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def materialize_left(
        self,
        token: object,
        *,
        descriptor_current_key,
        support_key,
        reverse_prefix_semantic_key,
        q,
    ) -> EmissionArtifact:
        """Atomically return/cache one real CSR artifact.

        A hit is possible only when the cache owns the complete artifact.  It
        returns that exact object and performs zero callback/emission work.
        Any exception, including ``MemoryError`` or another ``BaseException``,
        removes only this call's partial publication and restores all ledgers.
        """

        with self._lock:
            self._check_token_unlocked(token)
            if self._sealed:
                raise C2ArtifactReject("transaction_sealed")
            if self._build_open:
                raise C2ArtifactReject("recursive_materialization")

            descriptor_payload = _stable_key_payload(descriptor_current_key)
            support_payload = _stable_key_payload(support_key)
            prefix_payload = _stable_key_payload(reverse_prefix_semantic_key)
            canonical_q = _canonical_owned_float64_csr(q, name="q")
            q_digest = _csr_digest(canonical_q)
            key: ArtifactKey = (
                descriptor_payload,
                support_payload,
                prefix_payload,
                q_digest,
            )

            cached = self._cache.get(key)
            if cached is not None:
                self._validate_artifact_unlocked(cached)
                return cached

            # No key-only ledger exists: a miss always invokes the bound
            # materializer and can earn cache credit only by committing CSR.
            artifact: EmissionArtifact | None = None
            before_scalars = (
                self._work_used,
                self._nnz_used,
                self._entries_used,
                self._buffer_bytes_used,
                self._retained_bytes_used,
                self._transient_peak_bytes,
                self._version,
            )
            try:
                # Enter the guarded region before the first invocation-local
                # allocation.  Otherwise an interrupt while obtaining the
                # nonce could leave ``build_open`` stuck without reaching the
                # cleanup path.
                self._build_open = True
                invocation_id = secrets.token_bytes(32)
                recorder = _TrustedLedgerRecorder(
                    authentication_secret=self._authentication_secret,
                    invocation_id=invocation_id,
                    q=canonical_q,
                    q_digest=q_digest,
                )
                handoff = self._callback(canonical_q, recorder)
                if not isinstance(handoff, TrustedCallbackHandoff):
                    raise C2ArtifactReject("callback_handoff_type")
                raw_matrix = handoff.matrix
                ledger = handoff.ledger
                self._verify_ledger_unlocked(
                    ledger,
                    q_digest=q_digest,
                    result_object_id=id(raw_matrix),
                    invocation_id=invocation_id,
                )
                if not isinstance(raw_matrix, sp.csr_matrix):
                    raise C2ArtifactReject("callback_result_not_csr")
                raw_result_bytes = _raw_csr_buffer_bytes(raw_matrix)
                matrix = _canonical_owned_float64_csr(
                    raw_matrix, name="callback_result"
                )
                actual_nnz = int(matrix.nnz)
                if actual_nnz > self._limits.max_result_nnz:
                    raise C2ArtifactReject("artifact_nnz_limit")
                actual_entries = _csr_buffer_entries(matrix)
                actual_bytes = _raw_csr_buffer_bytes(matrix)
                q_bytes = _raw_csr_buffer_bytes(canonical_q)
                retained_bytes = q_bytes + actual_bytes
                # Conservative controlled-copy upper bound.  Callback-local
                # peak remains trusted instrumentation, not formal evidence.
                transient_upper = (
                    2 * q_bytes
                    + 2 * raw_result_bytes
                    + actual_bytes
                    + ledger.callback_peak_transient_bytes
                )
                if transient_upper > self._limits.max_transient_bytes:
                    raise C2ArtifactReject("artifact_transient_limit")
                projected_work = self._work_used + ledger.instrumented_work
                if projected_work > self._limits.max_transaction_work:
                    raise C2ArtifactReject("transaction_work_limit")

                matrix_digest = _csr_digest(matrix)
                artifact = EmissionArtifact(
                    key=key,
                    q=canonical_q,
                    matrix=matrix,
                    q_digest=q_digest,
                    matrix_digest=matrix_digest,
                    ledger=ledger,
                    actual_nnz=actual_nnz,
                    actual_buffer_entries=actual_entries,
                    actual_buffer_bytes=actual_bytes,
                    retained_buffer_bytes=retained_bytes,
                    controlled_transient_upper_bytes=transient_upper,
                    charged_work=ledger.instrumented_work,
                )
                _validate_frozen_csr(
                    artifact.q,
                    expected_digest=artifact.q_digest,
                    name="artifact_q",
                )
                _validate_frozen_csr(
                    artifact.matrix,
                    expected_digest=artifact.matrix_digest,
                    name="artifact_matrix",
                )

                self._cache[key] = artifact
                self._work_used = projected_work
                self._nnz_used += actual_nnz
                self._entries_used += actual_entries
                self._buffer_bytes_used += actual_bytes
                self._retained_bytes_used += retained_bytes
                self._transient_peak_bytes = max(
                    self._transient_peak_bytes, transient_upper
                )
                self._version += 1

                # The hook is private test instrumentation for an asynchronous
                # exception at the commit-to-caller handoff boundary.
                if self._post_commit_handoff_hook is not None:
                    self._post_commit_handoff_hook(artifact)
                return artifact
            except BaseException:
                # Do not depend on a post-insertion flag: an asynchronous
                # exception may land between dict publication and that flag.
                if artifact is not None and self._cache.get(key) is artifact:
                    del self._cache[key]
                (
                    self._work_used,
                    self._nnz_used,
                    self._entries_used,
                    self._buffer_bytes_used,
                    self._retained_bytes_used,
                    self._transient_peak_bytes,
                    self._version,
                ) = before_scalars
                raise
            finally:
                self._build_open = False

    def seal(self, token: object) -> ArtifactTransactionSnapshot:
        with self._lock:
            self._check_token_unlocked(token)
            if self._sealed:
                raise C2ArtifactReject("transaction_already_sealed")
            if self._build_open:
                raise C2ArtifactReject("build_open")
            # Validate before changing the state so corruption cannot be
            # blessed by sealing.
            self._snapshot_unlocked()
            self._sealed = True
            self._version += 1
            return self._snapshot_unlocked()


__all__ = (
    "ArtifactLimits",
    "ArtifactTransactionSnapshot",
    "C2_ARTIFACT_NO_CLAIMS",
    "C2ArtifactReject",
    "C2EmissionArtifactTransaction",
    "EmissionArtifact",
    "TrustedCallbackHandoff",
    "TrustedEmissionLedger",
)
