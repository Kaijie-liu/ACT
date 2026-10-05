"""Isolated production-shaped composed Conv2D stencil candidate V2.

This module is an experiment-only implementation.  No production module
imports it and it does not alter Hybrid-Z execution.  It contracts two
``ImplicitConv2DOp`` descriptors separated only by exact, batch/spatially
stationary channel scales.  Group intersections are compiled by ascending
global middle-channel rank-one updates; neither input convolution is expanded
to a full-channel tap tensor or spatial CSR.

The implementation carries the frozen per-descriptor gate, an opaque
descriptor-compilation transaction with rollback, typed binary content
identity, and separate logical/resident/prospective-materialization ledgers.
It deliberately does not claim that emission occurred: a real materializer
must charge actual CSR construction or retain the cached artifact. Bias,
residual ancestry, predicates, frames, slots, and witnesses remain outside
this linear operator and therefore cannot be changed by it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import operator
import struct
import threading
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)


MIB = 1024 * 1024
GIB = 1024 * MIB
ALGORITHM_KEY = (
    "group_intersection_rank1",
    "ascending_global_middle_channel",
    2,
)


class CandidateV2Reject(ValueError):
    """A stable, deterministic, fail-closed rejection."""


@dataclass(frozen=True)
class FrozenV1Limits:
    max_descriptor_contraction_products: int = 200_000_000
    max_transaction_work: int = 256_000_000
    max_coefficient_entries: int = 2_000_000
    max_resident_bytes: int = 64 * MIB
    max_transient_bytes: int = GIB
    max_result_nnz: int = 64_000_000
    max_work_numerator: int = 1
    max_work_denominator: int = 4


@dataclass(frozen=True)
class ResidentBreakdown:
    coefficient_bytes: int
    block_metadata_bytes: int
    outer_group_indptr_bytes: int
    reachable_input_count_bytes: int
    outer_row_mask_bytes: int

    @property
    def total_bytes(self) -> int:
        return int(
            self.coefficient_bytes
            + self.block_metadata_bytes
            + self.outer_group_indptr_bytes
            + self.reachable_input_count_bytes
            + self.outer_row_mask_bytes
        )


@dataclass(frozen=True)
class CompilationLedger:
    actual_contraction_products: int
    gate_formula_products: int
    intersection_formula_products: int
    sigma_fold_products: int
    expected_sigma_fold_products: int
    rank_one_updates: int
    peak_rank_one_workspace_bytes: int


@dataclass(frozen=True)
class StencilEstimateV2:
    selected_rows: int
    selected_active_rows: int
    unfused_path_products: int
    contraction_products: int
    emission_contributions: int
    fused_total_work: int
    avoided_products: int
    coefficient_entries: int
    pair_count: int
    intersection_count: int
    exact_selected_logical_nnz: int
    exact_full_logical_nnz: int
    resident: ResidentBreakdown
    selected_csr_upper_bytes: int
    input_snapshot_bytes: int
    compilation_transient_bytes: int
    controlled_transient_bytes: int
    physical_after_bytes: int | None
    transaction_contraction_delta: int
    prospective_emission_products: int
    emission_accounting_deferred: bool
    transaction_resident_delta_bytes: int
    transaction_transient_delta_bytes: int
    transaction_projected_total_work: int

    @property
    def resident_bytes(self) -> int:
        return self.resident.total_bytes

    @property
    def result_nnz_upper(self) -> int:
        return self.exact_selected_logical_nnz


@dataclass(frozen=True)
class TransactionSnapshot:
    compiled_content_keys: frozenset[tuple]
    contraction_used: int
    descriptor_resident_used_bytes: int
    transient_live_bytes: int
    transient_peak_bytes: int
    reservation_open: bool
    version: int


@dataclass(frozen=True)
class _Reservation:
    owner_token: object
    reservation_token: object


@dataclass(frozen=True)
class DescriptorQuote:
    needs_compile: bool
    contraction_delta: int
    resident_delta_bytes: int
    transient_delta: int
    projected_contraction_work: int
    prospective_request_work: int
    physical_after_bytes: int | None


@dataclass
class _PendingDescriptor:
    reservation_token: object
    content_key: tuple
    needs_compile: bool
    contraction_delta: int
    resident_delta_bytes: int
    transient_delta: int
    cached_operator: "ComposedConv2DStencilCandidateV2 | None"
    snapshot_before: TransactionSnapshot
    commit_value: "ComposedConv2DStencilCandidateV2 | None" = None


class V2Transaction:
    """One descriptor-compilation transaction with atomic rollback.

    Emission is deliberately not committed here.  A real materializer must
    reserve around actual CSR construction and either cache the resulting
    artifact or charge every consumer.  Recording support without an artifact
    would be false accounting.  Descriptor commit is this local object's
    linearization point; an outer fresh staging transaction is still required
    so an asynchronous exception after commit cannot publish cache aliases to
    production.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._owner_token = object()
        self._active_reservation_token: object | None = None
        self._pending: dict[object, _PendingDescriptor] = {}
        self._compiled: dict[tuple, ComposedConv2DStencilCandidateV2] = {}
        self._reserved_content_keys: set[tuple] = set()
        self._contraction_used = 0
        self._descriptor_resident_used_bytes = 0
        self._transient_live_bytes = 0
        self._transient_peak_bytes = 0
        self._version = 0

    def _snapshot_unlocked(self) -> TransactionSnapshot:
        return TransactionSnapshot(
            compiled_content_keys=frozenset(self._compiled),
            contraction_used=self._contraction_used,
            descriptor_resident_used_bytes=(
                self._descriptor_resident_used_bytes
            ),
            transient_live_bytes=self._transient_live_bytes,
            transient_peak_bytes=self._transient_peak_bytes,
            reservation_open=self._active_reservation_token is not None,
            version=self._version,
        )

    def snapshot(self) -> TransactionSnapshot:
        with self._lock:
            actual_resident = sum(
                int(operator_value.resident_bytes)
                for operator_value in self._compiled.values()
            )
            if actual_resident != self._descriptor_resident_used_bytes:
                raise RuntimeError("descriptor_resident_ledger_mismatch")
            return self._snapshot_unlocked()

    def new_reservation_request(self) -> _Reservation:
        """Return an opaque request token before any ledger mutation.

        The caller must retain this token across ``reserve_descriptor``.  It
        lets a ``finally`` block identify and undo exactly this call if an
        asynchronous exception lands after reserve has mutated the
        transaction but before its return value is assigned locally.
        """

        return _Reservation(
            owner_token=self._owner_token,
            reservation_token=object(),
        )

    def _quote_unlocked(
        self,
        *,
        content_key: tuple,
        contraction_products: int,
        descriptor_resident_bytes: int,
        prospective_emission_products: int,
        transient_bytes: int,
        reachable_before_bytes: int | None,
        reachable_after_nontransaction_bytes: int | None,
    ) -> DescriptorQuote:
        if not isinstance(content_key, tuple):
            raise CandidateV2Reject("content_key_type")
        contraction_products = _strict_nonnegative_int(
            contraction_products, name="contraction_products"
        )
        descriptor_resident_bytes = _strict_nonnegative_int(
            descriptor_resident_bytes, name="descriptor_resident_bytes"
        )
        prospective_emission_products = _strict_nonnegative_int(
            prospective_emission_products,
            name="prospective_emission_products",
        )
        transient_bytes = _strict_nonnegative_int(
            transient_bytes, name="transient_bytes"
        )
        if reachable_before_bytes is not None:
            reachable_before_bytes = _strict_nonnegative_int(
                reachable_before_bytes, name="reachable_before_bytes"
            )
        if reachable_after_nontransaction_bytes is not None:
            reachable_after_nontransaction_bytes = _strict_nonnegative_int(
                reachable_after_nontransaction_bytes,
                name="reachable_after_nontransaction_bytes",
            )

        cached = self._compiled.get(content_key)
        needs_compile = cached is None
        if cached is not None:
            if contraction_products != int(
                cached.compilation_ledger.gate_formula_products
            ):
                raise CandidateV2Reject(
                    "cached_descriptor_contraction_quote_mismatch"
                )
            if descriptor_resident_bytes != int(cached.resident_bytes):
                raise CandidateV2Reject(
                    "cached_descriptor_resident_quote_mismatch"
                )
        contraction_delta = contraction_products if needs_compile else 0
        resident_delta = descriptor_resident_bytes if needs_compile else 0
        transient_delta = transient_bytes if needs_compile else 0
        projected_contraction = self._contraction_used + contraction_delta
        prospective_request = (
            projected_contraction + prospective_emission_products
        )
        physical_after = (
            None
            if reachable_after_nontransaction_bytes is None
            else reachable_after_nontransaction_bytes
            + self._descriptor_resident_used_bytes
            + resident_delta
        )
        return DescriptorQuote(
            needs_compile=needs_compile,
            contraction_delta=contraction_delta,
            resident_delta_bytes=resident_delta,
            transient_delta=transient_delta,
            projected_contraction_work=projected_contraction,
            prospective_request_work=prospective_request,
            physical_after_bytes=physical_after,
        )

    def preview_descriptor(
        self,
        *,
        content_key: tuple,
        contraction_products: int,
        descriptor_resident_bytes: int,
        prospective_emission_products: int,
        transient_bytes: int,
        reachable_before_bytes: int | None,
        reachable_after_nontransaction_bytes: int | None,
    ) -> DescriptorQuote:
        with self._lock:
            return self._quote_unlocked(
                content_key=content_key,
                contraction_products=contraction_products,
                descriptor_resident_bytes=descriptor_resident_bytes,
                prospective_emission_products=(
                    prospective_emission_products
                ),
                transient_bytes=transient_bytes,
                reachable_before_bytes=reachable_before_bytes,
                reachable_after_nontransaction_bytes=(
                    reachable_after_nontransaction_bytes
                ),
            )

    def reserve_descriptor(
        self,
        *,
        request: _Reservation,
        content_key: tuple,
        contraction_products: int,
        descriptor_resident_bytes: int,
        prospective_emission_products: int,
        transient_bytes: int,
        reachable_before_bytes: int | None,
        reachable_after_nontransaction_bytes: int | None,
        limits: FrozenV1Limits,
    ) -> tuple[_Reservation | None, str | None, DescriptorQuote]:
        """Atomically quote and reserve one descriptor compilation."""

        _validate_limits(limits)
        with self._lock:
            if not isinstance(request, _Reservation):
                raise RuntimeError("reservation_type_mismatch")
            if request.owner_token is not self._owner_token:
                raise RuntimeError("foreign_reservation")
            request_token = request.reservation_token
            if request_token in self._pending:
                raise RuntimeError("reservation_request_already_active")
            quote = self._quote_unlocked(
                content_key=content_key,
                contraction_products=contraction_products,
                descriptor_resident_bytes=descriptor_resident_bytes,
                prospective_emission_products=(
                    prospective_emission_products
                ),
                transient_bytes=transient_bytes,
                reachable_before_bytes=reachable_before_bytes,
                reachable_after_nontransaction_bytes=(
                    reachable_after_nontransaction_bytes
                ),
            )
            if self._active_reservation_token is not None:
                return None, "transaction_reservation_conflict", quote
            cached_operator = self._compiled.get(content_key)
            descriptor_contraction = (
                quote.contraction_delta
                if cached_operator is None
                else int(
                    cached_operator.compilation_ledger.gate_formula_products
                )
            )
            descriptor_resident = (
                quote.resident_delta_bytes
                if cached_operator is None
                else int(cached_operator.resident_bytes)
            )
            if (
                descriptor_contraction
                > limits.max_descriptor_contraction_products
            ):
                return None, "contraction_product_limit", quote
            if (
                descriptor_resident > limits.max_resident_bytes
            ):
                return None, "resident_payload_limit", quote
            if quote.prospective_request_work > limits.max_transaction_work:
                return None, "transaction_work_limit", quote
            if (
                self._transient_live_bytes + quote.transient_delta
                > limits.max_transient_bytes
            ):
                return None, "controlled_transient_limit", quote
            if reachable_before_bytes is None or (
                reachable_after_nontransaction_bytes is None
            ):
                return None, "physical_metric_unproven", quote
            if quote.physical_after_bytes >= reachable_before_bytes:
                return None, "physical_metric_not_reduced", quote

            reservation_token = request_token
            before = self._snapshot_unlocked()
            pending = _PendingDescriptor(
                reservation_token=reservation_token,
                content_key=content_key,
                needs_compile=quote.needs_compile,
                contraction_delta=quote.contraction_delta,
                resident_delta_bytes=quote.resident_delta_bytes,
                transient_delta=quote.transient_delta,
                cached_operator=self._compiled.get(content_key),
                snapshot_before=before,
            )
            try:
                self._contraction_used += quote.contraction_delta
                self._transient_live_bytes += quote.transient_delta
                self._transient_peak_bytes = max(
                    self._transient_peak_bytes,
                    self._transient_live_bytes,
                )
                if quote.needs_compile:
                    self._reserved_content_keys.add(content_key)
                self._pending[reservation_token] = pending
                self._active_reservation_token = reservation_token
            except BaseException:
                self._contraction_used = before.contraction_used
                self._descriptor_resident_used_bytes = (
                    before.descriptor_resident_used_bytes
                )
                self._transient_live_bytes = before.transient_live_bytes
                self._transient_peak_bytes = before.transient_peak_bytes
                self._reserved_content_keys.discard(content_key)
                self._pending.pop(reservation_token, None)
                self._active_reservation_token = None
                raise
            return request, None, quote

    def _require_owned_active(
        self, reservation: _Reservation
    ) -> _PendingDescriptor:
        if not isinstance(reservation, _Reservation):
            raise RuntimeError("reservation_type_mismatch")
        if reservation.owner_token is not self._owner_token:
            raise RuntimeError("foreign_reservation")
        token = reservation.reservation_token
        if self._active_reservation_token is not token:
            raise RuntimeError("reservation_not_active")
        pending = self._pending.get(token)
        if pending is None:
            raise RuntimeError("reservation_pending_missing")
        return pending

    def reservation_descriptor_state(
        self, reservation: _Reservation
    ) -> tuple[bool, "ComposedConv2DStencilCandidateV2 | None"]:
        with self._lock:
            pending = self._require_owned_active(reservation)
            return pending.needs_compile, pending.cached_operator

    def _restore_pending(self, pending: _PendingDescriptor) -> None:
        if pending.needs_compile and (
            self._compiled.get(pending.content_key) is pending.commit_value
        ):
            self._compiled.pop(pending.content_key, None)
        before = pending.snapshot_before
        self._contraction_used = before.contraction_used
        self._descriptor_resident_used_bytes = (
            before.descriptor_resident_used_bytes
        )
        self._transient_live_bytes = before.transient_live_bytes
        self._transient_peak_bytes = before.transient_peak_bytes
        self._version = before.version
        self._reserved_content_keys.discard(pending.content_key)
        self._pending.pop(pending.reservation_token, None)
        self._active_reservation_token = None

    def commit(
        self,
        reservation: _Reservation,
        operator_value: "ComposedConv2DStencilCandidateV2",
    ) -> None:
        with self._lock:
            pending = self._require_owned_active(reservation)
            if pending.needs_compile:
                if pending.content_key in self._compiled:
                    raise RuntimeError("descriptor_commit_collision")
                if pending.content_key not in self._reserved_content_keys:
                    raise RuntimeError("descriptor_reservation_missing")
                if not isinstance(
                    operator_value, ComposedConv2DStencilCandidateV2
                ) or operator_value.content_key != pending.content_key:
                    raise RuntimeError("descriptor_content_mismatch")
                if operator_value.resident_bytes != pending.resident_delta_bytes:
                    raise RuntimeError("descriptor_resident_mismatch")
                ledger = operator_value.compilation_ledger
                if not (
                    int(ledger.gate_formula_products)
                    == int(ledger.actual_contraction_products)
                    == pending.contraction_delta
                ):
                    raise RuntimeError("descriptor_contraction_ledger_mismatch")
            elif pending.cached_operator is not operator_value:
                raise RuntimeError("cached_descriptor_identity_mismatch")

            pending.commit_value = operator_value
            try:
                if pending.needs_compile:
                    self._compiled[pending.content_key] = operator_value
                    self._descriptor_resident_used_bytes += (
                        pending.resident_delta_bytes
                    )
                    self._reserved_content_keys.remove(pending.content_key)
                    self._version += 1
                self._transient_live_bytes -= pending.transient_delta
                self._pending.pop(pending.reservation_token)
                self._active_reservation_token = None
            except BaseException:
                self._restore_pending(pending)
                raise

    def rollback(self, reservation: _Reservation) -> None:
        with self._lock:
            if not isinstance(reservation, _Reservation):
                raise RuntimeError("reservation_type_mismatch")
            if reservation.owner_token is not self._owner_token:
                raise RuntimeError("foreign_reservation")
            token = reservation.reservation_token
            # No allocation is permitted on cleanup.  With no other active
            # reservation, a same-owner token absent from the private pending
            # table is an already-closed/no-longer-visible reservation.
            if (
                self._active_reservation_token is None
                and token not in self._pending
            ):
                return
            pending = self._require_owned_active(reservation)
            self._restore_pending(pending)

    def rollback_if_pending(self, request: _Reservation) -> None:
        """Undo only ``request`` if it owns a currently pending reservation.

        A missing request is a no-op even when another request is active, so
        exception cleanup can never roll back a concurrent/re-entrant
        reservation that belongs to a different call.
        """

        with self._lock:
            if not isinstance(request, _Reservation):
                raise RuntimeError("reservation_type_mismatch")
            if request.owner_token is not self._owner_token:
                raise RuntimeError("foreign_reservation")
            token = request.reservation_token
            pending = self._pending.get(token)
            if pending is None:
                return
            if self._active_reservation_token is not token:
                raise RuntimeError("reservation_active_token_mismatch")
            self._restore_pending(pending)


@dataclass(frozen=True)
class GateRequestV2:
    selected_rows: object
    reachable_before_bytes: int | None
    reachable_after_other_bytes: int | None
    transaction: V2Transaction
    limits: FrozenV1Limits = FrozenV1Limits()


@dataclass(frozen=True)
class BuildDecisionV2:
    triggered: bool
    reason: str
    estimate: StencilEstimateV2 | None
    operator: "ComposedConv2DStencilCandidateV2 | None"
    reused_descriptor: bool = False


@dataclass(frozen=True)
class _BlockSpec:
    outer_group: int
    inner_group: int
    middle_start: int
    middle_stop: int
    output_start: int
    output_stop: int
    input_start: int
    input_stop: int


def _strict_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise CandidateV2Reject(f"{name}_not_integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise CandidateV2Reject(f"{name}_not_integer") from exc
    if result < 0:
        raise CandidateV2Reject(f"{name}_negative")
    return result


def _normalize_rows(rows, size: int) -> np.ndarray:
    raw = np.asarray(rows)
    if raw.dtype.kind == "b":
        flat = raw.reshape(-1)
        if flat.size != size:
            raise CandidateV2Reject("selected_row_mask_shape")
        result = np.flatnonzero(flat).astype(np.int64)
    else:
        if raw.dtype.kind not in "iu":
            raise CandidateV2Reject("selected_rows_not_integer")
        result = np.asarray(raw, dtype=np.int64).reshape(-1).copy()
    if result.size and (int(result.min()) < 0 or int(result.max()) >= size):
        raise CandidateV2Reject("selected_row_out_of_range")
    return result


def _typed_float64_key(array: np.ndarray) -> tuple:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"canonical_ndarray_v1\x00")
    digest.update(b"float64_le\x00")
    digest.update(struct.pack(">I", canonical.ndim))
    for size in canonical.shape:
        digest.update(struct.pack(">Q", int(size)))
    digest.update(canonical.tobytes(order="C"))
    return (
        "canonical_ndarray_v1",
        "float64_le",
        tuple(int(v) for v in canonical.shape),
        digest.digest(),
    )


def _typed_int64_key(array: np.ndarray) -> tuple:
    canonical = np.ascontiguousarray(array, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"canonical_index_vector_v1\x00")
    digest.update(struct.pack(">Q", int(canonical.size)))
    digest.update(canonical.tobytes(order="C"))
    return (
        "canonical_index_vector_v1",
        "int64_le",
        (int(canonical.size),),
        digest.digest(),
    )


def _typed_bool_key(array: np.ndarray) -> tuple:
    canonical = np.ascontiguousarray(array, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(b"canonical_bool_vector_v1\x00")
    digest.update(struct.pack(">Q", int(canonical.size)))
    digest.update(canonical.tobytes(order="C"))
    return (
        "canonical_bool_vector_v1",
        "bool_u8",
        (int(canonical.size),),
        digest.digest(),
    )


def _stationary_channel_vector(value, shape, *, name: str) -> np.ndarray:
    batch, channels, height, width = (int(v) for v in shape)
    total = batch * channels * height * width
    if isinstance(value, DiagonalLinearOp):
        raw = np.asarray(value._diagonal)
    elif sp.issparse(value):
        matrix = value.tocsr().astype(np.float64, copy=True)
        if matrix.shape != (total, total):
            raise CandidateV2Reject(f"{name}_diagonal_shape")
        diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
        residual = matrix - sp.diags(diagonal, format="csr")
        residual.eliminate_zeros()
        if residual.nnz:
            raise CandidateV2Reject(f"{name}_not_diagonal")
        raw = diagonal
    else:
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise CandidateV2Reject(f"{name}_not_array") from exc
        if array.dtype.kind not in "biuf":
            raise CandidateV2Reject(f"{name}_not_real")
        if array.ndim == 0:
            raw = np.full(channels, float(array), dtype=np.float64)
        elif array.ndim == 2:
            if array.shape != (total, total):
                raise CandidateV2Reject(f"{name}_diagonal_shape")
            dense = np.asarray(array, dtype=np.float64)
            diagonal = np.diag(dense)
            if np.any(dense != np.diag(diagonal)):
                raise CandidateV2Reject(f"{name}_not_diagonal")
            raw = diagonal
        else:
            raw = np.asarray(array, dtype=np.float64).reshape(-1)

    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(raw)):
        raise CandidateV2Reject(f"{name}_nonfinite")
    if raw.size == channels:
        return np.array(raw, dtype=np.float64, copy=True)
    if raw.size != total:
        raise CandidateV2Reject(f"{name}_shape")
    full = raw.reshape(batch, channels, height, width)
    channel = np.array(full[0, :, 0, 0], dtype=np.float64, copy=True)
    expected = np.broadcast_to(channel.reshape(1, channels, 1, 1), full.shape)
    if not np.array_equal(full, expected):
        raise CandidateV2Reject(f"{name}_not_channel_stationary")
    return channel


def _conv_payload(op, *, name: str):
    if not isinstance(op, ImplicitConv2DOp):
        raise CandidateV2Reject(f"{name}_not_implicit_conv2d")
    required = (
        "_kernel",
        "_input_shape",
        "_output_shape",
        "_stride",
        "_padding",
        "_dilation",
        "_groups",
        "_row_mask",
    )
    if any(not hasattr(op, field) for field in required):
        raise CandidateV2Reject(f"{name}_descriptor_layout")
    # The current ImplicitConv2DOp payload buffers are writeable even though its
    # constructor-time content_key is fixed.  Snapshot the semantic payload and
    # derive identity from that snapshot so a later in-place weight/mask edit
    # can never reuse a stale compiled descriptor.
    kernel = np.array(op._kernel, dtype=np.float64, order="C", copy=True)
    if kernel.ndim != 4 or not np.all(np.isfinite(kernel)):
        raise CandidateV2Reject(f"{name}_kernel_invalid")
    input_shape = tuple(int(v) for v in op._input_shape)
    output_shape = tuple(int(v) for v in op._output_shape)
    stride = tuple(int(v) for v in op._stride)
    padding = tuple(int(v) for v in op._padding)
    dilation = tuple(int(v) for v in op._dilation)
    groups = int(op._groups)
    row_mask = (
        None
        if op._row_mask is None
        else np.array(op._row_mask, dtype=bool, copy=True).reshape(-1)
    )
    if row_mask is not None and row_mask.size != math.prod(output_shape):
        raise CandidateV2Reject(f"{name}_row_mask_shape")
    snapshot_key = (
        "implicit_conv2d_semantic_snapshot_v1",
        input_shape,
        output_shape,
        stride,
        padding,
        dilation,
        groups,
        _typed_float64_key(kernel),
        None if row_mask is None else _typed_bool_key(row_mask),
    )
    kernel.flags.writeable = False
    if row_mask is not None:
        row_mask.flags.writeable = False
    return (
        kernel,
        input_shape,
        output_shape,
        stride,
        padding,
        dilation,
        groups,
        row_mask,
        snapshot_key,
    )


def _block_specs(
    *,
    input_channels: int,
    middle_channels: int,
    output_channels: int,
    inner_groups: int,
    outer_groups: int,
) -> tuple[tuple[_BlockSpec, ...], tuple[int, ...]]:
    if input_channels % inner_groups:
        raise CandidateV2Reject("input_group_divisibility")
    if middle_channels % inner_groups:
        raise CandidateV2Reject("inner_output_group_divisibility")
    if middle_channels % outer_groups:
        raise CandidateV2Reject("outer_input_group_divisibility")
    if output_channels % outer_groups:
        raise CandidateV2Reject("output_group_divisibility")

    inner_middle = middle_channels // inner_groups
    inner_input = input_channels // inner_groups
    outer_middle = middle_channels // outer_groups
    outer_output = output_channels // outer_groups
    specs: list[_BlockSpec] = []
    reachable: list[int] = []
    for gb in range(outer_groups):
        outer_m0 = gb * outer_middle
        outer_m1 = outer_m0 + outer_middle
        output0 = gb * outer_output
        output1 = output0 + outer_output
        group_reachable = 0
        covered = 0
        for ga in range(inner_groups):
            inner_m0 = ga * inner_middle
            inner_m1 = inner_m0 + inner_middle
            middle0 = max(outer_m0, inner_m0)
            middle1 = min(outer_m1, inner_m1)
            if middle0 >= middle1:
                continue
            input0 = ga * inner_input
            input1 = input0 + inner_input
            specs.append(
                _BlockSpec(
                    outer_group=gb,
                    inner_group=ga,
                    middle_start=middle0,
                    middle_stop=middle1,
                    output_start=output0,
                    output_stop=output1,
                    input_start=input0,
                    input_stop=input1,
                )
            )
            covered += middle1 - middle0
            group_reachable += inner_input
        if covered != outer_middle:
            raise CandidateV2Reject("group_intersection_partition")
        reachable.append(group_reachable)
    return tuple(specs), tuple(reachable)


def _spatial_metrics(
    *,
    input_shape,
    middle_shape,
    output_shape,
    inner_stride,
    inner_padding,
    inner_dilation,
    outer_stride,
    outer_padding,
    outer_dilation,
    inner_kernel_shape,
    outer_kernel_shape,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, input_h, input_w = input_shape
    _, _, middle_h, middle_w = middle_shape
    _, _, output_h, output_w = output_shape
    inner_h, inner_w = inner_kernel_shape
    outer_h, outer_w = outer_kernel_shape
    paths = np.zeros(output_h * output_w, dtype=np.int64)
    unique = np.zeros(output_h * output_w, dtype=np.int64)
    for oh in range(output_h):
        for ow in range(output_w):
            columns: set[int] = set()
            path_count = 0
            for th in range(outer_h):
                mh = (
                    oh * outer_stride[0]
                    - outer_padding[0]
                    + th * outer_dilation[0]
                )
                if mh < 0 or mh >= middle_h:
                    continue
                for tw in range(outer_w):
                    mw = (
                        ow * outer_stride[1]
                        - outer_padding[1]
                        + tw * outer_dilation[1]
                    )
                    if mw < 0 or mw >= middle_w:
                        continue
                    for qh in range(inner_h):
                        ih = (
                            mh * inner_stride[0]
                            - inner_padding[0]
                            + qh * inner_dilation[0]
                        )
                        if ih < 0 or ih >= input_h:
                            continue
                        for qw in range(inner_w):
                            iw = (
                                mw * inner_stride[1]
                                - inner_padding[1]
                                + qw * inner_dilation[1]
                            )
                            if 0 <= iw < input_w:
                                path_count += 1
                                columns.add(ih * input_w + iw)
            position = oh * output_w + ow
            paths[position] = path_count
            unique[position] = len(columns)
    return paths, unique


def _selected_metrics(
    *,
    rows: np.ndarray,
    output_shape,
    inner_input_per_group: int,
    outer_input_per_group: int,
    outer_groups: int,
    outer_row_mask,
    reachable_inputs_by_group,
    spatial_paths: np.ndarray,
    spatial_unique: np.ndarray,
) -> tuple[int, int, int, int]:
    _, output_channels, output_h, output_w = output_shape
    spatial = output_h * output_w
    per_batch = output_channels * spatial
    output_per_group = output_channels // outer_groups
    active = 0
    unfused = 0
    emission = 0
    logical = 0
    for raw_row in rows:
        row = int(raw_row)
        if outer_row_mask is not None and not bool(outer_row_mask[row]):
            continue
        active += 1
        _, within_batch = divmod(row, per_batch)
        output_channel, position = divmod(within_batch, spatial)
        group = output_channel // output_per_group
        paths = int(spatial_paths[position])
        reachable = int(reachable_inputs_by_group[group])
        unfused += (
            paths * outer_input_per_group * inner_input_per_group
        )
        emission += paths * reachable
        logical += int(spatial_unique[position]) * reachable
    return active, unfused, emission, logical


def _exact_full_logical_nnz(
    *,
    output_shape,
    outer_groups: int,
    outer_row_mask,
    reachable_inputs_by_group,
    spatial_unique: np.ndarray,
) -> int:
    batch, output_channels, output_h, output_w = output_shape
    spatial = output_h * output_w
    output_per_group = output_channels // outer_groups
    if outer_row_mask is None:
        spatial_total = int(spatial_unique.sum())
        return int(
            batch
            * output_per_group
            * spatial_total
            * sum(int(v) for v in reachable_inputs_by_group)
        )
    total = 0
    per_batch = output_channels * spatial
    for row in np.flatnonzero(outer_row_mask):
        _, within_batch = divmod(int(row), per_batch)
        output_channel, position = divmod(within_batch, spatial)
        group = output_channel // output_per_group
        total += int(spatial_unique[position]) * int(
            reachable_inputs_by_group[group]
        )
    return total


def _validate_limits(limits: FrozenV1Limits) -> None:
    if not isinstance(limits, FrozenV1Limits):
        raise CandidateV2Reject("limits_type")
    for name in (
        "max_descriptor_contraction_products",
        "max_transaction_work",
        "max_coefficient_entries",
        "max_resident_bytes",
        "max_transient_bytes",
        "max_result_nnz",
        "max_work_numerator",
        "max_work_denominator",
    ):
        _strict_nonnegative_int(getattr(limits, name), name=name)
    if limits.max_work_denominator == 0:
        raise CandidateV2Reject("max_work_denominator_zero")


def _canonical_left(Q, *, expected_columns: int) -> sp.csr_matrix:
    if sp.issparse(Q):
        if Q.ndim != 2 or Q.dtype.kind not in "biuf":
            raise CandidateV2Reject("left_factor_invalid")
        left = Q.tocsr().astype(np.float64, copy=True)
    else:
        raw = np.asarray(Q)
        if raw.ndim != 2 or raw.dtype.kind not in "biuf":
            raise CandidateV2Reject("left_factor_invalid")
        left = sp.csr_matrix(np.asarray(raw, dtype=np.float64))
    if left.shape[1] != expected_columns:
        raise CandidateV2Reject("left_factor_shape")
    left.sum_duplicates()
    left.sort_indices()
    left.eliminate_zeros()
    if not np.all(np.isfinite(left.data)):
        raise CandidateV2Reject("left_factor_nonfinite")
    return left


class ComposedConv2DStencilCandidateV2:
    """Exact factored Conv/diagonal/Conv descriptor, experiment-only."""

    # Columns in the resident int64 block metadata table.
    _GB = 0
    _GA = 1
    _M0 = 2
    _M1 = 3
    _O0 = 4
    _O1 = 5
    _I0 = 6
    _I1 = 7
    _META_WIDTH = 8

    def __init__(
        self,
        *,
        input_shape,
        middle_shape,
        output_shape,
        inner_stride,
        inner_padding,
        inner_dilation,
        outer_stride,
        outer_padding,
        outer_dilation,
        inner_kernel_shape,
        outer_kernel_shape,
        inner_groups,
        outer_groups,
        coefficients,
        block_metadata,
        outer_group_indptr,
        reachable_input_counts,
        outer_row_mask,
        content_key,
        logical_expanded_nnz,
        compilation_ledger,
        resident,
    ):
        self.input_shape = tuple(int(v) for v in input_shape)
        self.middle_shape = tuple(int(v) for v in middle_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        self.inner_stride = tuple(int(v) for v in inner_stride)
        self.inner_padding = tuple(int(v) for v in inner_padding)
        self.inner_dilation = tuple(int(v) for v in inner_dilation)
        self.outer_stride = tuple(int(v) for v in outer_stride)
        self.outer_padding = tuple(int(v) for v in outer_padding)
        self.outer_dilation = tuple(int(v) for v in outer_dilation)
        self._inner_kernel_shape = tuple(int(v) for v in inner_kernel_shape)
        self._outer_kernel_shape = tuple(int(v) for v in outer_kernel_shape)
        self._inner_groups = int(inner_groups)
        self._outer_groups = int(outer_groups)
        self._coefficients = tuple(coefficients)
        self._block_metadata = block_metadata
        self._outer_group_indptr = outer_group_indptr
        self._reachable_input_counts = reachable_input_counts
        self._outer_row_mask = outer_row_mask
        self._content_key = content_key
        self._logical_expanded_nnz = int(logical_expanded_nnz)
        self.compilation_ledger = compilation_ledger
        self._resident = resident

        for array in self._coefficients:
            array.flags.writeable = False
        for array in (
            self._block_metadata,
            self._outer_group_indptr,
            self._reachable_input_counts,
        ):
            array.flags.writeable = False
        if self._outer_row_mask is not None:
            self._outer_row_mask.flags.writeable = False
        if self.resident_bytes != self._resident.total_bytes:
            raise RuntimeError("resident_ledger_mismatch")

    @classmethod
    def try_build(
        cls,
        inner,
        middle_ops: Sequence[object],
        outer,
        gate: GateRequestV2,
    ) -> BuildDecisionV2:
        estimate = None
        reservation_request = None
        transaction = None
        try:
            if not isinstance(gate, GateRequestV2):
                raise CandidateV2Reject("gate_type")
            if not isinstance(gate.transaction, V2Transaction):
                raise CandidateV2Reject("transaction_type")
            transaction = gate.transaction
            _validate_limits(gate.limits)
            for name in (
                "reachable_before_bytes",
                "reachable_after_other_bytes",
            ):
                value = getattr(gate, name)
                if value is not None:
                    _strict_nonnegative_int(value, name=name)

            inner_payload = _conv_payload(inner, name="inner")
            outer_payload = _conv_payload(outer, name="outer")
            (
                inner_kernel,
                input_shape,
                middle_shape,
                inner_stride,
                inner_padding,
                inner_dilation,
                inner_groups,
                inner_mask,
                inner_snapshot_key,
            ) = inner_payload
            (
                outer_kernel,
                outer_input_shape,
                output_shape,
                outer_stride,
                outer_padding,
                outer_dilation,
                outer_groups,
                outer_mask,
                outer_snapshot_key,
            ) = outer_payload
            if middle_shape != outer_input_shape:
                raise CandidateV2Reject("intermediate_shape_mismatch")
            if not (
                input_shape[0] == middle_shape[0] == output_shape[0]
            ):
                raise CandidateV2Reject("batch_mismatch")

            batch, middle_channels, _, _ = middle_shape
            del batch
            if not isinstance(middle_ops, Sequence) or isinstance(
                middle_ops, (str, bytes)
            ):
                raise CandidateV2Reject("middle_ops_not_sequence")
            sigma = np.ones(middle_channels, dtype=np.float64)
            for index, middle in enumerate(middle_ops):
                scale = _stationary_channel_vector(
                    middle,
                    middle_shape,
                    name=f"middle_{index}",
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    sigma = np.multiply(sigma, scale)
                if not np.all(np.isfinite(sigma)):
                    raise CandidateV2Reject("middle_scale_product_nonfinite")
            if inner_mask is not None:
                channel_mask = _stationary_channel_vector(
                    inner_mask,
                    middle_shape,
                    name="inner_row_mask",
                )
                if np.any((channel_mask != 0.0) & (channel_mask != 1.0)):
                    raise CandidateV2Reject("inner_row_mask_not_binary")
                sigma = np.multiply(sigma, channel_mask)

            input_channels = int(input_shape[1])
            output_channels = int(output_shape[1])
            if int(inner_kernel.shape[0]) != middle_channels:
                raise CandidateV2Reject("inner_output_channel_mismatch")
            if int(inner_kernel.shape[1]) * inner_groups != input_channels:
                raise CandidateV2Reject("inner_input_channel_mismatch")
            if int(outer_kernel.shape[1]) * outer_groups != middle_channels:
                raise CandidateV2Reject("outer_input_channel_mismatch")
            if int(outer_kernel.shape[0]) != output_channels:
                raise CandidateV2Reject("outer_output_channel_mismatch")

            specs, reachable_inputs = _block_specs(
                input_channels=input_channels,
                middle_channels=middle_channels,
                output_channels=output_channels,
                inner_groups=inner_groups,
                outer_groups=outer_groups,
            )
            rows = _normalize_rows(gate.selected_rows, math.prod(output_shape))
            inner_h, inner_w = (int(v) for v in inner_kernel.shape[2:])
            outer_h, outer_w = (int(v) for v in outer_kernel.shape[2:])
            inner_taps = inner_h * inner_w
            outer_taps = outer_h * outer_w
            inner_input_per_group = input_channels // inner_groups
            outer_input_per_group = middle_channels // outer_groups
            outer_output_per_group = output_channels // outer_groups

            coefficient_entries = sum(
                outer_taps
                * inner_taps
                * (spec.output_stop - spec.output_start)
                * (spec.input_stop - spec.input_start)
                for spec in specs
            )
            pair_count = sum(
                (spec.output_stop - spec.output_start)
                * (spec.input_stop - spec.input_start)
                for spec in specs
            )
            intersection_contraction = sum(
                outer_taps
                * inner_taps
                * (spec.middle_stop - spec.middle_start)
                * (spec.output_stop - spec.output_start)
                * (spec.input_stop - spec.input_start)
                for spec in specs
            )
            contraction_products = int(
                outer_taps
                * inner_taps
                * output_channels
                * outer_input_per_group
                * inner_input_per_group
            )
            if intersection_contraction != contraction_products:
                raise CandidateV2Reject("contraction_formula_mismatch")

            spatial_paths, spatial_unique = _spatial_metrics(
                input_shape=input_shape,
                middle_shape=middle_shape,
                output_shape=output_shape,
                inner_stride=inner_stride,
                inner_padding=inner_padding,
                inner_dilation=inner_dilation,
                outer_stride=outer_stride,
                outer_padding=outer_padding,
                outer_dilation=outer_dilation,
                inner_kernel_shape=(inner_h, inner_w),
                outer_kernel_shape=(outer_h, outer_w),
            )
            active, unfused, emission, selected_logical = _selected_metrics(
                rows=rows,
                output_shape=output_shape,
                inner_input_per_group=inner_input_per_group,
                outer_input_per_group=outer_input_per_group,
                outer_groups=outer_groups,
                outer_row_mask=outer_mask,
                reachable_inputs_by_group=reachable_inputs,
                spatial_paths=spatial_paths,
                spatial_unique=spatial_unique,
            )
            full_logical = _exact_full_logical_nnz(
                output_shape=output_shape,
                outer_groups=outer_groups,
                outer_row_mask=outer_mask,
                reachable_inputs_by_group=reachable_inputs,
                spatial_unique=spatial_unique,
            )
            address_limit = int(np.iinfo(np.intp).max)
            for name, value in (
                ("coefficient_entries", coefficient_entries),
                ("pair_count", pair_count),
                ("contraction_products", contraction_products),
                ("emission_contributions", emission),
                ("selected_logical_nnz", selected_logical),
                ("full_logical_nnz", full_logical),
            ):
                if value < 0 or value > address_limit:
                    raise CandidateV2Reject(f"{name}_index_overflow")

            coefficient_bytes = int(coefficient_entries * 8)
            block_metadata_bytes = int(len(specs) * cls._META_WIDTH * 8)
            group_indptr_bytes = int((outer_groups + 1) * 8)
            reachable_bytes = int(outer_groups * 8)
            mask_bytes = 0 if outer_mask is None else int(outer_mask.nbytes)
            resident = ResidentBreakdown(
                coefficient_bytes=coefficient_bytes,
                block_metadata_bytes=block_metadata_bytes,
                outer_group_indptr_bytes=group_indptr_bytes,
                reachable_input_count_bytes=reachable_bytes,
                outer_row_mask_bytes=mask_bytes,
            )
            peak_rank_one_entries = (
                outer_output_per_group
                + outer_output_per_group * inner_input_per_group
            )
            compilation_transient = int(
                resident.total_bytes
                + sigma.nbytes
                + spatial_paths.nbytes
                + spatial_unique.nbytes
                + peak_rank_one_entries * 8
            )
            selected_csr_bytes = int(
                selected_logical * 16 + (rows.size + 1) * 8
            )
            input_snapshot_bytes = int(
                inner_kernel.nbytes
                + outer_kernel.nbytes
                + (0 if inner_mask is None else inner_mask.nbytes)
                + (0 if outer_mask is None else outer_mask.nbytes)
            )
            controlled_transient = max(
                compilation_transient + input_snapshot_bytes,
                resident.total_bytes + selected_csr_bytes,
            )

            content_key = (
                "composed_conv2d_stencil_candidate_v2",
                ("algorithm", ALGORITHM_KEY),
                ("inner_operator", inner_snapshot_key),
                ("middle_scale", _typed_float64_key(sigma)),
                ("outer_operator", outer_snapshot_key),
            )
            descriptor_compile_transient = int(
                compilation_transient + input_snapshot_bytes
            )
            quote = transaction.preview_descriptor(
                content_key=content_key,
                contraction_products=contraction_products,
                descriptor_resident_bytes=resident.total_bytes,
                prospective_emission_products=emission,
                transient_bytes=descriptor_compile_transient,
                reachable_before_bytes=gate.reachable_before_bytes,
                reachable_after_nontransaction_bytes=(
                    gate.reachable_after_other_bytes
                ),
            )
            fused_work = int(contraction_products + emission)
            estimate = StencilEstimateV2(
                selected_rows=int(rows.size),
                selected_active_rows=int(active),
                unfused_path_products=int(unfused),
                contraction_products=contraction_products,
                emission_contributions=int(emission),
                fused_total_work=fused_work,
                avoided_products=int(max(0, unfused - fused_work)),
                coefficient_entries=int(coefficient_entries),
                pair_count=int(pair_count),
                intersection_count=len(specs),
                exact_selected_logical_nnz=int(selected_logical),
                exact_full_logical_nnz=int(full_logical),
                resident=resident,
                selected_csr_upper_bytes=selected_csr_bytes,
                input_snapshot_bytes=input_snapshot_bytes,
                compilation_transient_bytes=compilation_transient,
                controlled_transient_bytes=controlled_transient,
                physical_after_bytes=quote.physical_after_bytes,
                transaction_contraction_delta=quote.contraction_delta,
                prospective_emission_products=int(emission),
                emission_accounting_deferred=True,
                transaction_resident_delta_bytes=(
                    quote.resident_delta_bytes
                ),
                transaction_transient_delta_bytes=(
                    quote.transient_delta
                ),
                transaction_projected_total_work=(
                    quote.prospective_request_work
                ),
            )

            reason = cls._gate_reason(estimate, gate)
            if reason is not None:
                return BuildDecisionV2(False, reason, estimate, None)

            reservation_request = transaction.new_reservation_request()
            reservation, reason, reserved_quote = (
                transaction.reserve_descriptor(
                    request=reservation_request,
                    content_key=content_key,
                    contraction_products=contraction_products,
                    descriptor_resident_bytes=resident.total_bytes,
                    prospective_emission_products=emission,
                    transient_bytes=descriptor_compile_transient,
                    reachable_before_bytes=gate.reachable_before_bytes,
                    reachable_after_nontransaction_bytes=(
                        gate.reachable_after_other_bytes
                    ),
                    limits=gate.limits,
                )
            )
            if reservation is not None and reservation is not reservation_request:
                raise RuntimeError("reservation_request_identity_mismatch")
            if reserved_quote != quote:
                estimate = replace(
                    estimate,
                    physical_after_bytes=reserved_quote.physical_after_bytes,
                    transaction_contraction_delta=(
                        reserved_quote.contraction_delta
                    ),
                    transaction_resident_delta_bytes=(
                        reserved_quote.resident_delta_bytes
                    ),
                    transaction_transient_delta_bytes=(
                        reserved_quote.transient_delta
                    ),
                    transaction_projected_total_work=(
                        reserved_quote.prospective_request_work
                    ),
                )
            if reason is not None or reservation is None:
                return BuildDecisionV2(
                    False,
                    "reservation_failed" if reason is None else reason,
                    estimate,
                    None,
                )

            needs_compile, cached = transaction.reservation_descriptor_state(
                reservation
            )
            if not needs_compile:
                if cached is None:
                    raise CandidateV2Reject("cached_descriptor_missing")
                success = BuildDecisionV2(
                    True,
                    "triggered_cached_descriptor",
                    estimate,
                    cached,
                    True,
                )
                transaction.commit(reservation, cached)
                return success

            (
                coefficients,
                block_metadata,
                group_indptr,
                reachable_array,
                compilation_ledger,
            ) = cls._compile_group_intersections(
                inner_kernel=inner_kernel,
                outer_kernel=outer_kernel,
                sigma=sigma,
                specs=specs,
                reachable_inputs=reachable_inputs,
                inner_groups=inner_groups,
                outer_groups=outer_groups,
                contraction_products=contraction_products,
            )
            candidate = cls(
                input_shape=input_shape,
                middle_shape=middle_shape,
                output_shape=output_shape,
                inner_stride=inner_stride,
                inner_padding=inner_padding,
                inner_dilation=inner_dilation,
                outer_stride=outer_stride,
                outer_padding=outer_padding,
                outer_dilation=outer_dilation,
                inner_kernel_shape=inner_kernel.shape[2:],
                outer_kernel_shape=outer_kernel.shape[2:],
                inner_groups=inner_groups,
                outer_groups=outer_groups,
                coefficients=coefficients,
                block_metadata=block_metadata,
                outer_group_indptr=group_indptr,
                reachable_input_counts=reachable_array,
                outer_row_mask=outer_mask,
                content_key=content_key,
                logical_expanded_nnz=full_logical,
                compilation_ledger=compilation_ledger,
                resident=resident,
            )
            success = BuildDecisionV2(
                True, "triggered", estimate, candidate, False
            )
            transaction.commit(reservation, candidate)
            return success
        except CandidateV2Reject as exc:
            return BuildDecisionV2(False, str(exc), estimate, None)
        except MemoryError:
            return BuildDecisionV2(
                False, "controlled_allocation_failed", estimate, None
            )
        finally:
            if reservation_request is not None and transaction is not None:
                transaction.rollback_if_pending(reservation_request)

    @staticmethod
    def _gate_reason(estimate: StencilEstimateV2, gate: GateRequestV2):
        limits = gate.limits
        if (
            estimate.contraction_products
            > limits.max_descriptor_contraction_products
        ):
            return "contraction_product_limit"
        if estimate.transaction_projected_total_work > limits.max_transaction_work:
            return "transaction_work_limit"
        if estimate.coefficient_entries > limits.max_coefficient_entries:
            return "coefficient_entry_limit"
        if estimate.resident_bytes > limits.max_resident_bytes:
            return "resident_payload_limit"
        if estimate.controlled_transient_bytes > limits.max_transient_bytes:
            return "controlled_transient_limit"
        if estimate.result_nnz_upper > limits.max_result_nnz:
            return "result_nnz_limit"
        if estimate.unfused_path_products <= 0:
            return "no_unfused_work"
        if (
            estimate.fused_total_work * limits.max_work_denominator
            > estimate.unfused_path_products * limits.max_work_numerator
        ):
            return "insufficient_work_reduction"
        if gate.reachable_before_bytes is None or (
            gate.reachable_after_other_bytes is None
        ):
            return "physical_metric_unproven"
        if estimate.physical_after_bytes >= gate.reachable_before_bytes:
            return "physical_metric_not_reduced"
        return None

    @classmethod
    def _compile_group_intersections(
        cls,
        *,
        inner_kernel,
        outer_kernel,
        sigma,
        specs,
        reachable_inputs,
        inner_groups,
        outer_groups,
        contraction_products,
    ):
        inner_taps = int(inner_kernel.shape[2] * inner_kernel.shape[3])
        outer_taps = int(outer_kernel.shape[2] * outer_kernel.shape[3])
        outer_middle_per_group = int(outer_kernel.shape[1])
        coefficients: list[np.ndarray] = []
        metadata = np.empty((len(specs), cls._META_WIDTH), dtype=np.int64)
        group_indptr = np.zeros(outer_groups + 1, dtype=np.int64)
        actual_products = 0
        intersection_products = 0
        sigma_products = 0
        rank_one_updates = 0
        peak_workspace = 0

        for block_index, spec in enumerate(specs):
            output_count = spec.output_stop - spec.output_start
            input_count = spec.input_stop - spec.input_start
            block = np.zeros(
                (outer_taps, inner_taps, output_count, input_count),
                dtype=np.float64,
            )
            block_products = 0
            outer_middle0 = spec.outer_group * outer_middle_per_group
            with np.errstate(over="ignore", invalid="ignore"):
                for outer_tap in range(outer_taps):
                    outer_kh, outer_kw = divmod(
                        outer_tap, int(outer_kernel.shape[3])
                    )
                    for middle in range(spec.middle_start, spec.middle_stop):
                        outer_column = np.multiply(
                            outer_kernel[
                                spec.output_start : spec.output_stop,
                                middle - outer_middle0,
                                outer_kh,
                                outer_kw,
                            ],
                            sigma[middle],
                        )
                        sigma_products += output_count
                        for inner_tap in range(inner_taps):
                            inner_kh, inner_kw = divmod(
                                inner_tap, int(inner_kernel.shape[3])
                            )
                            inner_row = inner_kernel[
                                middle, :, inner_kh, inner_kw
                            ]
                            rank_one = np.multiply(
                                outer_column[:, np.newaxis],
                                inner_row[np.newaxis, :],
                            )
                            np.add(
                                block[outer_tap, inner_tap],
                                rank_one,
                                out=block[outer_tap, inner_tap],
                            )
                            products = output_count * input_count
                            block_products += products
                            actual_products += products
                            rank_one_updates += 1
                            peak_workspace = max(
                                peak_workspace,
                                (output_count + products) * 8,
                            )
            expected_block = (
                outer_taps
                * inner_taps
                * (spec.middle_stop - spec.middle_start)
                * output_count
                * input_count
            )
            if block_products != expected_block:
                raise CandidateV2Reject("block_contraction_ledger_mismatch")
            if not np.all(np.isfinite(block)):
                raise CandidateV2Reject("compiled_coefficients_nonfinite")
            intersection_products += expected_block
            metadata[block_index] = (
                spec.outer_group,
                spec.inner_group,
                spec.middle_start,
                spec.middle_stop,
                spec.output_start,
                spec.output_stop,
                spec.input_start,
                spec.input_stop,
            )
            coefficients.append(block)
            group_indptr[spec.outer_group + 1] += 1

        np.cumsum(group_indptr, out=group_indptr)
        reachable_array = np.asarray(reachable_inputs, dtype=np.int64)
        expected_sigma = int(outer_kernel.size)
        if not (
            actual_products
            == intersection_products
            == contraction_products
        ):
            raise CandidateV2Reject("gate_contraction_ledger_mismatch")
        if sigma_products != expected_sigma:
            raise CandidateV2Reject("sigma_fold_ledger_mismatch")
        ledger = CompilationLedger(
            actual_contraction_products=actual_products,
            gate_formula_products=contraction_products,
            intersection_formula_products=intersection_products,
            sigma_fold_products=sigma_products,
            expected_sigma_fold_products=expected_sigma,
            rank_one_updates=rank_one_updates,
            peak_rank_one_workspace_bytes=peak_workspace,
        )
        return (
            tuple(coefficients),
            metadata,
            group_indptr,
            reachable_array,
            ledger,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return math.prod(self.output_shape), math.prod(self.input_shape)

    @property
    def content_key(self) -> tuple:
        return self._content_key

    @property
    def resident_breakdown(self) -> ResidentBreakdown:
        return self._resident

    @property
    def resident_entries(self) -> int:
        return int(
            sum(array.size for array in self._coefficients)
            + self._block_metadata.size
            + self._outer_group_indptr.size
            + self._reachable_input_counts.size
            + (0 if self._outer_row_mask is None else self._outer_row_mask.size)
        )

    @property
    def resident_bytes(self) -> int:
        return int(
            sum(array.nbytes for array in self._coefficients)
            + self._block_metadata.nbytes
            + self._outer_group_indptr.nbytes
            + self._reachable_input_counts.nbytes
            + (0 if self._outer_row_mask is None else self._outer_row_mask.nbytes)
        )

    @property
    def logical_expanded_nnz(self) -> int:
        return self._logical_expanded_nnz

    def _row(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        total_rows, _ = self.shape
        if row < 0 or row >= total_rows:
            raise IndexError(row)
        if self._outer_row_mask is not None and not self._outer_row_mask[row]:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        batch, input_channels, input_h, input_w = self.input_shape
        _, _, middle_h, middle_w = self.middle_shape
        _, output_channels, output_h, output_w = self.output_shape
        del batch
        output_spatial = output_h * output_w
        output_per_batch = output_channels * output_spatial
        batch_index, within_batch = divmod(row, output_per_batch)
        output_channel, spatial = divmod(within_batch, output_spatial)
        oh, ow = divmod(spatial, output_w)
        output_per_group = output_channels // self._outer_groups
        outer_group = output_channel // output_per_group
        block_start = int(self._outer_group_indptr[outer_group])
        block_stop = int(self._outer_group_indptr[outer_group + 1])
        inner_h, inner_w = self._inner_kernel_shape
        outer_h, outer_w = self._outer_kernel_shape
        batch_offset = batch_index * input_channels * input_h * input_w
        accumulator: dict[int, float] = {}

        for th in range(outer_h):
            mh = (
                oh * self.outer_stride[0]
                - self.outer_padding[0]
                + th * self.outer_dilation[0]
            )
            if mh < 0 or mh >= middle_h:
                continue
            for tw in range(outer_w):
                mw = (
                    ow * self.outer_stride[1]
                    - self.outer_padding[1]
                    + tw * self.outer_dilation[1]
                )
                if mw < 0 or mw >= middle_w:
                    continue
                outer_tap = th * outer_w + tw
                for qh in range(inner_h):
                    ih = (
                        mh * self.inner_stride[0]
                        - self.inner_padding[0]
                        + qh * self.inner_dilation[0]
                    )
                    if ih < 0 or ih >= input_h:
                        continue
                    for qw in range(inner_w):
                        iw = (
                            mw * self.inner_stride[1]
                            - self.inner_padding[1]
                            + qw * self.inner_dilation[1]
                        )
                        if iw < 0 or iw >= input_w:
                            continue
                        inner_tap = qh * inner_w + qw
                        for block_index in range(block_start, block_stop):
                            meta = self._block_metadata[block_index]
                            output_local = output_channel - int(meta[self._O0])
                            input_start = int(meta[self._I0])
                            values = self._coefficients[block_index][
                                outer_tap,
                                inner_tap,
                                output_local,
                            ]
                            for input_local, raw_value in enumerate(values):
                                value = float(raw_value)
                                if value == 0.0:
                                    continue
                                input_channel = input_start + input_local
                                column = (
                                    batch_offset
                                    + input_channel * input_h * input_w
                                    + ih * input_w
                                    + iw
                                )
                                total = accumulator.get(column, 0.0) + value
                                if not math.isfinite(total):
                                    raise CandidateV2Reject("row_sum_nonfinite")
                                if total == 0.0:
                                    accumulator.pop(column, None)
                                else:
                                    accumulator[column] = total
        columns = np.asarray(sorted(accumulator), dtype=np.int64)
        values = np.asarray(
            [accumulator[int(column)] for column in columns],
            dtype=np.float64,
        )
        return columns, values

    def gather_rows(self, rows, max_nnz: int) -> sp.csr_matrix:
        selected = _normalize_rows(rows, self.shape[0])
        cap = _strict_nonnegative_int(max_nnz, name="max_nnz")
        indptr = np.zeros(selected.size + 1, dtype=np.int64)
        indices: list[int] = []
        data: list[float] = []
        for local_row, global_row in enumerate(selected):
            columns, values = self._row(int(global_row))
            if len(data) + values.size > cap:
                raise CandidateV2Reject("gather_result_nnz_limit")
            indices.extend(int(value) for value in columns)
            data.extend(float(value) for value in values)
            indptr[local_row + 1] = len(data)
        result = sp.csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                indptr,
            ),
            shape=(selected.size, self.shape[1]),
        )
        if not np.all(np.isfinite(result.data)):
            raise CandidateV2Reject("gather_result_nonfinite")
        return result

    def matvec(self, vector) -> np.ndarray:
        raw = np.asarray(vector)
        if raw.dtype.kind not in "biuf":
            raise CandidateV2Reject("matvec_input_invalid")
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.size != self.shape[1] or not np.all(np.isfinite(values)):
            raise CandidateV2Reject("matvec_input_invalid")
        result = np.zeros(self.shape[0], dtype=np.float64)
        for row in range(self.shape[0]):
            columns, coefficients = self._row(row)
            total = 0.0
            for column, coefficient in zip(columns, coefficients, strict=True):
                product = float(coefficient) * float(values[int(column)])
                if not math.isfinite(product):
                    raise CandidateV2Reject("matvec_product_nonfinite")
                total += product
                if not math.isfinite(total):
                    raise CandidateV2Reject("matvec_sum_nonfinite")
            result[row] = total
        return result

    def left_compose(self, Q, max_nnz: int) -> sp.csr_matrix:
        cap = _strict_nonnegative_int(max_nnz, name="max_nnz")
        left = _canonical_left(Q, expected_columns=self.shape[0])
        row_widths = np.diff(left.indptr)
        if np.all(row_widths <= 1):
            indptr = np.zeros(left.shape[0] + 1, dtype=np.int64)
            indices: list[int] = []
            data: list[float] = []
            for output_row in range(left.shape[0]):
                start = int(left.indptr[output_row])
                stop = int(left.indptr[output_row + 1])
                if start != stop:
                    source_row = int(left.indices[start])
                    scale = float(left.data[start])
                    columns, values = self._row(source_row)
                    with np.errstate(over="ignore", invalid="ignore"):
                        scaled = np.multiply(values, scale)
                    if not np.all(np.isfinite(scaled)):
                        raise CandidateV2Reject("left_result_nonfinite")
                    nonzero = scaled != 0.0
                    if len(data) + int(nonzero.sum()) > cap:
                        raise CandidateV2Reject("left_result_nnz_limit")
                    indices.extend(int(v) for v in columns[nonzero])
                    data.extend(float(v) for v in scaled[nonzero])
                indptr[output_row + 1] = len(data)
            return sp.csr_matrix(
                (
                    np.asarray(data, dtype=np.float64),
                    np.asarray(indices, dtype=np.int64),
                    indptr,
                ),
                shape=(left.shape[0], self.shape[1]),
            )

        support = np.unique(left.indices).astype(np.int64)
        if support.size == self.shape[0]:
            raise CandidateV2Reject("support_slice_would_expand_full_operator")
        gathered = self.gather_rows(support, max_nnz=cap)
        gathered_widths = np.diff(gathered.indptr)
        support_positions = np.searchsorted(support, left.indices)
        contribution_upper = sum(
            int(gathered_widths[int(position)])
            for position in support_positions
        )
        if contribution_upper > cap:
            # Bound multiplication before SciPy is allowed to allocate the
            # product.  This is conservative under overlap/cancellation and is
            # therefore a fail-closed resource guard, not a semantic change.
            raise CandidateV2Reject("left_product_contribution_limit")
        result = (left[:, support] @ gathered).tocsr()
        result.sum_duplicates()
        result.sort_indices()
        result.eliminate_zeros()
        if result.nnz > cap:
            raise CandidateV2Reject("left_result_nnz_limit")
        if not np.all(np.isfinite(result.data)):
            raise CandidateV2Reject("left_result_nonfinite")
        return result


__all__ = [
    "BuildDecisionV2",
    "CandidateV2Reject",
    "CompilationLedger",
    "ComposedConv2DStencilCandidateV2",
    "DescriptorQuote",
    "FrozenV1Limits",
    "GateRequestV2",
    "ResidentBreakdown",
    "StencilEstimateV2",
    "TransactionSnapshot",
    "V2Transaction",
]
