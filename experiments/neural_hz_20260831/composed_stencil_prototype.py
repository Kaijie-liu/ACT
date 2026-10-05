"""Isolated S0-C1 channel-contracted Conv-stencil prototype.

This module is deliberately outside the ACT runtime.  It reads immutable
``ImplicitConv2DOp`` descriptors, but no production path imports it.  The
prototype implements the real-algebra identity registered in
``S0_C1_COMPOSED_STENCIL_PREREG.md`` and the frozen candidate-v1 resource
gates.  It never expands either input convolution to CSR.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import operator
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp


MIB = 1024 * 1024
GIB = 1024 * MIB


class PrototypeReject(ValueError):
    """A deterministic, fail-closed prototype rejection."""


@dataclass(frozen=True)
class FrozenV1Limits:
    """Frozen S0-C1 candidate-v1 limits from the pre-registration."""

    max_descriptor_contraction_products: int = 200_000_000
    max_transaction_work: int = 256_000_000
    max_coefficient_entries: int = 2_000_000
    max_resident_bytes: int = 64 * MIB
    max_transient_bytes: int = GIB
    max_result_nnz: int = 64_000_000
    max_work_numerator: int = 1
    max_work_denominator: int = 4


@dataclass(frozen=True)
class GateContext:
    """State needed to evaluate all uniform pre-allocation gates."""

    selected_rows: object
    reachable_before_bytes: int | None
    reachable_after_other_bytes: int | None
    transaction_contraction_products: int = 0
    transaction_total_work: int = 0
    transaction_transient_live_bytes: int = 0
    limits: FrozenV1Limits = FrozenV1Limits()


@dataclass(frozen=True)
class StencilEstimate:
    selected_rows: int
    selected_active_rows: int
    unfused_path_products: int
    contraction_products: int
    emission_contributions: int
    fused_total_work: int
    avoided_products: int
    coefficient_entries: int
    pair_count: int
    resident_bytes: int
    controlled_transient_bytes: int
    result_nnz_upper: int
    physical_after_bytes: int | None


@dataclass(frozen=True)
class BuildDecision:
    triggered: bool
    reason: str
    estimate: StencilEstimate | None
    operator: "ComposedStencilPrototype | None"


def _strict_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise PrototypeReject(f"{name}_not_integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise PrototypeReject(f"{name}_not_integer") from exc
    if result < 0:
        raise PrototypeReject(f"{name}_negative")
    return result


def _normalize_rows(rows, size: int) -> np.ndarray:
    raw = np.asarray(rows)
    if raw.dtype.kind == "b":
        flat = raw.reshape(-1)
        if flat.size != size:
            raise PrototypeReject("selected_row_mask_shape")
        result = np.flatnonzero(flat).astype(np.int64)
    else:
        if raw.dtype.kind not in "iu":
            raise PrototypeReject("selected_rows_not_integer")
        result = np.asarray(raw, dtype=np.int64).reshape(-1)
    if result.size and (int(result.min()) < 0 or int(result.max()) >= size):
        raise PrototypeReject("selected_row_out_of_range")
    return result


def _stationary_channel_vector(value, shape, *, name: str) -> np.ndarray:
    """Return an exact per-channel vector or reject spatial/batch variation."""
    batch, channels, height, width = (int(v) for v in shape)
    total = batch * channels * height * width
    if sp.issparse(value):
        matrix = value.tocsr().astype(np.float64, copy=True)
        if matrix.shape != (total, total):
            raise PrototypeReject(f"{name}_diagonal_shape")
        diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
        residual = matrix - sp.diags(diagonal, format="csr")
        residual.eliminate_zeros()
        if residual.nnz:
            raise PrototypeReject(f"{name}_not_diagonal")
        raw = diagonal
    else:
        try:
            array = np.asarray(value)
        except Exception as exc:  # pragma: no cover - foreign array defense
            raise PrototypeReject(f"{name}_not_array") from exc
        if array.dtype.kind not in "biuf":
            raise PrototypeReject(f"{name}_not_real")
        if array.ndim == 0:
            raw = np.full(channels, float(array), dtype=np.float64)
        elif array.ndim == 2 and array.shape == (total, total):
            dense = np.asarray(array, dtype=np.float64)
            diagonal = np.diag(dense)
            if np.any(dense != np.diag(diagonal)):
                raise PrototypeReject(f"{name}_not_diagonal")
            raw = diagonal
        else:
            raw = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(raw)):
        raise PrototypeReject(f"{name}_nonfinite")
    if raw.size == channels:
        return np.array(raw, dtype=np.float64, copy=True)
    if raw.size != total:
        raise PrototypeReject(f"{name}_shape")
    full = raw.reshape(batch, channels, height, width)
    channel = np.array(full[0, :, 0, 0], dtype=np.float64, copy=True)
    expected = np.broadcast_to(channel.reshape(1, channels, 1, 1), full.shape)
    if not np.array_equal(full, expected):
        raise PrototypeReject(f"{name}_not_channel_stationary")
    return channel


def _conv_payload(op, *, name: str):
    if not isinstance(op, ImplicitConv2DOp):
        raise PrototypeReject(f"{name}_not_implicit_conv2d")
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
        raise PrototypeReject(f"{name}_descriptor_layout")
    kernel = np.asarray(op._kernel, dtype=np.float64)
    if kernel.ndim != 4 or not np.all(np.isfinite(kernel)):
        raise PrototypeReject(f"{name}_kernel_invalid")
    return (
        kernel,
        tuple(int(v) for v in op._input_shape),
        tuple(int(v) for v in op._output_shape),
        tuple(int(v) for v in op._stride),
        tuple(int(v) for v in op._padding),
        tuple(int(v) for v in op._dilation),
        int(op._groups),
        None
        if op._row_mask is None
        else np.asarray(op._row_mask, dtype=bool).reshape(-1),
    )


def _reachable_channel_pairs(
    *,
    input_channels: int,
    middle_channels: int,
    output_channels: int,
    inner_groups: int,
    outer_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner_out_per_group = middle_channels // inner_groups
    inner_in_per_group = input_channels // inner_groups
    outer_out_per_group = output_channels // outer_groups
    outer_in_per_group = middle_channels // outer_groups
    pair_out: list[int] = []
    pair_in: list[int] = []
    indptr = np.zeros(output_channels + 1, dtype=np.int64)
    for output_channel in range(output_channels):
        outer_group = output_channel // outer_out_per_group
        middle_start = outer_group * outer_in_per_group
        reachable = set()
        for middle_channel in range(
            middle_start, middle_start + outer_in_per_group
        ):
            inner_group = middle_channel // inner_out_per_group
            input_start = inner_group * inner_in_per_group
            reachable.update(
                range(input_start, input_start + inner_in_per_group)
            )
        for input_channel in sorted(reachable):
            pair_out.append(output_channel)
            pair_in.append(input_channel)
        indptr[output_channel + 1] = len(pair_out)
    return (
        np.asarray(pair_out, dtype=np.int64),
        np.asarray(pair_in, dtype=np.int64),
        indptr,
    )


def _matrix_taps(kernel, *, input_channels: int, groups: int) -> np.ndarray:
    """Expand only the small channel matrices, never a spatial Conv CSR."""
    output_channels, input_per_group, kernel_h, kernel_w = kernel.shape
    output_per_group = output_channels // groups
    taps = np.zeros(
        (kernel_h * kernel_w, output_channels, input_channels),
        dtype=np.float64,
    )
    for output_channel in range(output_channels):
        group = output_channel // output_per_group
        first_input = group * input_per_group
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                tap = kh * kernel_w + kw
                taps[
                    tap,
                    output_channel,
                    first_input : first_input + input_per_group,
                ] = kernel[output_channel, :, kh, kw]
    return taps


class ComposedStencilPrototype:
    """Exact-real channel-contracted descriptor for two NCHW Conv operators."""

    def __init__(
        self,
        *,
        inner_shape,
        middle_shape,
        output_shape,
        inner_stride,
        inner_padding,
        inner_dilation,
        outer_stride,
        outer_padding,
        outer_dilation,
        pair_out,
        pair_in,
        pair_indptr,
        coefficients,
        inner_kernel_shape,
        outer_kernel_shape,
        outer_row_mask,
        content_key,
        estimate,
    ):
        self.input_shape = tuple(inner_shape)
        self.middle_shape = tuple(middle_shape)
        self.output_shape = tuple(output_shape)
        self.inner_stride = tuple(inner_stride)
        self.inner_padding = tuple(inner_padding)
        self.inner_dilation = tuple(inner_dilation)
        self.outer_stride = tuple(outer_stride)
        self.outer_padding = tuple(outer_padding)
        self.outer_dilation = tuple(outer_dilation)
        self._pair_out = pair_out
        self._pair_in = pair_in
        self._pair_indptr = pair_indptr
        self._coefficients = coefficients
        self._inner_kernel_shape = tuple(inner_kernel_shape)
        self._outer_kernel_shape = tuple(outer_kernel_shape)
        self._outer_row_mask = outer_row_mask
        self._content_key = content_key
        self.estimate = estimate

        for array in (
            self._pair_out,
            self._pair_in,
            self._pair_indptr,
            self._coefficients,
        ):
            array.setflags(write=False)
        if self._outer_row_mask is not None:
            self._outer_row_mask.setflags(write=False)

    @classmethod
    def try_build(
        cls,
        inner,
        middle_ops: Sequence[object],
        outer,
        gate: GateContext,
    ) -> BuildDecision:
        """Apply every frozen v1 gate before allocating fused coefficients."""
        estimate = None
        try:
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
            ) = outer_payload
            if tuple(middle_shape) != tuple(outer_input_shape):
                raise PrototypeReject("intermediate_shape_mismatch")
            if input_shape[0] != middle_shape[0] or (
                middle_shape[0] != output_shape[0]
            ):
                raise PrototypeReject("batch_mismatch")

            batch, middle_channels, middle_h, middle_w = middle_shape
            sigma = np.ones(middle_channels, dtype=np.float64)
            if not isinstance(middle_ops, Sequence):
                raise PrototypeReject("middle_ops_not_sequence")
            for index, middle in enumerate(middle_ops):
                scale = _stationary_channel_vector(
                    middle,
                    middle_shape,
                    name=f"middle_{index}",
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    sigma = sigma * scale
                if not np.all(np.isfinite(sigma)):
                    raise PrototypeReject("middle_scale_product_nonfinite")
            if inner_mask is not None:
                channel_mask = _stationary_channel_vector(
                    inner_mask,
                    middle_shape,
                    name="inner_row_mask",
                )
                if np.any((channel_mask != 0.0) & (channel_mask != 1.0)):
                    raise PrototypeReject("inner_row_mask_not_binary")
                sigma = sigma * channel_mask

            input_channels = input_shape[1]
            output_channels = output_shape[1]
            if inner_kernel.shape[0] != middle_channels:
                raise PrototypeReject("inner_output_channel_mismatch")
            if outer_kernel.shape[1] * outer_groups != middle_channels:
                raise PrototypeReject("outer_input_channel_mismatch")

            pair_out, pair_in, pair_indptr = _reachable_channel_pairs(
                input_channels=input_channels,
                middle_channels=middle_channels,
                output_channels=output_channels,
                inner_groups=inner_groups,
                outer_groups=outer_groups,
            )
            rows = _normalize_rows(gate.selected_rows, math.prod(output_shape))
            limits = gate.limits
            for field in (
                "reachable_before_bytes",
                "reachable_after_other_bytes",
                "transaction_contraction_products",
                "transaction_total_work",
                "transaction_transient_live_bytes",
            ):
                value = getattr(gate, field)
                if value is not None:
                    _strict_nonnegative_int(value, name=field)

            kh_a, kw_a = inner_kernel.shape[2:]
            kh_b, kw_b = outer_kernel.shape[2:]
            taps_a = kh_a * kw_a
            taps_b = kh_b * kw_b
            inner_input_per_group = inner_kernel.shape[1]
            outer_input_per_group = outer_kernel.shape[1]
            contraction_products = int(
                taps_a
                * taps_b
                * output_channels
                * outer_input_per_group
                * inner_input_per_group
            )
            coefficient_entries = int(taps_a * taps_b * pair_out.size)
            outer_mask_bytes = 0 if outer_mask is None else outer_mask.nbytes
            resident_bytes = int(
                coefficient_entries * np.dtype(np.float64).itemsize
                + pair_out.nbytes
                + pair_in.nbytes
                + pair_indptr.nbytes
                + outer_mask_bytes
            )
            channel_workspace_bytes = int(
                taps_a * middle_channels * input_channels * 8
                # Scaling B creates a new array before the unscaled payload is
                # released, so charge both copies at the controlled peak.
                + 2 * taps_b * output_channels * middle_channels * 8
                + output_channels * input_channels * 8
            )
            controlled_transient_bytes = int(
                resident_bytes + channel_workspace_bytes
            )

            (
                active_rows,
                unfused_products,
                emission_contributions,
                result_nnz_upper,
            ) = cls._selected_work_bounds(
                rows=rows,
                input_shape=input_shape,
                middle_shape=middle_shape,
                output_shape=output_shape,
                inner_stride=inner_stride,
                inner_padding=inner_padding,
                inner_dilation=inner_dilation,
                outer_stride=outer_stride,
                outer_padding=outer_padding,
                outer_dilation=outer_dilation,
                inner_kernel_shape=inner_kernel.shape,
                outer_kernel_shape=outer_kernel.shape,
                pair_indptr=pair_indptr,
                outer_row_mask=outer_mask,
            )
            fused_work = int(contraction_products + emission_contributions)
            physical_after = (
                None
                if gate.reachable_after_other_bytes is None
                else int(gate.reachable_after_other_bytes) + resident_bytes
            )
            estimate = StencilEstimate(
                selected_rows=int(rows.size),
                selected_active_rows=int(active_rows),
                unfused_path_products=int(unfused_products),
                contraction_products=int(contraction_products),
                emission_contributions=int(emission_contributions),
                fused_total_work=int(fused_work),
                avoided_products=int(max(0, unfused_products - fused_work)),
                coefficient_entries=int(coefficient_entries),
                pair_count=int(pair_out.size),
                resident_bytes=int(resident_bytes),
                controlled_transient_bytes=int(controlled_transient_bytes),
                result_nnz_upper=int(result_nnz_upper),
                physical_after_bytes=physical_after,
            )

            reason = cls._gate_reason(estimate, gate)
            if reason is not None:
                return BuildDecision(False, reason, estimate, None)

            a_taps = _matrix_taps(
                inner_kernel,
                input_channels=input_channels,
                groups=inner_groups,
            )
            b_taps = _matrix_taps(
                outer_kernel,
                input_channels=middle_channels,
                groups=outer_groups,
            )
            b_taps = b_taps * sigma.reshape(1, 1, middle_channels)
            coefficients = np.empty(
                (taps_b, taps_a, pair_out.size), dtype=np.float64
            )
            for outer_tap in range(taps_b):
                for inner_tap in range(taps_a):
                    product = b_taps[outer_tap] @ a_taps[inner_tap]
                    coefficients[outer_tap, inner_tap] = product[
                        pair_out, pair_in
                    ]
            if not np.all(np.isfinite(coefficients)):
                return BuildDecision(
                    False, "compiled_coefficients_nonfinite", estimate, None
                )

            digest = hashlib.sha256()
            digest.update(repr(inner.content_key).encode("utf-8"))
            digest.update(repr(outer.content_key).encode("utf-8"))
            digest.update(np.ascontiguousarray(sigma).tobytes())
            content_key = (
                "isolated_composed_stencil_prototype_v1",
                digest.hexdigest(),
            )
            op = cls(
                inner_shape=input_shape,
                middle_shape=middle_shape,
                output_shape=output_shape,
                inner_stride=inner_stride,
                inner_padding=inner_padding,
                inner_dilation=inner_dilation,
                outer_stride=outer_stride,
                outer_padding=outer_padding,
                outer_dilation=outer_dilation,
                pair_out=pair_out,
                pair_in=pair_in,
                pair_indptr=pair_indptr,
                coefficients=coefficients,
                inner_kernel_shape=inner_kernel.shape,
                outer_kernel_shape=outer_kernel.shape,
                outer_row_mask=(
                    None
                    if outer_mask is None
                    else np.array(outer_mask, dtype=bool, copy=True)
                ),
                content_key=content_key,
                estimate=estimate,
            )
            return BuildDecision(True, "triggered", estimate, op)
        except PrototypeReject as exc:
            return BuildDecision(False, str(exc), None, None)
        except MemoryError:
            return BuildDecision(False, "controlled_allocation_failed", estimate, None)

    @staticmethod
    def _gate_reason(estimate: StencilEstimate, gate: GateContext):
        limits = gate.limits
        if (
            estimate.contraction_products
            > limits.max_descriptor_contraction_products
        ):
            return "contraction_product_limit"
        if (
            gate.transaction_total_work + estimate.fused_total_work
            > limits.max_transaction_work
        ):
            return "transaction_work_limit"
        if estimate.coefficient_entries > limits.max_coefficient_entries:
            return "coefficient_entry_limit"
        if estimate.resident_bytes > limits.max_resident_bytes:
            return "resident_payload_limit"
        if (
            gate.transaction_transient_live_bytes
            + estimate.controlled_transient_bytes
            > limits.max_transient_bytes
        ):
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

    @staticmethod
    def _selected_work_bounds(
        *,
        rows,
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
        pair_indptr,
        outer_row_mask,
    ):
        _, input_channels, input_h, input_w = input_shape
        _, _, middle_h, middle_w = middle_shape
        _, output_channels, output_h, output_w = output_shape
        kh_a, kw_a = inner_kernel_shape[2:]
        kh_b, kw_b = outer_kernel_shape[2:]
        inner_input_per_group = inner_kernel_shape[1]
        outer_input_per_group = outer_kernel_shape[1]
        output_spatial = output_h * output_w
        output_per_batch = output_channels * output_spatial
        active_rows = 0
        unfused = 0
        emission = 0
        for row in rows:
            row = int(row)
            if outer_row_mask is not None and not outer_row_mask[row]:
                continue
            active_rows += 1
            _, within_batch = divmod(row, output_per_batch)
            output_channel, spatial = divmod(within_batch, output_spatial)
            oh, ow = divmod(spatial, output_w)
            pairs = int(
                pair_indptr[output_channel + 1]
                - pair_indptr[output_channel]
            )
            for th in range(kh_b):
                mh = (
                    oh * outer_stride[0]
                    - outer_padding[0]
                    + th * outer_dilation[0]
                )
                if mh < 0 or mh >= middle_h:
                    continue
                for tw in range(kw_b):
                    mw = (
                        ow * outer_stride[1]
                        - outer_padding[1]
                        + tw * outer_dilation[1]
                    )
                    if mw < 0 or mw >= middle_w:
                        continue
                    valid_inner = 0
                    for qh in range(kh_a):
                        ih = (
                            mh * inner_stride[0]
                            - inner_padding[0]
                            + qh * inner_dilation[0]
                        )
                        if ih < 0 or ih >= input_h:
                            continue
                        for qw in range(kw_a):
                            iw = (
                                mw * inner_stride[1]
                                - inner_padding[1]
                                + qw * inner_dilation[1]
                            )
                            if 0 <= iw < input_w:
                                valid_inner += 1
                    unfused += (
                        outer_input_per_group
                        * inner_input_per_group
                        * valid_inner
                    )
                    emission += pairs * valid_inner
        result_upper = min(
            emission,
            active_rows * input_channels * input_h * input_w,
        )
        return active_rows, unfused, emission, result_upper

    @property
    def shape(self):
        return math.prod(self.output_shape), math.prod(self.input_shape)

    @property
    def content_key(self):
        return self._content_key

    @property
    def resident_entries(self):
        mask = 0 if self._outer_row_mask is None else self._outer_row_mask.size
        return int(
            self._coefficients.size
            + self._pair_out.size
            + self._pair_in.size
            + self._pair_indptr.size
            + mask
        )

    @property
    def resident_bytes(self):
        mask = 0 if self._outer_row_mask is None else self._outer_row_mask.nbytes
        return int(
            self._coefficients.nbytes
            + self._pair_out.nbytes
            + self._pair_in.nbytes
            + self._pair_indptr.nbytes
            + mask
        )

    @property
    def logical_expanded_nnz(self):
        """Conservative full-product CSR nnz after spatial collision."""
        rows = np.arange(self.shape[0], dtype=np.int64)
        return int(
            self._selected_work_bounds(
                rows=rows,
                input_shape=self.input_shape,
                middle_shape=self.middle_shape,
                output_shape=self.output_shape,
                inner_stride=self.inner_stride,
                inner_padding=self.inner_padding,
                inner_dilation=self.inner_dilation,
                outer_stride=self.outer_stride,
                outer_padding=self.outer_padding,
                outer_dilation=self.outer_dilation,
                inner_kernel_shape=self._inner_kernel_shape,
                outer_kernel_shape=self._outer_kernel_shape,
                pair_indptr=self._pair_indptr,
                outer_row_mask=self._outer_row_mask,
            )[3]
        )

    def _row(self, row: int):
        total_rows, _ = self.shape
        if row < 0 or row >= total_rows:
            raise IndexError(row)
        if self._outer_row_mask is not None and not self._outer_row_mask[row]:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        batch, input_channels, input_h, input_w = self.input_shape
        _, middle_channels, middle_h, middle_w = self.middle_shape
        _, output_channels, output_h, output_w = self.output_shape
        del batch, input_channels, middle_channels
        output_spatial = output_h * output_w
        output_per_batch = output_channels * output_spatial
        batch_index, within_batch = divmod(row, output_per_batch)
        output_channel, spatial = divmod(within_batch, output_spatial)
        oh, ow = divmod(spatial, output_w)
        kh_a, kw_a = self._inner_kernel_shape[2:]
        kh_b, kw_b = self._outer_kernel_shape[2:]
        start = int(self._pair_indptr[output_channel])
        stop = int(self._pair_indptr[output_channel + 1])
        accumulator: dict[int, float] = {}
        batch_offset = batch_index * self.input_shape[1] * input_h * input_w
        for th in range(kh_b):
            mh = (
                oh * self.outer_stride[0]
                - self.outer_padding[0]
                + th * self.outer_dilation[0]
            )
            if mh < 0 or mh >= middle_h:
                continue
            for tw in range(kw_b):
                mw = (
                    ow * self.outer_stride[1]
                    - self.outer_padding[1]
                    + tw * self.outer_dilation[1]
                )
                if mw < 0 or mw >= middle_w:
                    continue
                outer_tap = th * kw_b + tw
                for qh in range(kh_a):
                    ih = (
                        mh * self.inner_stride[0]
                        - self.inner_padding[0]
                        + qh * self.inner_dilation[0]
                    )
                    if ih < 0 or ih >= input_h:
                        continue
                    for qw in range(kw_a):
                        iw = (
                            mw * self.inner_stride[1]
                            - self.inner_padding[1]
                            + qw * self.inner_dilation[1]
                        )
                        if iw < 0 or iw >= input_w:
                            continue
                        inner_tap = qh * kw_a + qw
                        values = self._coefficients[
                            outer_tap, inner_tap, start:stop
                        ]
                        for offset, value in enumerate(values):
                            value = float(value)
                            if value == 0.0:
                                continue
                            input_channel = int(self._pair_in[start + offset])
                            column = (
                                batch_offset
                                + input_channel * input_h * input_w
                                + ih * input_w
                                + iw
                            )
                            total = accumulator.get(column, 0.0) + value
                            if not math.isfinite(total):
                                raise PrototypeReject("row_sum_nonfinite")
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
                raise PrototypeReject("gather_result_nnz_limit")
            indices.extend(int(value) for value in columns)
            data.extend(float(value) for value in values)
            indptr[local_row + 1] = len(data)
        return sp.csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                indptr,
            ),
            shape=(selected.size, self.shape[1]),
        )

    def matvec(self, vector) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != self.shape[1] or not np.all(np.isfinite(values)):
            raise PrototypeReject("matvec_input_invalid")
        result = np.zeros(self.shape[0], dtype=np.float64)
        for row in range(self.shape[0]):
            columns, coefficients = self._row(row)
            result[row] = float(coefficients @ values[columns])
        if not np.all(np.isfinite(result)):
            raise PrototypeReject("matvec_output_nonfinite")
        return result

    def left_compose(self, Q, max_nnz: int) -> sp.csr_matrix:
        cap = _strict_nonnegative_int(max_nnz, name="max_nnz")
        if sp.issparse(Q):
            left = Q.tocsr().astype(np.float64, copy=True)
        else:
            raw = np.asarray(Q)
            if raw.dtype.kind not in "biuf" or raw.ndim != 2:
                raise PrototypeReject("left_factor_invalid")
            left = sp.csr_matrix(np.asarray(raw, dtype=np.float64))
        if left.shape[1] != self.shape[0]:
            raise PrototypeReject("left_factor_shape")
        left.sum_duplicates()
        left.sort_indices()
        left.eliminate_zeros()
        if not np.all(np.isfinite(left.data)):
            raise PrototypeReject("left_factor_nonfinite")

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
                    scaled = values * scale
                    nonzero = scaled != 0.0
                    if len(data) + int(nonzero.sum()) > cap:
                        raise PrototypeReject("left_result_nnz_limit")
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
            raise PrototypeReject("support_slice_would_expand_full_operator")
        gathered = self.gather_rows(support, max_nnz=cap)
        sliced_left = left[:, support]
        result = (sliced_left @ gathered).tocsr()
        result.sum_duplicates()
        result.sort_indices()
        result.eliminate_zeros()
        if result.nnz > cap:
            raise PrototypeReject("left_result_nnz_limit")
        if not np.all(np.isfinite(result.data)):
            raise PrototypeReject("left_result_nonfinite")
        return result


__all__ = [
    "BuildDecision",
    "ComposedStencilPrototype",
    "FrozenV1Limits",
    "GateContext",
    "PrototypeReject",
    "StencilEstimate",
]
