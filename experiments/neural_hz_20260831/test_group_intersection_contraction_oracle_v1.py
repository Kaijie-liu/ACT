"""Isolated group-intersection contraction oracle and focused tests.

This file is deliberately self-contained and is not imported by the ACT
runtime.  It exercises the exact channel contraction

    C[b, a, o, i] = sum_m B[b, o, m] * sigma[m] * A[a, m, i]

without forming full-channel tap matrices for either input convolution.  A
non-empty intersection of an outer middle-channel group and an inner output
group owns one coefficient block.  Within every tap pair, rank-one updates
are accumulated in ascending global middle-channel order.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np
import pytest


class OracleReject(ValueError):
    """A deterministic, fail-closed oracle rejection."""


@dataclass(frozen=True)
class AllocationRecord:
    """An array that the contraction oracle owns after compilation."""

    purpose: str
    shape: tuple[int, ...]
    entries: int


@dataclass(frozen=True)
class IntersectionBlock:
    """Coefficients owned by one non-empty middle-group intersection."""

    outer_group: int
    inner_group: int
    middle_start: int
    middle_stop: int
    output_start: int
    output_stop: int
    input_start: int
    input_stop: int
    contraction_products: int
    coefficients: np.ndarray

    @property
    def middle_count(self) -> int:
        return self.middle_stop - self.middle_start


@dataclass(frozen=True)
class ContractionResult:
    """The oracle payload plus an executable arithmetic proof ledger."""

    input_channels: int
    middle_channels: int
    output_channels: int
    inner_groups: int
    outer_groups: int
    inner_taps: int
    outer_taps: int
    blocks: tuple[IntersectionBlock, ...]
    actual_contraction_products: int
    intersection_formula_products: int
    gate_formula_products: int
    sigma_fold_products: int
    expected_sigma_fold_products: int
    rank_one_updates: int
    coefficient_entries: int
    allocations: tuple[AllocationRecord, ...]
    peak_temporary_entries: int
    transient_shapes: tuple[tuple[int, ...], ...]
    forbidden_inner_dense_tap_shape: tuple[int, int, int]
    forbidden_outer_dense_tap_shape: tuple[int, int, int]


def _positive_index(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise OracleReject(f"{name}_not_integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise OracleReject(f"{name}_not_integer") from exc
    if result <= 0:
        raise OracleReject(f"{name}_not_positive")
    return result


def _real_kernel(value, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 4 or raw.dtype.kind not in "biuf":
        raise OracleReject(f"{name}_kernel_layout")
    result = np.asarray(raw, dtype=np.float64)
    if any(int(size) <= 0 for size in result.shape):
        raise OracleReject(f"{name}_kernel_empty")
    if not np.all(np.isfinite(result)):
        raise OracleReject(f"{name}_kernel_nonfinite")
    return result


def _channel_scale(value, *, middle_channels: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "biuf":
        raise OracleReject("sigma_layout")
    result = np.asarray(raw, dtype=np.float64)
    if result.size != middle_channels:
        raise OracleReject("sigma_channels")
    if not np.all(np.isfinite(result)):
        raise OracleReject("sigma_nonfinite")
    return result


def contract_group_intersections(
    inner_kernel,
    outer_kernel,
    sigma,
    *,
    input_channels: int,
    inner_groups: int,
    outer_groups: int,
) -> ContractionResult:
    """Contract compact grouped kernels by exact group intersections.

    Kernels use the standard compact layouts
    ``[Cmid, Cin / gA, KAh, KAw]`` and
    ``[Cout, Cmid / gB, KBh, KBw]``.  The implementation owns no expanded
    ``[Ta, Cmid, Cin]`` or ``[Tb, Cout, Cmid]`` tap tensor.

    ``actual_contraction_products`` counts the scalar products executed by
    the rank-one updates.  Folding ``sigma`` into each compact outer column
    is counted separately in ``sigma_fold_products``.  Both counters are
    checked against closed-form identities before a result is returned.
    """

    inner = _real_kernel(inner_kernel, name="inner")
    outer = _real_kernel(outer_kernel, name="outer")
    cin = _positive_index(input_channels, name="input_channels")
    ga_count = _positive_index(inner_groups, name="inner_groups")
    gb_count = _positive_index(outer_groups, name="outer_groups")

    cmid = int(inner.shape[0])
    cout = int(outer.shape[0])
    if cin % ga_count:
        raise OracleReject("input_channels_not_divisible_by_inner_groups")
    if cmid % ga_count:
        raise OracleReject("middle_channels_not_divisible_by_inner_groups")
    if cmid % gb_count:
        raise OracleReject("middle_channels_not_divisible_by_outer_groups")
    if cout % gb_count:
        raise OracleReject("output_channels_not_divisible_by_outer_groups")
    if int(inner.shape[1]) * ga_count != cin:
        raise OracleReject("inner_compact_input_channels")
    if int(outer.shape[1]) * gb_count != cmid:
        raise OracleReject("outer_compact_middle_channels")

    scale = _channel_scale(sigma, middle_channels=cmid)
    inner_output_per_group = cmid // ga_count
    inner_input_per_group = cin // ga_count
    outer_input_per_group = cmid // gb_count
    outer_output_per_group = cout // gb_count
    inner_taps = int(inner.shape[2] * inner.shape[3])
    outer_taps = int(outer.shape[2] * outer.shape[3])

    blocks: list[IntersectionBlock] = []
    allocations: list[AllocationRecord] = []
    actual_products = 0
    intersection_products = 0
    sigma_fold_products = 0
    rank_one_updates = 0
    coefficient_entries = 0
    peak_temporary_entries = 0
    transient_shapes: set[tuple[int, ...]] = set()

    # gb-major, ga-minor is the canonical intersection order.  For a fixed
    # (outer tap, inner tap), the enclosing m loop visits global channels in
    # ascending order, so every coefficient has one deterministic sum order.
    for gb in range(gb_count):
        outer_middle_start = gb * outer_input_per_group
        outer_middle_stop = outer_middle_start + outer_input_per_group
        output_start = gb * outer_output_per_group
        output_stop = output_start + outer_output_per_group
        covered_middle_channels = 0

        for ga in range(ga_count):
            inner_middle_start = ga * inner_output_per_group
            inner_middle_stop = inner_middle_start + inner_output_per_group
            middle_start = max(outer_middle_start, inner_middle_start)
            middle_stop = min(outer_middle_stop, inner_middle_stop)
            if middle_start >= middle_stop:
                continue

            input_start = ga * inner_input_per_group
            input_stop = input_start + inner_input_per_group
            coefficients = np.zeros(
                (
                    outer_taps,
                    inner_taps,
                    outer_output_per_group,
                    inner_input_per_group,
                ),
                dtype=np.float64,
            )
            allocations.append(
                AllocationRecord(
                    purpose=f"coefficients_gb{gb}_ga{ga}",
                    shape=tuple(int(v) for v in coefficients.shape),
                    entries=int(coefficients.size),
                )
            )
            block_actual_products = 0

            for outer_tap in range(outer_taps):
                outer_kh, outer_kw = divmod(
                    outer_tap, int(outer.shape[3])
                )
                for middle_channel in range(middle_start, middle_stop):
                    outer_local_middle = (
                        middle_channel - outer_middle_start
                    )
                    scaled_outer_column = np.multiply(
                        outer[
                            output_start:output_stop,
                            outer_local_middle,
                            outer_kh,
                            outer_kw,
                        ],
                        scale[middle_channel],
                    )
                    sigma_fold_products += outer_output_per_group
                    transient_shapes.add((outer_output_per_group,))

                    for inner_tap in range(inner_taps):
                        inner_kh, inner_kw = divmod(
                            inner_tap, int(inner.shape[3])
                        )
                        inner_row = inner[
                            middle_channel, :, inner_kh, inner_kw
                        ]
                        rank_one = np.multiply(
                            scaled_outer_column[:, np.newaxis],
                            inner_row[np.newaxis, :],
                        )
                        np.add(
                            coefficients[outer_tap, inner_tap],
                            rank_one,
                            out=coefficients[outer_tap, inner_tap],
                        )
                        products = (
                            outer_output_per_group
                            * inner_input_per_group
                        )
                        block_actual_products += products
                        actual_products += products
                        rank_one_updates += 1
                        transient_shapes.add(
                            (
                                outer_output_per_group,
                                inner_input_per_group,
                            )
                        )
                        peak_temporary_entries = max(
                            peak_temporary_entries,
                            outer_output_per_group + products,
                        )

            block_formula_products = (
                outer_taps
                * inner_taps
                * (middle_stop - middle_start)
                * outer_output_per_group
                * inner_input_per_group
            )
            if block_actual_products != block_formula_products:
                raise AssertionError("block_product_ledger_mismatch")
            intersection_products += block_formula_products
            covered_middle_channels += middle_stop - middle_start
            coefficients.flags.writeable = False
            blocks.append(
                IntersectionBlock(
                    outer_group=gb,
                    inner_group=ga,
                    middle_start=middle_start,
                    middle_stop=middle_stop,
                    output_start=output_start,
                    output_stop=output_stop,
                    input_start=input_start,
                    input_stop=input_stop,
                    contraction_products=block_actual_products,
                    coefficients=coefficients,
                )
            )
            coefficient_entries += int(coefficients.size)

        if covered_middle_channels != outer_input_per_group:
            raise AssertionError("intersection_partition_mismatch")

    # For every outer group, its Cmid/gB middle channels are partitioned by
    # the inner groups.  Hence
    #   sum_{gb,ga} |J| (Cout/gB) (Cin/gA) Ta Tb
    # = Ta Tb Cout (Cmid/gB) (Cin/gA).
    gate_formula_products = (
        outer_taps
        * inner_taps
        * cout
        * outer_input_per_group
        * inner_input_per_group
    )
    expected_sigma_fold_products = int(outer.size)
    if not (
        actual_products
        == intersection_products
        == gate_formula_products
    ):
        raise AssertionError("gate_product_identity_mismatch")
    if sigma_fold_products != expected_sigma_fold_products:
        raise AssertionError("sigma_fold_product_identity_mismatch")

    return ContractionResult(
        input_channels=cin,
        middle_channels=cmid,
        output_channels=cout,
        inner_groups=ga_count,
        outer_groups=gb_count,
        inner_taps=inner_taps,
        outer_taps=outer_taps,
        blocks=tuple(blocks),
        actual_contraction_products=actual_products,
        intersection_formula_products=intersection_products,
        gate_formula_products=gate_formula_products,
        sigma_fold_products=sigma_fold_products,
        expected_sigma_fold_products=expected_sigma_fold_products,
        rank_one_updates=rank_one_updates,
        coefficient_entries=coefficient_entries,
        allocations=tuple(allocations),
        peak_temporary_entries=peak_temporary_entries,
        transient_shapes=tuple(sorted(transient_shapes)),
        forbidden_inner_dense_tap_shape=(inner_taps, cmid, cin),
        forbidden_outer_dense_tap_shape=(outer_taps, cout, cmid),
    )


# ---------------------------------------------------------------------------
# Focused tests.  Full-channel tap matrices occur only in the test reference
# below; contract_group_intersections never calls this helper.


@dataclass(frozen=True)
class _Case:
    name: str
    input_channels: int
    middle_channels: int
    output_channels: int
    inner_groups: int
    outer_groups: int
    inner_kernel_shape: tuple[int, int]
    outer_kernel_shape: tuple[int, int]


def _dyadic_kernel(shape, *, offset: int) -> np.ndarray:
    size = int(np.prod(shape))
    values = ((np.arange(size, dtype=np.int64) + offset) % 13 - 6) / 16.0
    return values.reshape(shape)


def _dyadic_sigma(channels: int) -> np.ndarray:
    pattern = np.array(
        [0.0, -2.0, 1.0, -0.5, 0.25, -0.25, 2.0, 0.5],
        dtype=np.float64,
    )
    return np.resize(pattern, channels)


CASES = (
    _Case("ordinary", 3, 4, 2, 1, 1, (2, 2), (2, 1)),
    _Case("aligned_groups", 4, 8, 6, 2, 2, (1, 2), (2, 1)),
    _Case("misaligned_groups", 6, 12, 8, 3, 4, (2, 1), (1, 2)),
    # Depthwise inner convolution with channel multiplier two.
    _Case("inner_depthwise_multiplier", 3, 6, 4, 3, 2, (2, 2), (1, 2)),
    # Depthwise outer convolution with channel multiplier two.
    _Case("outer_depthwise_multiplier", 4, 4, 8, 2, 4, (2, 1), (2, 2)),
)


def _case_payload(case: _Case, *, dyadic: bool = True):
    inner_shape = (
        case.middle_channels,
        case.input_channels // case.inner_groups,
        *case.inner_kernel_shape,
    )
    outer_shape = (
        case.output_channels,
        case.middle_channels // case.outer_groups,
        *case.outer_kernel_shape,
    )
    if dyadic:
        inner = _dyadic_kernel(inner_shape, offset=1)
        outer = _dyadic_kernel(outer_shape, offset=5)
        scale = _dyadic_sigma(case.middle_channels)
    else:
        rng = np.random.default_rng(20260831)
        inner = rng.normal(size=inner_shape)
        outer = rng.normal(size=outer_shape)
        scale = rng.normal(size=case.middle_channels)
    return inner, outer, scale


def _expanded_taps_reference(
    compact_kernel: np.ndarray,
    *,
    input_channels: int,
    groups: int,
) -> np.ndarray:
    """Reference-only expansion; it is intentionally outside the oracle."""

    output_channels, input_per_group, kernel_h, kernel_w = (
        int(v) for v in compact_kernel.shape
    )
    output_per_group = output_channels // groups
    taps = np.zeros(
        (kernel_h * kernel_w, output_channels, input_channels),
        dtype=np.float64,
    )
    for output_channel in range(output_channels):
        group = output_channel // output_per_group
        input_start = group * input_per_group
        input_stop = input_start + input_per_group
        taps[:, output_channel, input_start:input_stop] = (
            compact_kernel[output_channel]
            .transpose(1, 2, 0)
            .reshape(kernel_h * kernel_w, input_per_group)
        )
    return taps


def _dense_reference(
    case: _Case,
    inner: np.ndarray,
    outer: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    inner_taps = _expanded_taps_reference(
        inner,
        input_channels=case.input_channels,
        groups=case.inner_groups,
    )
    outer_taps = _expanded_taps_reference(
        outer,
        input_channels=case.middle_channels,
        groups=case.outer_groups,
    )
    result = np.zeros(
        (
            outer_taps.shape[0],
            inner_taps.shape[0],
            case.output_channels,
            case.input_channels,
        ),
        dtype=np.float64,
    )
    for outer_tap in range(outer_taps.shape[0]):
        scaled_outer = outer_taps[outer_tap] * scale[np.newaxis, :]
        for inner_tap in range(inner_taps.shape[0]):
            result[outer_tap, inner_tap] = (
                scaled_outer @ inner_taps[inner_tap]
            )
    return result


def _assemble_result(result: ContractionResult) -> np.ndarray:
    dense = np.zeros(
        (
            result.outer_taps,
            result.inner_taps,
            result.output_channels,
            result.input_channels,
        ),
        dtype=np.float64,
    )
    for block in result.blocks:
        dense[
            :,
            :,
            block.output_start : block.output_stop,
            block.input_start : block.input_stop,
        ] = block.coefficients
    return dense


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_dyadic_oracle_matches_dense_real_reference_exactly(case):
    inner, outer, scale = _case_payload(case)
    result = contract_group_intersections(
        inner,
        outer,
        scale,
        input_channels=case.input_channels,
        inner_groups=case.inner_groups,
        outer_groups=case.outer_groups,
    )
    reference = _dense_reference(case, inner, outer, scale)

    assert np.array_equal(_assemble_result(result), reference)
    expected_gate = (
        result.inner_taps
        * result.outer_taps
        * case.output_channels
        * (case.middle_channels // case.outer_groups)
        * (case.input_channels // case.inner_groups)
    )
    assert result.actual_contraction_products == expected_gate
    assert result.intersection_formula_products == expected_gate
    assert result.gate_formula_products == expected_gate
    assert result.sigma_fold_products == outer.size
    assert result.expected_sigma_fold_products == outer.size
    assert sum(block.contraction_products for block in result.blocks) == (
        expected_gate
    )
    assert sum(record.entries for record in result.allocations) == (
        result.coefficient_entries
    )
    assert all(not block.coefficients.flags.writeable for block in result.blocks)


def test_misaligned_intersections_are_canonical_and_partition_middle_groups():
    case = next(item for item in CASES if item.name == "misaligned_groups")
    inner, outer, scale = _case_payload(case)
    result = contract_group_intersections(
        inner,
        outer,
        scale,
        input_channels=case.input_channels,
        inner_groups=case.inner_groups,
        outer_groups=case.outer_groups,
    )
    intersections = [
        (
            block.outer_group,
            block.inner_group,
            block.middle_start,
            block.middle_stop,
        )
        for block in result.blocks
    ]
    assert intersections == [
        (0, 0, 0, 3),
        (1, 0, 3, 4),
        (1, 1, 4, 6),
        (2, 1, 6, 8),
        (2, 2, 8, 9),
        (3, 2, 9, 12),
    ]


def test_depthwise_channel_multipliers_are_represented_without_special_case():
    inner_case = next(
        item for item in CASES if item.name == "inner_depthwise_multiplier"
    )
    outer_case = next(
        item for item in CASES if item.name == "outer_depthwise_multiplier"
    )
    assert inner_case.inner_groups == inner_case.input_channels
    assert inner_case.middle_channels // inner_case.input_channels == 2
    assert outer_case.outer_groups == outer_case.middle_channels
    assert outer_case.output_channels // outer_case.middle_channels == 2


def test_zero_and_negative_sigma_are_exact_and_still_charged_to_work_ledger():
    case = CASES[0]
    inner, outer, scale = _case_payload(case)
    assert np.any(scale == 0)
    assert np.any(scale < 0)
    result = contract_group_intersections(
        inner,
        outer,
        scale,
        input_channels=case.input_channels,
        inner_groups=case.inner_groups,
        outer_groups=case.outer_groups,
    )
    assert result.actual_contraction_products == result.gate_formula_products
    assert result.sigma_fold_products == outer.size
    assert np.array_equal(
        _assemble_result(result), _dense_reference(case, inner, outer, scale)
    )


def test_non_dyadic_compilation_is_bitwise_deterministic_and_equivalent():
    case = next(item for item in CASES if item.name == "misaligned_groups")
    inner, outer, scale = _case_payload(case, dyadic=False)
    kwargs = dict(
        input_channels=case.input_channels,
        inner_groups=case.inner_groups,
        outer_groups=case.outer_groups,
    )
    first = contract_group_intersections(inner, outer, scale, **kwargs)
    second = contract_group_intersections(inner, outer, scale, **kwargs)

    first_structure = [
        (
            block.outer_group,
            block.inner_group,
            block.middle_start,
            block.middle_stop,
            block.output_start,
            block.output_stop,
            block.input_start,
            block.input_stop,
        )
        for block in first.blocks
    ]
    second_structure = [
        (
            block.outer_group,
            block.inner_group,
            block.middle_start,
            block.middle_stop,
            block.output_start,
            block.output_stop,
            block.input_start,
            block.input_stop,
        )
        for block in second.blocks
    ]
    assert first_structure == second_structure
    assert first.actual_contraction_products == second.actual_contraction_products
    assert all(
        np.array_equal(left.coefficients, right.coefficients)
        for left, right in zip(first.blocks, second.blocks, strict=True)
    )
    assert np.allclose(
        _assemble_result(first),
        _dense_reference(case, inner, outer, scale),
        rtol=2e-13,
        atol=2e-13,
    )


def test_oracle_never_allocates_full_channel_input_tap_matrices(monkeypatch):
    case = next(item for item in CASES if item.name == "misaligned_groups")
    inner, outer, scale = _case_payload(case)
    inner_taps = case.inner_kernel_shape[0] * case.inner_kernel_shape[1]
    outer_taps = case.outer_kernel_shape[0] * case.outer_kernel_shape[1]
    forbidden = {
        (inner_taps, case.middle_channels, case.input_channels),
        (outer_taps, case.output_channels, case.middle_channels),
    }

    # Guard the explicit NumPy allocation entry points used by this prototype.
    for constructor_name in ("zeros", "empty", "ones", "full"):
        original = getattr(np, constructor_name)

        def guarded(shape, *args, _original=original, **kwargs):
            try:
                normalized = tuple(int(value) for value in shape)
            except TypeError:
                normalized = (int(shape),)
            if normalized in forbidden:
                raise AssertionError(f"expanded_tap_allocation:{normalized}")
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, constructor_name, guarded)

    result = contract_group_intersections(
        inner,
        outer,
        scale,
        input_channels=case.input_channels,
        inner_groups=case.inner_groups,
        outer_groups=case.outer_groups,
    )
    assert result.forbidden_inner_dense_tap_shape in forbidden
    assert result.forbidden_outer_dense_tap_shape in forbidden
    assert all(record.shape not in forbidden for record in result.allocations)
    assert all(len(shape) <= 2 for shape in result.transient_shapes)
    max_outer_block = case.output_channels // case.outer_groups
    max_inner_block = case.input_channels // case.inner_groups
    assert result.peak_temporary_entries <= (
        max_outer_block + max_outer_block * max_inner_block
    )


def test_invalid_geometry_and_nonfinite_values_fail_closed():
    case = CASES[0]
    inner, outer, scale = _case_payload(case)
    with pytest.raises(OracleReject, match="outer_compact_middle_channels"):
        contract_group_intersections(
            inner,
            outer[:, :-1],
            scale,
            input_channels=case.input_channels,
            inner_groups=case.inner_groups,
            outer_groups=case.outer_groups,
        )
    bad_scale = scale.copy()
    bad_scale[0] = np.nan
    with pytest.raises(OracleReject, match="sigma_nonfinite"):
        contract_group_intersections(
            inner,
            outer,
            bad_scale,
            input_channels=case.input_channels,
            inner_groups=case.inner_groups,
            outer_groups=case.outer_groups,
        )
    with pytest.raises(
        OracleReject, match="input_channels_not_divisible_by_inner_groups"
    ):
        contract_group_intersections(
            inner,
            outer,
            scale,
            input_channels=case.input_channels,
            inner_groups=2,
            outer_groups=case.outer_groups,
        )
