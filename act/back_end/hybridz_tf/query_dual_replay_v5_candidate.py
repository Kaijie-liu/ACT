#===- query_dual_replay_v5_candidate.py - Scalar Conv guard prototype ---===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
"""Non-authoritative V5 prototype for a dense Conv2D scalar error guard.

This module is deliberately isolated from the proof-producing replay path.
It preserves the dense Conv2D nominal kernel-offset/channel-GEMM sequence,
but replaces the componentwise coefficient-radius tensor with two scalar
guards per query:

``D``
    channel-dot error, aggregated against an outward predecessor support;

``A``
    rounding error from adding each kernel-offset term to the nominal
    transpose-convolution accumulator.

For predecessor support ``m_i = max(abs(lb_i), abs(ub_i))``, output
coefficient ``c``, and one group/offset ``o``, preparation computes

``S_o[co,p] >= sum_ci abs(W_o[co,ci]) * m[ci,target_o(p)]``.

Replay then forms the following outward scalar masses:

``D_o[q] >= sum_(p,co) abs(c[q,co,p]) * S_o[co,p]``

``A_o[q] >= sum_(p,ci) m[ci,target_o(p)] *``
``(abs(accumulator_before[q,ci,p]) + abs(term_o[q,ci,p]))``.

If ``gamma_D,tau_D`` are the frozen channel-dot Higham/subnormal constants
and ``gamma_A,tau_A`` are the frozen binary-add constants, the applied guard
is

``sum_o (gamma_D*D_o + tau_D*sum(m_o))``
``+ sum_o (gamma_A*A_o + tau_A*sum(m_o))``,

with every product and sum rounded outward.  ``sum(m_o)`` is deliberately
repeated once per computed channel-dot result or offset addition, so padding,
stride, dilation, groups, and subnormal-error multiplicities are explicit.
Here ``tau_k = up(k*eta/(1-k*u))`` rather than merely ``k*eta``.

The sparse scatter branch is explicitly out of scope and must continue to use
the frozen replay implementation.  Every public result here therefore carries
``proof_authority=False``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen


@dataclass(frozen=True)
class _OffsetSupport:
    group: int
    kh: int
    kw: int
    co_start: int
    co_end: int
    ci_start: int
    ci_end: int
    output_h_indices: np.ndarray
    output_w_indices: np.ndarray
    targets: np.ndarray
    support_flat: np.ndarray
    channel_support_flat: np.ndarray
    support_sum_upper: float


@dataclass(frozen=True)
class DenseConvScalarGuardPlan:
    """Owned CPU-f64 support transform for one frozen dense Conv2D layer."""

    layer_id: int
    input_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    weight: np.ndarray
    support: np.ndarray
    offsets: Tuple[_OffsetSupport, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class DenseConvScalarGuardResult:
    """Research result; never proof authority."""

    coefficient: np.ndarray
    scalar_guard: np.ndarray
    channel_dot_guard: np.ndarray
    accumulation_guard: np.ndarray
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("the isolated V5 candidate cannot issue authority")


def _outward_roundoff_parameters(operations: int) -> Tuple[float, float]:
    """Return ``gamma_k`` and ``tau_k`` with one outward conversion."""

    count = max(1, int(operations))
    product = np.longdouble(count) * np.longdouble(frozen._U)
    if product >= np.longdouble(0.5):
        frozen._fail("V5_INVALID_GUARD", "roundoff operation count is too large")
    denominator = np.longdouble(1.0) - product
    gamma = float(frozen._longdouble_to_f64_up(product / denominator))
    tau = float(
        frozen._longdouble_to_f64_up(
            np.longdouble(count)
            * np.longdouble(frozen._ETA)
            / denominator
        )
    )
    if (
        not math.isfinite(gamma)
        or not math.isfinite(tau)
        or gamma < 0.0
        or tau < 0.0
    ):
        frozen._fail("V5_INVALID_GUARD", "invalid outward roundoff parameters")
    return gamma, tau


def _immutable_i64(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    frozen_array = np.frombuffer(
        array.tobytes(order="C"), dtype=np.int64
    ).reshape(array.shape)
    frozen_array.setflags(write=False)
    return frozen_array


def _i64_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<i8\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _offset_manifest(offset: _OffsetSupport) -> Mapping[str, Any]:
    return {
        "group": offset.group,
        "kh": offset.kh,
        "kw": offset.kw,
        "co": [offset.co_start, offset.co_end],
        "ci": [offset.ci_start, offset.ci_end],
        "output_h_indices_sha256": _i64_digest(
            offset.output_h_indices
        ),
        "output_w_indices_sha256": _i64_digest(
            offset.output_w_indices
        ),
        "targets_sha256": _i64_digest(offset.targets),
        "support_sha256": frozen._array_digest(offset.support_flat),
        "channel_support_sha256": frozen._array_digest(
            offset.channel_support_flat
        ),
        "support_sum_upper_hex": float(
            offset.support_sum_upper
        ).hex(),
        "channel_tau_multiplicity": "support_sum_upper_once_per_channel_dot",
        "addition_tau_multiplicity": "support_sum_upper_once_per_offset_add",
    }


def _plan_manifest_body(
    *,
    layer_id: int,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    weight: np.ndarray,
    support: np.ndarray,
    offsets: Tuple[_OffsetSupport, ...],
) -> Mapping[str, Any]:
    dot_operations = 2 * (output_shape[0] // groups) + 2
    dot_gamma, dot_tau = _outward_roundoff_parameters(dot_operations)
    add_gamma, add_tau = _outward_roundoff_parameters(2)
    return {
        "schema": "act.query_dual_replay_v5_dense_conv_candidate.v1",
        "layer_id": int(layer_id),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "weight_sha256": frozen._array_digest(weight),
        "support_sha256": frozen._array_digest(support),
        "offset_count": len(offsets),
        "offsets": [_offset_manifest(offset) for offset in offsets],
        "channel_dot_operations": dot_operations,
        "channel_dot_gamma_hex": dot_gamma.hex(),
        "channel_dot_tau_hex": dot_tau.hex(),
        "addition_operations": 2,
        "addition_gamma_hex": add_gamma.hex(),
        "addition_tau_hex": add_tau.hex(),
        "tau_definition": "up(k*eta/(1-k*u))",
        "guard": "two_stage_channel_dot_D_plus_offset_accumulation_A",
        "sparse_branch": "unchanged_frozen_replay_only",
        "proof_authority": False,
    }


def _validate_plan_structure(plan: DenseConvScalarGuardPlan) -> None:
    if (
        len(plan.input_shape) != 3
        or len(plan.output_shape) != 3
        or len(plan.stride) != 2
        or len(plan.padding) != 2
        or len(plan.dilation) != 2
        or any(value <= 0 for value in plan.input_shape)
        or any(value <= 0 for value in plan.output_shape)
        or any(value <= 0 for value in plan.stride)
        or any(value < 0 for value in plan.padding)
        or any(value <= 0 for value in plan.dilation)
        or plan.groups <= 0
        or plan.output_shape[0] % plan.groups
        or plan.input_shape[0] % plan.groups
    ):
        frozen._fail("V5_INVALID_PLAN", "invalid plan geometry")
    out_c, out_h, out_w = plan.output_shape
    in_c, in_h, in_w = plan.input_shape
    out_per_group = out_c // plan.groups
    in_per_group = in_c // plan.groups
    if (
        plan.weight.dtype != np.float64
        or plan.weight.ndim != 4
        or plan.weight.shape[0] != out_c
        or plan.weight.shape[1] != in_per_group
        or plan.weight.shape[2] <= 0
        or plan.weight.shape[3] <= 0
        or plan.weight.flags.writeable
        or plan.support.dtype != np.float64
        or plan.support.shape != (in_c * in_h * in_w,)
        or plan.support.flags.writeable
        or np.any(plan.support < 0.0)
        or not np.all(np.isfinite(plan.support))
    ):
        frozen._fail("V5_INVALID_PLAN", "invalid plan weight or support")
    expected_keys = []
    for group in range(plan.groups):
        for kh in range(plan.weight.shape[2]):
            input_h_indices = (
                np.arange(out_h, dtype=np.int64) * plan.stride[0]
                - plan.padding[0]
                + kh * plan.dilation[0]
            )
            valid_h = (input_h_indices >= 0) & (input_h_indices < in_h)
            if not np.any(valid_h):
                continue
            for kw in range(plan.weight.shape[3]):
                input_w_indices = (
                    np.arange(out_w, dtype=np.int64) * plan.stride[1]
                    - plan.padding[1]
                    + kw * plan.dilation[1]
                )
                valid_w = (input_w_indices >= 0) & (input_w_indices < in_w)
                if np.any(valid_w):
                    expected_keys.append((group, kh, kw))
    if [(value.group, value.kh, value.kw) for value in plan.offsets] != expected_keys:
        frozen._fail("V5_INVALID_PLAN", "plan offset coverage changed")
    for offset in plan.offsets:
        expected_co_start = offset.group * out_per_group
        expected_ci_start = offset.group * in_per_group
        expected_output_h = np.flatnonzero(
            (
                np.arange(out_h, dtype=np.int64) * plan.stride[0]
                - plan.padding[0]
                + offset.kh * plan.dilation[0]
                >= 0
            )
            & (
                np.arange(out_h, dtype=np.int64) * plan.stride[0]
                - plan.padding[0]
                + offset.kh * plan.dilation[0]
                < in_h
            )
        )
        expected_output_w = np.flatnonzero(
            (
                np.arange(out_w, dtype=np.int64) * plan.stride[1]
                - plan.padding[1]
                + offset.kw * plan.dilation[1]
                >= 0
            )
            & (
                np.arange(out_w, dtype=np.int64) * plan.stride[1]
                - plan.padding[1]
                + offset.kw * plan.dilation[1]
                < in_w
            )
        )
        input_h = (
            expected_output_h * plan.stride[0]
            - plan.padding[0]
            + offset.kh * plan.dilation[0]
        )
        input_w = (
            expected_output_w * plan.stride[1]
            - plan.padding[1]
            + offset.kw * plan.dilation[1]
        )
        expected_targets = (
            input_h[:, None] * in_w + input_w[None, :]
        ).reshape(-1)
        positions = expected_targets.size
        if (
            offset.co_start != expected_co_start
            or offset.co_end != expected_co_start + out_per_group
            or offset.ci_start != expected_ci_start
            or offset.ci_end != expected_ci_start + in_per_group
            or offset.output_h_indices.dtype != np.int64
            or offset.output_w_indices.dtype != np.int64
            or offset.targets.dtype != np.int64
            or offset.output_h_indices.flags.writeable
            or offset.output_w_indices.flags.writeable
            or offset.targets.flags.writeable
            or not np.array_equal(
                offset.output_h_indices, expected_output_h
            )
            or not np.array_equal(
                offset.output_w_indices, expected_output_w
            )
            or not np.array_equal(offset.targets, expected_targets)
            or offset.support_flat.dtype != np.float64
            or offset.support_flat.shape != (in_per_group * positions,)
            or offset.support_flat.flags.writeable
            or offset.channel_support_flat.dtype != np.float64
            or offset.channel_support_flat.shape
            != (out_per_group * positions,)
            or offset.channel_support_flat.flags.writeable
            or np.any(offset.support_flat < 0.0)
            or np.any(offset.channel_support_flat < 0.0)
            or not np.all(np.isfinite(offset.support_flat))
            or not np.all(np.isfinite(offset.channel_support_flat))
            or not math.isfinite(offset.support_sum_upper)
            or offset.support_sum_upper < 0.0
        ):
            frozen._fail("V5_INVALID_PLAN", "invalid plan offset record")


def _upper_nonnegative_row_dot(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Outward upper bound on each exact nonnegative row dot."""

    if (
        left.ndim != 2
        or right.ndim != 1
        or left.shape[1] != right.size
        or np.any(left < 0.0)
        or np.any(right < 0.0)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        frozen._fail(
            "V5_INVALID_MASS", "nonnegative row-dot operands are invalid"
        )
    operations = 2 * int(left.shape[1]) + 2
    gamma, tau = _outward_roundoff_parameters(operations)
    nominal = np.asarray(left @ right, dtype=np.float64)
    frozen._require_finite(nominal, where="V5 nonnegative row-dot nominal")
    # If |fl(dot)-dot| <= gamma*dot+tau and dot is nonnegative, then
    # dot <= (fl(dot)+tau)/(1-gamma).
    upper = frozen._upper_gamma_enclosure(nominal, gamma, tau)
    exact_zero = ~np.any(
        (left != 0.0) & (right.reshape(1, -1) != 0.0), axis=1
    )
    upper[exact_zero] = 0.0
    frozen._require_finite(upper, where="V5 nonnegative row-dot upper")
    return np.ascontiguousarray(upper, dtype=np.float64)


def _upper_nonnegative_matrix_product(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Outward upper bound on an exact nonnegative matrix product."""

    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
        or np.any(left < 0.0)
        or np.any(right < 0.0)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        frozen._fail(
            "V5_INVALID_MASS", "nonnegative matrix operands are invalid"
        )
    operations = 2 * int(left.shape[1]) + 2
    gamma, tau = _outward_roundoff_parameters(operations)
    nominal = np.asarray(left @ right, dtype=np.float64)
    frozen._require_finite(
        nominal, where="V5 nonnegative matrix-product nominal"
    )
    upper = frozen._upper_gamma_enclosure(nominal, gamma, tau)
    exact_zero = ~(
        np.any(left != 0.0, axis=1).reshape(-1, 1)
        & np.any(right != 0.0, axis=0).reshape(1, -1)
    )
    upper[exact_zero] = 0.0
    frozen._require_finite(
        upper, where="V5 nonnegative matrix-product upper"
    )
    return np.ascontiguousarray(upper, dtype=np.float64)


def _upper_scaled_mass(
    mass_upper: np.ndarray,
    *,
    gamma_upper: float,
    underflow_upper: float,
    support_sum_upper: float,
) -> np.ndarray:
    """Outward ``gamma*M + underflow*sum(support)``."""

    if (
        mass_upper.ndim != 1
        or np.any(mass_upper < 0.0)
        or gamma_upper < 0.0
        or underflow_upper < 0.0
        or support_sum_upper < 0.0
    ):
        frozen._fail("V5_INVALID_GUARD", "negative scalar-guard input")
    value = (
        np.longdouble(gamma_upper)
        * np.asarray(mass_upper, dtype=np.longdouble)
        + np.longdouble(underflow_upper)
        * np.longdouble(support_sum_upper)
    )
    result = frozen._longdouble_to_f64_up(value)
    frozen._require_finite(result, where="V5 scaled scalar guard")
    return np.ascontiguousarray(result, dtype=np.float64)


def _support_sum_upper(support_flat: np.ndarray) -> float:
    value = _upper_nonnegative_row_dot(
        support_flat.reshape(1, -1),
        np.ones(support_flat.size, dtype=np.float64),
    )
    return float(value[0])


def prepare_dense_conv_scalar_guard(
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    *,
    deadline: frozen._Deadline,
) -> DenseConvScalarGuardPlan:
    """Precompute the outward support transform for every valid offset.

    The transform for an offset is

    ``D[co,p] >= sum_ci abs(W[co,ci,kh,kw]) * support[ci,target(p)]``.

    It is independent of the query batch and therefore reusable for all
    objectives replayed against this exact predecessor box.
    """

    deadline.check(force=True)
    if not isinstance(layer, frozen._FrozenLayer) or layer.kind != "CONV2D":
        frozen._fail("V5_INVALID_LAYER", "candidate requires frozen CONV2D")
    p = layer.params
    weight = frozen._immutable_f64_array(p["weight"], name="V5 weight")
    input_shape = tuple(int(value) for value in p["input_shape"])
    output_shape = tuple(int(value) for value in p["output_shape"])
    stride = tuple(int(value) for value in p["stride"])
    padding = tuple(int(value) for value in p["padding"])
    dilation = tuple(int(value) for value in p["dilation"])
    groups = int(p["groups"])
    if (
        len(input_shape) != 3
        or len(output_shape) != 3
        or len(stride) != 2
        or len(padding) != 2
        or len(dilation) != 2
        or any(value <= 0 for value in input_shape)
        or any(value <= 0 for value in output_shape)
        or any(value <= 0 for value in stride)
        or any(value < 0 for value in padding)
        or any(value <= 0 for value in dilation)
        or groups <= 0
        or weight.ndim != 4
        or weight.shape[0] != output_shape[0]
        or output_shape[0] % groups
        or input_shape[0] != weight.shape[1] * groups
    ):
        frozen._fail("V5_INVALID_LAYER", "invalid Conv2D geometry")
    frozen._conv_output_padding(layer)
    lb = frozen._as_f64_array(predecessor_box.lb, name="V5 support lb").reshape(-1)
    ub = frozen._as_f64_array(predecessor_box.ub, name="V5 support ub").reshape(-1)
    if lb.shape != ub.shape or lb.size != int(np.prod(input_shape)):
        frozen._fail("V5_INVALID_SUPPORT", "predecessor box width mismatch")
    if np.any(lb > ub):
        frozen._fail("V5_INVALID_SUPPORT", "predecessor box has lb > ub")
    support = frozen._immutable_f64_array(
        np.maximum(np.abs(lb), np.abs(ub)), name="V5 outward support"
    )

    out_c, out_h, out_w = output_shape
    in_c, in_h, in_w = input_shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    out_per_group = out_c // groups
    in_per_group = in_c // groups
    offsets = []
    for group in range(groups):
        co_start = group * out_per_group
        co_end = co_start + out_per_group
        ci_start = group * in_per_group
        ci_end = ci_start + in_per_group
        for kh in range(int(weight.shape[2])):
            deadline.check(force=True)
            input_h_indices = (
                np.arange(out_h, dtype=np.int64) * stride_h
                - padding_h
                + kh * dilation_h
            )
            valid_h = (input_h_indices >= 0) & (input_h_indices < in_h)
            if not np.any(valid_h):
                continue
            output_h_indices = np.flatnonzero(valid_h)
            input_h_indices = input_h_indices[valid_h]
            for kw in range(int(weight.shape[3])):
                deadline.check(force=True)
                input_w_indices = (
                    np.arange(out_w, dtype=np.int64) * stride_w
                    - padding_w
                    + kw * dilation_w
                )
                valid_w = (input_w_indices >= 0) & (input_w_indices < in_w)
                if not np.any(valid_w):
                    continue
                output_w_indices = np.flatnonzero(valid_w)
                input_w_indices = input_w_indices[valid_w]
                targets = (
                    input_h_indices[:, None] * in_w
                    + input_w_indices[None, :]
                ).reshape(-1)
                support_selected = np.ascontiguousarray(
                    support.reshape(in_c, -1)[ci_start:ci_end, :][
                        :, targets
                    ]
                )
                weight_abs = np.ascontiguousarray(
                    np.abs(weight[co_start:co_end, :, kh, kw])
                )
                channel_upper = _upper_nonnegative_matrix_product(
                    weight_abs, support_selected
                )
                support_flat = frozen._immutable_f64_array(
                    support_selected.reshape(-1),
                    name="V5 offset support",
                )
                channel_flat = frozen._immutable_f64_array(
                    channel_upper.T.reshape(-1),
                    name="V5 channel support",
                )
                offsets.append(
                    _OffsetSupport(
                        group=group,
                        kh=kh,
                        kw=kw,
                        co_start=co_start,
                        co_end=co_end,
                        ci_start=ci_start,
                        ci_end=ci_end,
                        output_h_indices=_immutable_i64(
                            output_h_indices
                        ),
                        output_w_indices=_immutable_i64(
                            output_w_indices
                        ),
                        targets=_immutable_i64(targets),
                        support_flat=support_flat,
                        channel_support_flat=channel_flat,
                        support_sum_upper=_support_sum_upper(
                            support_flat
                        ),
                    )
                )
    deadline.check(force=True)
    frozen_offsets = tuple(offsets)
    manifest_body = dict(
        _plan_manifest_body(
            layer_id=int(layer.id),
            input_shape=input_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight=weight,
            support=support,
            offsets=frozen_offsets,
        )
    )
    manifest_body["content_sha256"] = frozen._json_digest(manifest_body)
    manifest = MappingProxyType(manifest_body)
    return DenseConvScalarGuardPlan(
        layer_id=int(layer.id),
        input_shape=input_shape,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weight=weight,
        support=support,
        offsets=frozen_offsets,
        manifest=manifest,
    )


def replay_dense_conv_scalar_guard(
    coefficient: Any,
    plan: DenseConvScalarGuardPlan,
    *,
    deadline: frozen._Deadline,
) -> DenseConvScalarGuardResult:
    """Run dense nominal Conv adjoints plus the two scalar guard stages."""

    deadline.check(force=True)
    if not isinstance(plan, DenseConvScalarGuardPlan):
        frozen._fail("V5_INVALID_PLAN", "invalid scalar-guard plan")
    _validate_plan_structure(plan)
    manifest_body = dict(
        _plan_manifest_body(
            layer_id=plan.layer_id,
            input_shape=plan.input_shape,
            output_shape=plan.output_shape,
            stride=plan.stride,
            padding=plan.padding,
            dilation=plan.dilation,
            groups=plan.groups,
            weight=plan.weight,
            support=plan.support,
            offsets=plan.offsets,
        )
    )
    expected_content = frozen._json_digest(manifest_body)
    expected_manifest = dict(manifest_body)
    expected_manifest["content_sha256"] = expected_content
    if (
        not hmac.compare_digest(
            str(plan.manifest.get("content_sha256", "")),
            expected_content,
        )
        or dict(plan.manifest) != expected_manifest
    ):
        frozen._fail("V5_INVALID_PLAN", "scalar-guard plan seal mismatch")
    coeff_matrix = frozen._as_f64_array(
        coefficient, name="V5 coefficient"
    )
    out_c, out_h, out_w = plan.output_shape
    in_c, in_h, in_w = plan.input_shape
    if (
        coeff_matrix.ndim != 2
        or coeff_matrix.shape[1] != out_c * out_h * out_w
        or coeff_matrix.shape[0] == 0
    ):
        frozen._fail("SHAPE_MISMATCH", "V5 Conv2D coefficient width")
    nonzero_count = int(np.count_nonzero(coeff_matrix))
    if nonzero_count * 8 <= int(coeff_matrix.size):
        frozen._fail(
            "V5_SPARSE_UNCHANGED",
            "sparse Conv queries must use the frozen scatter replay",
        )
    batch = coeff_matrix.shape[0]
    coeff = coeff_matrix.reshape(batch, out_c, out_h, out_w)
    nominal = np.zeros((batch, in_c, in_h * in_w), dtype=np.float64)
    channel_total = np.zeros(batch, dtype=np.float64)
    accumulation_total = np.zeros(batch, dtype=np.float64)
    out_per_group = out_c // plan.groups
    in_per_group = in_c // plan.groups
    dot_operations = 2 * out_per_group + 2
    dot_gamma, dot_underflow = _outward_roundoff_parameters(dot_operations)
    add_gamma, add_underflow = _outward_roundoff_parameters(2)

    for offset in plan.offsets:
        deadline.check(force=True)
        coeff_group = coeff[
            :,
            offset.co_start : offset.co_end,
            :,
            :,
        ]
        nominal_group = nominal[
            :,
            offset.ci_start : offset.ci_end,
            :,
        ]
        selected = np.take(
            coeff_group, offset.output_h_indices, axis=2
        )
        selected = np.take(
            selected, offset.output_w_indices, axis=3
        )
        left = np.ascontiguousarray(
            selected.transpose(0, 2, 3, 1).reshape(
                -1, out_per_group
            )
        )
        weight_slice = np.ascontiguousarray(
            plan.weight[
                offset.co_start : offset.co_end,
                :,
                offset.kh,
                offset.kw,
            ]
        )
        # This is deliberately the same nominal channel GEMM as the frozen
        # dense Conv replay and occurs once in the same offset order.
        term = np.asarray(left @ weight_slice, dtype=np.float64)
        frozen._require_finite(term, where="V5 nominal channel GEMM")
        nh = offset.output_h_indices.size
        nw = offset.output_w_indices.size
        term = term.reshape(
            batch, nh, nw, in_per_group
        ).transpose(0, 3, 1, 2)
        term = np.ascontiguousarray(
            term.reshape(batch, in_per_group, -1)
        )
        old = nominal_group[:, :, offset.targets]
        # Same nominal offset accumulation and ordering as the frozen replay.
        merged = np.asarray(old + term, dtype=np.float64)
        frozen._require_finite(
            merged, where="V5 nominal offset accumulation"
        )
        nominal_group[:, :, offset.targets] = merged

        channel_mass = _upper_nonnegative_row_dot(
            np.abs(selected)
            .transpose(0, 2, 3, 1)
            .reshape(batch, -1),
            offset.channel_support_flat,
        )
        channel_guard = _upper_scaled_mass(
            channel_mass,
            gamma_upper=dot_gamma,
            underflow_upper=dot_underflow,
            support_sum_upper=offset.support_sum_upper,
        )
        channel_total = frozen._upper_nonnegative_sum(
            channel_total, channel_guard
        )

        old_mass = _upper_nonnegative_row_dot(
            np.abs(old).reshape(batch, -1),
            offset.support_flat,
        )
        term_mass = _upper_nonnegative_row_dot(
            np.abs(term).reshape(batch, -1),
            offset.support_flat,
        )
        addition_mass = frozen._upper_nonnegative_sum(
            old_mass, term_mass
        )
        accumulation_guard = _upper_scaled_mass(
            addition_mass,
            gamma_upper=add_gamma,
            underflow_upper=add_underflow,
            support_sum_upper=offset.support_sum_upper,
        )
        accumulation_total = frozen._upper_nonnegative_sum(
            accumulation_total, accumulation_guard
        )
        deadline.check(force=True)

    scalar_guard = frozen._upper_nonnegative_sum(
        channel_total, accumulation_total
    )
    frozen._require_finite(nominal, where="V5 dense Conv nominal")
    frozen._require_finite(scalar_guard, where="V5 dense Conv scalar guard")
    deadline.check(force=True)
    return DenseConvScalarGuardResult(
        coefficient=frozen._immutable_f64_array(
            nominal.reshape(batch, -1), name="V5 nominal coefficient"
        ),
        scalar_guard=frozen._immutable_f64_array(
            scalar_guard, name="V5 scalar guard"
        ),
        channel_dot_guard=frozen._immutable_f64_array(
            channel_total, name="V5 channel-dot guard"
        ),
        accumulation_guard=frozen._immutable_f64_array(
            accumulation_total, name="V5 accumulation guard"
        ),
    )


def dense_conv_two_stage_scalar_guard(
    coefficient: Any,
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    *,
    deadline: frozen._Deadline,
) -> DenseConvScalarGuardResult:
    """Convenience entry point including outward-support preparation."""

    plan = prepare_dense_conv_scalar_guard(
        layer, predecessor_box, deadline=deadline
    )
    return replay_dense_conv_scalar_guard(
        coefficient, plan, deadline=deadline
    )


__all__ = [
    "DenseConvScalarGuardPlan",
    "DenseConvScalarGuardResult",
    "dense_conv_two_stage_scalar_guard",
    "prepare_dense_conv_scalar_guard",
    "replay_dense_conv_scalar_guard",
]
