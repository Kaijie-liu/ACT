#===- query_dual_replay_v51_conv.py - Structural Conv guard ----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
"""Isolated V5.1a dense-Conv scalar certificate.

This module is a controlled, non-authoritative successor to the rejected V5
dense-Conv prototype.  It deliberately leaves the V3 and V5 replay modules
unchanged.  The nominal transpose-convolution has exactly the frozen V3
kernel-offset order, channel GEMMs, and offset additions.

V5.1a changes only the support/roundoff certificate:

* every nonnegative support dot is enclosed in a platform-gated
  ``numpy.longdouble`` accumulator and conditionally rounded upward to f64;
* the support transform uses a true contraction-overlap structural mask;
* channel-dot ``D`` and offset-add ``A`` underflow terms are charged only to
  structurally active query rows;
* outward additions preserve exact structural zeros; and
* scalar absorption is row-local, so a zero-guard row is byte unchanged.

The dense/sparse decision remains exactly
``8 * count_nonzero(coefficient) <= coefficient.size``.  This module rejects
that sparse branch; the frozen V3 scatter implementation remains its only
implementation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen


SCHEMA = "act.query_dual_replay_v51_dense_conv_plan.v1"
RESULT_SCHEMA = "act.query_dual_replay_v51_dense_conv_result.v1"
NUMERIC_PROTOCOL = "wide_support_structural_activity_v51a"
_F64 = np.dtype(np.float64)
_LD = np.dtype(np.longdouble)
_F64_U = np.float64(2.0**-53)
_F64_ETA = np.nextafter(np.float64(0.0), np.float64(math.inf))
_REQUIRED_EXTRA_MANTISSA_BITS = 8


def _fail(code: str, message: str) -> None:
    frozen._fail(f"V51_{code}", message)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _bool_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0|b1\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _i64_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<i8\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_f64(value: Any, *, name: str) -> np.ndarray:
    return frozen._immutable_f64_array(value, name=name)


def _immutable_bool(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    result = np.frombuffer(
        array.tobytes(order="C"), dtype=np.bool_
    ).reshape(array.shape)
    result.setflags(write=False)
    return result


def _immutable_i64(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    result = np.frombuffer(
        array.tobytes(order="C"), dtype=np.int64
    ).reshape(array.shape)
    result.setflags(write=False)
    return result


def _wide_platform() -> Mapping[str, Any]:
    f64 = np.finfo(np.float64)
    wide = np.finfo(np.longdouble)
    if (
        np.dtype(np.float64).itemsize != 8
        or int(wide.nmant) < int(f64.nmant) + _REQUIRED_EXTRA_MANTISSA_BITS
        or not wide.eps < f64.eps
    ):
        _fail(
            "NUMERIC_PLATFORM",
            "longdouble must have at least eight more mantissa bits than f64",
        )
    eta_l = np.nextafter(
        np.longdouble(0.0), np.longdouble(math.inf)
    )
    if (
        eta_l <= np.longdouble(0.0)
        or np.longdouble(eta_l * np.longdouble(1.0)) != eta_l
    ):
        _fail("NUMERIC_PLATFORM", "longdouble gradual underflow failed")
    half_ulp = np.longdouble(2.0) ** np.longdouble(-(int(wide.nmant) + 1))
    above = np.nextafter(half_ulp, np.longdouble(math.inf))
    if (
        np.longdouble(1.0) + half_ulp != np.longdouble(1.0)
        or np.longdouble(1.0) + above == np.longdouble(1.0)
    ):
        _fail("NUMERIC_PLATFORM", "longdouble RN-even probe failed")
    f64_eta = np.nextafter(np.float64(0.0), np.float64(math.inf))
    if (
        np.float64(f64_eta * np.float64(1.0)) != f64_eta
        or float(
            np.asarray([f64_eta], dtype=np.float64)
            @ np.asarray([1.0], dtype=np.float64)
        )
        != float(f64_eta)
    ):
        _fail("NUMERIC_PLATFORM", "binary64 gradual underflow failed")
    return MappingProxyType(
        {
            "schema": "act.query_dual_replay_v51_wide_platform.v1",
            "f64_nmant": int(f64.nmant),
            "longdouble_nmant": int(wide.nmant),
            "required_extra_mantissa_bits": _REQUIRED_EXTRA_MANTISSA_BITS,
            "longdouble_eps": str(wide.eps),
            "longdouble_tiny": str(wide.tiny),
            "longdouble_eta": str(eta_l),
            "rounding": "round-to-nearest-even",
            "gradual_underflow": True,
        }
    )


def _ld_next_up(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.longdouble)
    result = np.nextafter(array, np.longdouble(math.inf))
    if not np.all(np.isfinite(result)):
        _fail("NONFINITE", "longdouble outward successor overflowed")
    return result


def _ld_next_down(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.longdouble)
    result = np.nextafter(array, np.longdouble(-math.inf))
    if not np.all(np.isfinite(result)):
        _fail("NONFINITE", "longdouble outward predecessor overflowed")
    return result


def _ld_up_add(left: Any, right: Any) -> np.ndarray:
    return _ld_next_up(
        np.asarray(left, dtype=np.longdouble)
        + np.asarray(right, dtype=np.longdouble)
    )


def _ld_up_mul(left: Any, right: Any) -> np.ndarray:
    return _ld_next_up(
        np.asarray(left, dtype=np.longdouble)
        * np.asarray(right, dtype=np.longdouble)
    )


def _ld_up_div(numerator: Any, denominator_lower: Any) -> np.ndarray:
    denominator = np.asarray(denominator_lower, dtype=np.longdouble)
    if np.any(denominator <= np.longdouble(0.0)):
        _fail("NUMERIC_GUARD", "nonpositive outward denominator")
    return _ld_next_up(
        np.asarray(numerator, dtype=np.longdouble) / denominator
    )


def _ceil_f64(value: Any, *, where: str) -> np.ndarray:
    """Smallest available f64 not below a nonnegative longdouble value."""

    extended = np.asarray(value, dtype=np.longdouble)
    scalar = extended.ndim == 0
    if (
        np.any(extended < np.longdouble(0.0))
        or not np.all(np.isfinite(extended))
    ):
        _fail("NONFINITE", f"invalid nonnegative wide value at {where}")
    narrowed = np.asarray(extended, dtype=np.float64)
    if not np.all(np.isfinite(narrowed)):
        _fail("NONFINITE", f"f64 conversion overflowed at {where}")
    below = np.asarray(narrowed, dtype=np.longdouble) < extended
    if np.any(below):
        if scalar:
            narrowed = np.asarray(
                np.nextafter(
                    np.float64(narrowed.item()),
                    np.float64(math.inf),
                ),
                dtype=np.float64,
            )
        else:
            narrowed = np.ascontiguousarray(narrowed)
            narrowed[below] = np.nextafter(
                narrowed[below], np.float64(math.inf)
            )
    if not np.all(np.isfinite(narrowed)):
        _fail("NONFINITE", f"f64 outward successor overflowed at {where}")
    if scalar:
        return np.asarray(narrowed, dtype=np.float64).reshape(())
    return np.ascontiguousarray(narrowed, dtype=np.float64)


def _wide_roundoff_parameters(
    width: int,
) -> Tuple[np.longdouble, np.longdouble]:
    """Directed ``gammaL`` and ``tauL`` for ``k=2*width+2``."""

    count = 2 * int(width) + 2
    if width <= 0:
        _fail("SHAPE_MISMATCH", "wide dot width must be positive")
    info = np.finfo(np.longdouble)
    unit = np.longdouble(info.eps) / np.longdouble(2.0)
    eta = np.nextafter(
        np.longdouble(0.0), np.longdouble(math.inf)
    )
    product = _ld_up_mul(np.longdouble(count), unit)
    if product >= np.longdouble(0.5):
        _fail("NUMERIC_GUARD", "wide dot operation count is too large")
    denominator = _ld_next_down(np.longdouble(1.0) - product)
    gamma = _ld_up_div(product, denominator)
    tau = _ld_up_div(
        _ld_up_mul(np.longdouble(count), eta), denominator
    )
    return np.longdouble(gamma), np.longdouble(tau)


def _f64_parameters_for_operations(count: int) -> Tuple[float, float]:
    if count <= 0:
        _fail("NUMERIC_GUARD", "binary64 operation count must be positive")
    product = _ld_up_mul(
        np.longdouble(count), np.longdouble(_F64_U)
    )
    if product >= np.longdouble(0.5):
        _fail("NUMERIC_GUARD", "binary64 operation count is too large")
    denominator = _ld_next_down(np.longdouble(1.0) - product)
    gamma = _ceil_f64(
        _ld_up_div(product, denominator), where="binary64 gamma"
    )
    tau = _ceil_f64(
        _ld_up_div(
            _ld_up_mul(
                np.longdouble(count), np.longdouble(_F64_ETA)
            ),
            denominator,
        ),
        where="binary64 tau",
    )
    return float(gamma), float(tau)


def _f64_roundoff_parameters(width: int) -> Tuple[float, float]:
    if width <= 0:
        _fail("SHAPE_MISMATCH", "binary64 dot width must be positive")
    return _f64_parameters_for_operations(2 * int(width) + 2)


def _dot_up_l_rows_unchecked(
    left: Any,
    right: Any,
    *,
    deadline: frozen._Deadline | None = None,
) -> np.ndarray:
    """Internal ``DotUpL`` after the public/session platform gate."""

    if deadline is not None:
        deadline.check(force=True)
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        left_array.dtype != _F64
        or right_array.dtype != _F64
        or left_array.ndim != 2
        or right_array.ndim != 1
        or left_array.shape[1] != right_array.size
        or left_array.shape[0] <= 0
        or right_array.size <= 0
        or np.any(left_array < 0.0)
        or np.any(right_array < 0.0)
        or not np.all(np.isfinite(left_array))
        or not np.all(np.isfinite(right_array))
    ):
        _fail("INVALID_MASS", "invalid nonnegative DotUpL operands")
    exact_zero = ~np.any(
        (left_array != 0.0)
        & (right_array.reshape(1, -1) != 0.0),
        axis=1,
    )
    nominal = np.asarray(
        np.asarray(left_array, dtype=np.longdouble)
        @ np.asarray(right_array, dtype=np.longdouble),
        dtype=np.longdouble,
    )
    gamma, tau = _wide_roundoff_parameters(right_array.size)
    numerator = _ld_up_add(nominal, tau)
    denominator = _ld_next_down(np.longdouble(1.0) - gamma)
    upper = _ceil_f64(
        _ld_up_div(numerator, denominator), where="DotUpL rows"
    )
    upper[exact_zero] = 0.0
    if deadline is not None:
        deadline.check(force=True)
    return np.ascontiguousarray(upper, dtype=np.float64)


def dot_up_l_rows(
    left: Any,
    right: Any,
    *,
    deadline: frozen._Deadline | None = None,
) -> np.ndarray:
    """Return ``DotUpL`` for every row of a nonnegative matrix."""

    _wide_platform()
    return _dot_up_l_rows_unchecked(left, right, deadline=deadline)


def _dot_up_l_matrix(
    left: np.ndarray,
    right: np.ndarray,
    *,
    deadline: frozen._Deadline,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wide nonnegative matrix product and exact contraction activity."""

    if (
        left.dtype != _F64
        or right.dtype != _F64
        or left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
        or np.any(left < 0.0)
        or np.any(right < 0.0)
    ):
        _fail("INVALID_MASS", "invalid wide matrix-product operands")
    deadline.check(force=True)
    activity = np.asarray(
        (left != 0.0) @ (right != 0.0), dtype=np.bool_
    )
    nominal = np.asarray(
        np.asarray(left, dtype=np.longdouble)
        @ np.asarray(right, dtype=np.longdouble),
        dtype=np.longdouble,
    )
    gamma, tau = _wide_roundoff_parameters(left.shape[1])
    numerator = _ld_up_add(nominal, tau)
    denominator = _ld_next_down(np.longdouble(1.0) - gamma)
    result = _ceil_f64(
        _ld_up_div(numerator, denominator), where="DotUpL matrix"
    )
    result[~activity] = 0.0
    deadline.check(force=True)
    return (
        np.ascontiguousarray(result),
        np.ascontiguousarray(activity),
    )


def _zero_preserving_upper_sum(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Outward nonnegative addition that leaves ``0+0`` exactly zero."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.shape != right_array.shape
        or np.any(left_array < 0.0)
        or np.any(right_array < 0.0)
        or not np.all(np.isfinite(left_array))
        or not np.all(np.isfinite(right_array))
    ):
        _fail("INVALID_GUARD", "invalid zero-preserving sum operands")
    left_active = left_array != 0.0
    right_active = right_array != 0.0
    both = left_active & right_active
    result = np.zeros(left_array.shape, dtype=np.float64)
    # Copying x through x+0 is exact.  Apart from retaining structural zeros,
    # this avoids manufacturing a successor merely because the directed-wide
    # helper was asked to add an exact zero.
    result[left_active & ~right_active] = left_array[
        left_active & ~right_active
    ]
    result[right_active & ~left_active] = right_array[
        right_active & ~left_active
    ]
    if np.any(both):
        wide = _ld_up_add(left_array[both], right_array[both])
        result[both] = _ceil_f64(
            wide, where="zero-preserving outward sum"
        )
    return result


def _scaled_guard(
    mass_upper: np.ndarray,
    *,
    gamma_upper: float,
    tau_upper: float,
    support_sum_upper: float,
    active: np.ndarray,
) -> np.ndarray:
    mass = np.asarray(mass_upper, dtype=np.float64)
    mask = np.asarray(active, dtype=np.bool_)
    if (
        mass.ndim != 1
        or mask.shape != mass.shape
        or np.any(mass < 0.0)
        or gamma_upper < 0.0
        or tau_upper < 0.0
        or support_sum_upper < 0.0
    ):
        _fail("INVALID_GUARD", "invalid scaled-guard operands")
    result = np.zeros(mass.shape, dtype=np.float64)
    if np.any(mask):
        first = _ld_up_mul(
            np.longdouble(gamma_upper),
            np.asarray(mass[mask], dtype=np.longdouble),
        )
        second = _ld_up_mul(
            np.longdouble(tau_upper),
            np.longdouble(support_sum_upper),
        )
        result[mask] = _ceil_f64(
            _ld_up_add(first, second), where="scaled D/A guard"
        )
    return result


@dataclass(frozen=True)
class _OffsetSupport:
    group: int
    kh: int
    kw: int
    co_start: int
    co_end: int
    ci_start: int
    ci_end: int
    output_h_indices: np.ndarray = field(repr=False, compare=False)
    output_w_indices: np.ndarray = field(repr=False, compare=False)
    targets: np.ndarray = field(repr=False, compare=False)
    support_flat: np.ndarray = field(repr=False, compare=False)
    channel_support_flat: np.ndarray = field(repr=False, compare=False)
    support_activity_flat: np.ndarray = field(repr=False, compare=False)
    support_sum_upper: float


@dataclass(frozen=True)
class DenseConvV51Plan:
    layer_id: int
    input_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    weight: np.ndarray = field(repr=False, compare=False)
    support: np.ndarray = field(repr=False, compare=False)
    offsets: Tuple[_OffsetSupport, ...] = field(repr=False, compare=False)
    manifest: Mapping[str, Any]
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("V5.1 Conv research plan cannot issue authority")


@dataclass(frozen=True)
class DenseConvV51Result:
    coefficient: np.ndarray = field(repr=False, compare=False)
    scalar_guard: np.ndarray = field(repr=False, compare=False)
    channel_dot_guard: np.ndarray = field(repr=False, compare=False)
    accumulation_guard: np.ndarray = field(repr=False, compare=False)
    active_mask: np.ndarray = field(repr=False, compare=False)
    channel_dot_active_mask: np.ndarray = field(repr=False, compare=False)
    accumulation_active_mask: np.ndarray = field(
        repr=False, compare=False
    )
    receipt: Mapping[str, Any]
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("V5.1 Conv research result cannot issue authority")
        batch = self.scalar_guard.size
        arrays = (
            self.coefficient,
            self.scalar_guard,
            self.channel_dot_guard,
            self.accumulation_guard,
        )
        if (
            self.coefficient.dtype != _F64
            or self.coefficient.ndim != 2
            or self.coefficient.shape[0] != batch
            or any(
                value.dtype != _F64
                or value.ndim != 1
                or value.size != batch
                or value.flags.writeable
                or not np.all(np.isfinite(value))
                for value in arrays[1:]
            )
            or self.coefficient.flags.writeable
            or not np.all(np.isfinite(self.coefficient))
            or np.any(self.scalar_guard < 0.0)
            or np.any(self.channel_dot_guard < 0.0)
            or np.any(self.accumulation_guard < 0.0)
        ):
            raise ValueError("malformed immutable V5.1 Conv result")
        masks = (
            self.active_mask,
            self.channel_dot_active_mask,
            self.accumulation_active_mask,
        )
        if any(
            mask.dtype != np.bool_
            or mask.shape != (batch,)
            or mask.flags.writeable
            for mask in masks
        ):
            raise ValueError("malformed immutable V5.1 activity mask")
        if not np.array_equal(
            self.active_mask,
            self.channel_dot_active_mask
            | self.accumulation_active_mask,
        ):
            raise ValueError("V5.1 final activity is not D OR A")
        if np.any(self.scalar_guard[~self.active_mask] != 0.0):
            raise ValueError("inactive V5.1 rows must have an exact zero guard")
        if np.any(
            self.channel_dot_guard[~self.channel_dot_active_mask] != 0.0
        ) or np.any(
            self.accumulation_guard[
                ~self.accumulation_active_mask
            ]
            != 0.0
        ):
            raise ValueError("inactive V5.1 D/A components must be exact zero")


def _offset_manifest(offset: _OffsetSupport) -> Mapping[str, Any]:
    return {
        "group": int(offset.group),
        "kh": int(offset.kh),
        "kw": int(offset.kw),
        "co": [int(offset.co_start), int(offset.co_end)],
        "ci": [int(offset.ci_start), int(offset.ci_end)],
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
        "support_activity_sha256": _bool_digest(
            offset.support_activity_flat
        ),
        "support_activity_count": int(
            np.count_nonzero(offset.support_activity_flat)
        ),
        "support_sum_upper_hex": float(
            offset.support_sum_upper
        ).hex(),
    }


def _manifest_body(
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
    out_per_group = output_shape[0] // groups
    dot_gamma, dot_tau = _f64_roundoff_parameters(out_per_group)
    add_gamma, add_tau = _f64_parameters_for_operations(2)
    platform = dict(_wide_platform())
    return {
        "schema": SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "layer_id": int(layer_id),
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "stride": list(stride),
        "padding": list(padding),
        "dilation": list(dilation),
        "groups": int(groups),
        "weight_sha256": frozen._array_digest(weight),
        "support_sha256": frozen._array_digest(support),
        "offset_count": len(offsets),
        "offsets": [_offset_manifest(value) for value in offsets],
        "dot_operations": 2 * out_per_group + 2,
        "dot_gamma_upper_hex": dot_gamma.hex(),
        "dot_tau_upper_hex": dot_tau.hex(),
        "addition_operations": 2,
        "addition_gamma_upper_hex": add_gamma.hex(),
        "addition_tau_upper_hex": add_tau.hex(),
        "wide_dot": "DotUpL_conditional_ceil_f64",
        "support_activity": "exact_contraction_overlap_E_S",
        "channel_activity": "E_D",
        "addition_activity": "E_A_old_AND_term",
        "outward_sum": "zero_preserving",
        "sparse_branch": "frozen_v3_only",
        "platform": platform,
        "platform_sha256": _canonical_digest(platform),
    }


def _geometry(layer: frozen._FrozenLayer) -> Mapping[str, Any]:
    if not isinstance(layer, frozen._FrozenLayer) or layer.kind != "CONV2D":
        _fail("INVALID_LAYER", "V5.1 Conv requires a frozen CONV2D layer")
    p = layer.params
    result = {
        "input_shape": tuple(int(value) for value in p["input_shape"]),
        "output_shape": tuple(int(value) for value in p["output_shape"]),
        "stride": tuple(int(value) for value in p["stride"]),
        "padding": tuple(int(value) for value in p["padding"]),
        "dilation": tuple(int(value) for value in p["dilation"]),
        "groups": int(p["groups"]),
    }
    weight = np.asarray(p["weight"])
    if (
        weight.dtype != _F64
        or weight.ndim != 4
        or any(value <= 0 for value in result["input_shape"])
        or any(value <= 0 for value in result["output_shape"])
        or any(value <= 0 for value in result["stride"])
        or any(value < 0 for value in result["padding"])
        or any(value <= 0 for value in result["dilation"])
        or result["groups"] <= 0
        or result["output_shape"][0] % result["groups"]
        or result["input_shape"][0]
        != weight.shape[1] * result["groups"]
        or weight.shape[0] != result["output_shape"][0]
        or not np.all(np.isfinite(weight))
    ):
        _fail("INVALID_LAYER", "invalid V5.1 Conv geometry")
    frozen._conv_output_padding(layer)
    return result


def prepare_dense_conv_v51_plan(
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    *,
    deadline: frozen._Deadline,
) -> DenseConvV51Plan:
    """Build an immutable wide-support V5.1 Conv plan."""

    deadline.check(force=True)
    _wide_platform()
    geometry = _geometry(layer)
    weight = _immutable_f64(layer.params["weight"], name="V5.1 weight")
    input_shape = geometry["input_shape"]
    output_shape = geometry["output_shape"]
    stride = geometry["stride"]
    padding = geometry["padding"]
    dilation = geometry["dilation"]
    groups = geometry["groups"]
    lb = frozen._as_f64_array(
        predecessor_box.lb, name="V5.1 support lower"
    ).reshape(-1)
    ub = frozen._as_f64_array(
        predecessor_box.ub, name="V5.1 support upper"
    ).reshape(-1)
    if (
        lb.shape != ub.shape
        or lb.size != int(np.prod(input_shape))
        or np.any(lb > ub)
    ):
        _fail("INVALID_SUPPORT", "invalid predecessor support box")
    support = _immutable_f64(
        np.maximum(np.abs(lb), np.abs(ub)), name="V5.1 maxabs support"
    )

    out_c, out_h, out_w = output_shape
    in_c, in_h, in_w = input_shape
    out_per_group = out_c // groups
    in_per_group = in_c // groups
    offsets = []
    for group in range(groups):
        co_start = group * out_per_group
        co_end = co_start + out_per_group
        ci_start = group * in_per_group
        ci_end = ci_start + in_per_group
        for kh in range(weight.shape[2]):
            deadline.check(force=True)
            input_h_indices = (
                np.arange(out_h, dtype=np.int64) * stride[0]
                - padding[0]
                + kh * dilation[0]
            )
            valid_h = (input_h_indices >= 0) & (input_h_indices < in_h)
            if not np.any(valid_h):
                continue
            output_h_indices = np.flatnonzero(valid_h)
            input_h_indices = input_h_indices[valid_h]
            for kw in range(weight.shape[3]):
                input_w_indices = (
                    np.arange(out_w, dtype=np.int64) * stride[1]
                    - padding[1]
                    + kw * dilation[1]
                )
                valid_w = (
                    (input_w_indices >= 0) & (input_w_indices < in_w)
                )
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
                    np.abs(
                        weight[
                            co_start:co_end,
                            :,
                            kh,
                            kw,
                        ]
                    )
                )
                channel_upper, support_activity = _dot_up_l_matrix(
                    weight_abs, support_selected, deadline=deadline
                )
                support_flat = _immutable_f64(
                    support_selected.reshape(-1),
                    name="V5.1 offset support",
                )
                channel_flat = _immutable_f64(
                    channel_upper.T.reshape(-1),
                    name="V5.1 channel support",
                )
                activity_flat = _immutable_bool(
                    support_activity.T.reshape(-1)
                )
                support_sum = float(
                    _dot_up_l_rows_unchecked(
                        support_flat.reshape(1, -1),
                        np.ones(support_flat.size, dtype=np.float64),
                        deadline=deadline,
                    )[0]
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
                        support_activity_flat=activity_flat,
                        support_sum_upper=support_sum,
                    )
                )
    frozen_offsets = tuple(offsets)
    body = dict(
        _manifest_body(
            layer_id=layer.id,
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
    body["content_sha256"] = _canonical_digest(body)
    deadline.check(force=True)
    return DenseConvV51Plan(
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
        manifest=MappingProxyType(body),
    )


def _expected_offset_geometry(
    plan: DenseConvV51Plan,
) -> Tuple[Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray], ...]:
    _, out_h, out_w = plan.output_shape
    _, in_h, in_w = plan.input_shape
    expected = []
    for group in range(plan.groups):
        for kh in range(plan.weight.shape[2]):
            input_h = (
                np.arange(out_h, dtype=np.int64) * plan.stride[0]
                - plan.padding[0]
                + kh * plan.dilation[0]
            )
            valid_h = (input_h >= 0) & (input_h < in_h)
            if not np.any(valid_h):
                continue
            output_h = np.flatnonzero(valid_h)
            input_h = input_h[valid_h]
            for kw in range(plan.weight.shape[3]):
                input_w = (
                    np.arange(out_w, dtype=np.int64) * plan.stride[1]
                    - plan.padding[1]
                    + kw * plan.dilation[1]
                )
                valid_w = (input_w >= 0) & (input_w < in_w)
                if not np.any(valid_w):
                    continue
                output_w = np.flatnonzero(valid_w)
                input_w = input_w[valid_w]
                targets = (
                    input_h[:, None] * in_w + input_w[None, :]
                ).reshape(-1)
                expected.append(
                    (group, kh, kw, output_h, output_w, targets)
                )
    return tuple(expected)


def _validate_plan(
    plan: DenseConvV51Plan,
    *,
    deadline: frozen._Deadline,
) -> None:
    deadline.check(force=True)
    if not isinstance(plan, DenseConvV51Plan) or plan.proof_authority:
        _fail("INVALID_PLAN", "invalid V5.1 Conv plan type")
    out_c, _, _ = plan.output_shape
    in_c, in_h, in_w = plan.input_shape
    if (
        plan.weight.dtype != _F64
        or plan.weight.ndim != 4
        or plan.weight.flags.writeable
        or plan.support.dtype != _F64
        or plan.support.shape != (in_c * in_h * in_w,)
        or plan.support.flags.writeable
        or np.any(plan.support < 0.0)
        or not np.all(np.isfinite(plan.support))
        or plan.groups <= 0
        or out_c % plan.groups
        or in_c % plan.groups
        or plan.weight.shape
        != (
            out_c,
            in_c // plan.groups,
            plan.weight.shape[2],
            plan.weight.shape[3],
        )
    ):
        _fail("INVALID_PLAN", "malformed V5.1 plan arrays")
    expected_geometry = _expected_offset_geometry(plan)
    if len(expected_geometry) != len(plan.offsets):
        _fail("INVALID_PLAN", "V5.1 offset coverage changed")
    out_per_group = out_c // plan.groups
    in_per_group = in_c // plan.groups
    support_view = plan.support.reshape(in_c, -1)
    for expected, offset in zip(expected_geometry, plan.offsets):
        group, kh, kw, output_h, output_w, targets = expected
        co_start = group * out_per_group
        ci_start = group * in_per_group
        positions = targets.size
        if (
            (offset.group, offset.kh, offset.kw) != (group, kh, kw)
            or (offset.co_start, offset.co_end)
            != (co_start, co_start + out_per_group)
            or (offset.ci_start, offset.ci_end)
            != (ci_start, ci_start + in_per_group)
            or not np.array_equal(offset.output_h_indices, output_h)
            or not np.array_equal(offset.output_w_indices, output_w)
            or not np.array_equal(offset.targets, targets)
            or offset.support_flat.shape
            != (in_per_group * positions,)
            or offset.channel_support_flat.shape
            != (out_per_group * positions,)
            or offset.support_activity_flat.shape
            != (out_per_group * positions,)
            or offset.support_flat.flags.writeable
            or offset.channel_support_flat.flags.writeable
            or offset.support_activity_flat.flags.writeable
        ):
            _fail("INVALID_PLAN", "malformed V5.1 offset record")
        selected = np.ascontiguousarray(
            support_view[ci_start : ci_start + in_per_group, :][
                :, targets
            ]
        )
        if not np.array_equal(
            offset.support_flat, selected.reshape(-1)
        ):
            _fail("INVALID_PLAN", "offset support substitution")
        weight_abs = np.ascontiguousarray(
            np.abs(
                plan.weight[
                    co_start : co_start + out_per_group,
                    :,
                    kh,
                    kw,
                ]
            )
        )
        expected_support, expected_activity = _dot_up_l_matrix(
            weight_abs, selected, deadline=deadline
        )
        if (
            not np.array_equal(
                offset.channel_support_flat,
                expected_support.T.reshape(-1),
            )
            or not np.array_equal(
                offset.support_activity_flat,
                expected_activity.T.reshape(-1),
            )
        ):
            _fail("INVALID_PLAN", "support/activity substitution")
        expected_sum = float(
            _dot_up_l_rows_unchecked(
                selected.reshape(1, -1),
                np.ones(selected.size, dtype=np.float64),
                deadline=deadline,
            )[0]
        )
        if expected_sum != offset.support_sum_upper:
            _fail("INVALID_PLAN", "support-sum substitution")
    body = dict(
        _manifest_body(
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
    content = _canonical_digest(body)
    expected_manifest = dict(body)
    expected_manifest["content_sha256"] = content
    if (
        not hmac.compare_digest(
            str(plan.manifest.get("content_sha256", "")), content
        )
        or dict(plan.manifest) != expected_manifest
    ):
        _fail("INVALID_PLAN", "V5.1 plan manifest substitution")
    deadline.check(force=True)


def replay_dense_conv_v51(
    coefficient: Any,
    plan: DenseConvV51Plan,
    *,
    deadline: frozen._Deadline,
) -> DenseConvV51Result:
    """Replay one dense Conv block with V5.1 structural scalar guards."""

    _wide_platform()
    _validate_plan(plan, deadline=deadline)
    coeff_matrix = frozen._as_f64_array(
        coefficient, name="V5.1 Conv coefficient"
    )
    out_c, out_h, out_w = plan.output_shape
    in_c, in_h, in_w = plan.input_shape
    if (
        coeff_matrix.ndim != 2
        or coeff_matrix.shape[0] <= 0
        or coeff_matrix.shape[1] != out_c * out_h * out_w
    ):
        _fail("SHAPE_MISMATCH", "V5.1 Conv coefficient width")
    nonzero_count = int(np.count_nonzero(coeff_matrix))
    dense_count = int(coeff_matrix.size)
    if nonzero_count * 8 <= dense_count:
        _fail(
            "SPARSE_UNCHANGED",
            "sparse Conv rows must use the frozen V3 scatter replay",
        )

    batch = coeff_matrix.shape[0]
    coeff = coeff_matrix.reshape(batch, out_c, out_h, out_w)
    nominal = np.zeros((batch, in_c, in_h * in_w), dtype=np.float64)
    channel_total = np.zeros(batch, dtype=np.float64)
    accumulation_total = np.zeros(batch, dtype=np.float64)
    channel_active_total = np.zeros(batch, dtype=np.bool_)
    accumulation_active_total = np.zeros(batch, dtype=np.bool_)
    out_per_group = out_c // plan.groups
    in_per_group = in_c // plan.groups
    dot_gamma, dot_tau = _f64_roundoff_parameters(out_per_group)
    add_gamma, add_tau = _f64_parameters_for_operations(2)

    for offset in plan.offsets:
        deadline.check(force=True)
        coeff_group = coeff[
            :, offset.co_start : offset.co_end, :, :
        ]
        nominal_group = nominal[
            :, offset.ci_start : offset.ci_end, :
        ]
        selected = np.take(
            coeff_group, offset.output_h_indices, axis=2
        )
        selected = np.take(
            selected, offset.output_w_indices, axis=3
        )
        selected_flat = np.ascontiguousarray(
            selected.transpose(0, 2, 3, 1).reshape(batch, -1)
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
        # This nominal sequence must stay bit-identical to frozen V3.
        term = np.asarray(left @ weight_slice, dtype=np.float64)
        frozen._require_finite(term, where="V5.1 channel GEMM")
        nh = offset.output_h_indices.size
        nw = offset.output_w_indices.size
        term = term.reshape(
            batch, nh, nw, in_per_group
        ).transpose(0, 3, 1, 2)
        term = np.ascontiguousarray(
            term.reshape(batch, in_per_group, -1)
        )
        old = np.ascontiguousarray(
            nominal_group[:, :, offset.targets]
        )
        merged = np.asarray(old + term, dtype=np.float64)
        frozen._require_finite(merged, where="V5.1 offset addition")
        nominal_group[:, :, offset.targets] = merged

        channel_active = np.any(
            (selected_flat != 0.0)
            & offset.support_activity_flat.reshape(1, -1),
            axis=1,
        )
        channel_mass = _dot_up_l_rows_unchecked(
            np.abs(selected_flat),
            offset.channel_support_flat,
            deadline=deadline,
        )
        channel_guard = _scaled_guard(
            channel_mass,
            gamma_upper=dot_gamma,
            tau_upper=dot_tau,
            support_sum_upper=offset.support_sum_upper,
            active=channel_active,
        )
        channel_total = _zero_preserving_upper_sum(
            channel_total, channel_guard
        )
        channel_active_total |= channel_active

        support_nonzero = (
            offset.support_flat.reshape(1, -1) != 0.0
        )
        addition_active = np.any(
            support_nonzero
            & (old.reshape(batch, -1) != 0.0)
            & (term.reshape(batch, -1) != 0.0),
            axis=1,
        )
        old_mass = _dot_up_l_rows_unchecked(
            np.abs(old).reshape(batch, -1),
            offset.support_flat,
            deadline=deadline,
        )
        term_mass = _dot_up_l_rows_unchecked(
            np.abs(term).reshape(batch, -1),
            offset.support_flat,
            deadline=deadline,
        )
        addition_mass = _zero_preserving_upper_sum(
            old_mass, term_mass
        )
        accumulation_guard = _scaled_guard(
            addition_mass,
            gamma_upper=add_gamma,
            tau_upper=add_tau,
            support_sum_upper=offset.support_sum_upper,
            active=addition_active,
        )
        accumulation_total = _zero_preserving_upper_sum(
            accumulation_total, accumulation_guard
        )
        accumulation_active_total |= addition_active
        deadline.check(force=True)

    active = channel_active_total | accumulation_active_total
    scalar_guard = _zero_preserving_upper_sum(
        channel_total, accumulation_total
    )
    scalar_guard[~active] = 0.0
    coefficient_out = _immutable_f64(
        nominal.reshape(batch, -1), name="V5.1 nominal coefficient"
    )
    scalar_out = _immutable_f64(
        scalar_guard, name="V5.1 scalar guard"
    )
    channel_out = _immutable_f64(
        channel_total, name="V5.1 channel guard"
    )
    accumulation_out = _immutable_f64(
        accumulation_total, name="V5.1 accumulation guard"
    )
    active_out = _immutable_bool(active)
    channel_active_out = _immutable_bool(channel_active_total)
    accumulation_active_out = _immutable_bool(
        accumulation_active_total
    )
    body: Dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "plan_content_sha256": plan.manifest["content_sha256"],
        "coefficient_input_sha256": frozen._array_digest(coeff_matrix),
        "coefficient_output_sha256": frozen._array_digest(coefficient_out),
        "scalar_guard_sha256": frozen._array_digest(scalar_out),
        "channel_dot_guard_sha256": frozen._array_digest(channel_out),
        "accumulation_guard_sha256": frozen._array_digest(
            accumulation_out
        ),
        "active_mask_sha256": _bool_digest(active_out),
        "active_count": int(np.count_nonzero(active_out)),
        "channel_dot_active_mask_sha256": _bool_digest(
            channel_active_out
        ),
        "channel_dot_active_count": int(
            np.count_nonzero(channel_active_out)
        ),
        "accumulation_active_mask_sha256": _bool_digest(
            accumulation_active_out
        ),
        "accumulation_active_count": int(
            np.count_nonzero(accumulation_active_out)
        ),
        "nonzero_count": nonzero_count,
        "dense_count": dense_count,
        "threshold_lhs": 8 * nonzero_count,
        "threshold_rhs": dense_count,
        "branch": "dense",
        "nominal_policy": "frozen_v3_bit_sequence",
        "guard_policy": "one_row_local_scalar_D_plus_A",
    }
    body["content_sha256"] = _canonical_digest(body)
    deadline.check(force=True)
    return DenseConvV51Result(
        coefficient=coefficient_out,
        scalar_guard=scalar_out,
        channel_dot_guard=channel_out,
        accumulation_guard=accumulation_out,
        active_mask=active_out,
        channel_dot_active_mask=channel_active_out,
        accumulation_active_mask=accumulation_active_out,
        receipt=MappingProxyType(body),
    )


def dense_conv_v51(
    coefficient: Any,
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    *,
    deadline: frozen._Deadline,
) -> DenseConvV51Result:
    plan = prepare_dense_conv_v51_plan(
        layer, predecessor_box, deadline=deadline
    )
    return replay_dense_conv_v51(
        coefficient, plan, deadline=deadline
    )


def absorb_scalar_guard_row_local(
    scalar: Any,
    result: DenseConvV51Result,
) -> np.ndarray:
    """Apply exactly one scalar guard only to structurally active rows."""

    values = frozen._as_f64_array(
        scalar, name="V5.1 row-local scalar"
    ).reshape(-1)
    if (
        not isinstance(result, DenseConvV51Result)
        or values.shape != result.scalar_guard.shape
    ):
        _fail("SHAPE_MISMATCH", "row-local guard/scalar mismatch")
    output = np.ascontiguousarray(values.copy(), dtype=np.float64)
    active = result.active_mask
    if np.any(active):
        output[active] = frozen._down_add(
            output[active],
            -result.scalar_guard[active],
            where="V5.1 row-local scalar guard",
        )
    frozen._require_finite(output, where="V5.1 row-local absorption")
    return output


def verify_dense_conv_v51_result(result: DenseConvV51Result) -> bool:
    """Verify research-result integrity while explicitly denying authority."""

    try:
        if (
            not isinstance(result, DenseConvV51Result)
            or result.proof_authority
        ):
            return False
        body = dict(result.receipt)
        claimed = str(body.pop("content_sha256"))
        nonzero_count = int(body.get("nonzero_count"))
        dense_count = int(body.get("dense_count"))
        return bool(
            body.get("schema") == RESULT_SCHEMA
            and body.get("numeric_protocol") == NUMERIC_PROTOCOL
            and body.get("proof_authority") is False
            and hmac.compare_digest(_canonical_digest(body), claimed)
            and body.get("coefficient_output_sha256")
            == frozen._array_digest(result.coefficient)
            and body.get("scalar_guard_sha256")
            == frozen._array_digest(result.scalar_guard)
            and body.get("channel_dot_guard_sha256")
            == frozen._array_digest(result.channel_dot_guard)
            and body.get("accumulation_guard_sha256")
            == frozen._array_digest(result.accumulation_guard)
            and body.get("active_mask_sha256")
            == _bool_digest(result.active_mask)
            and body.get("channel_dot_active_mask_sha256")
            == _bool_digest(result.channel_dot_active_mask)
            and body.get("accumulation_active_mask_sha256")
            == _bool_digest(result.accumulation_active_mask)
            and int(body.get("active_count"))
            == int(np.count_nonzero(result.active_mask))
            and int(body.get("channel_dot_active_count"))
            == int(np.count_nonzero(result.channel_dot_active_mask))
            and int(body.get("accumulation_active_count"))
            == int(np.count_nonzero(result.accumulation_active_mask))
            and body.get("branch") == "dense"
            and body.get("nominal_policy") == "frozen_v3_bit_sequence"
            and body.get("guard_policy")
            == "one_row_local_scalar_D_plus_A"
            and nonzero_count >= 0
            and dense_count > 0
            and int(body.get("threshold_lhs")) == 8 * nonzero_count
            and int(body.get("threshold_rhs")) == dense_count
            and 8 * nonzero_count > dense_count
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


__all__ = [
    "DenseConvV51Plan",
    "DenseConvV51Result",
    "NUMERIC_PROTOCOL",
    "SCHEMA",
    "absorb_scalar_guard_row_local",
    "dense_conv_v51",
    "dot_up_l_rows",
    "prepare_dense_conv_v51_plan",
    "replay_dense_conv_v51",
    "verify_dense_conv_v51_result",
]
