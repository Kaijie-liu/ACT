#===- query_dual_scalar_guard_v51.py - Wide Dense guard candidate -----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
"""Isolated V5.1a support-compressed Dense roundoff candidate.

This module has no proof authority and does not modify the frozen V3 replay
or the rejected V5 candidate.  It implements the pre-registered V5.1a Dense
algebra:

* nonnegative support dots are enclosed with platform-gated
  :class:`numpy.longdouble` arithmetic and then minimally rounded upward to
  binary64;
* the nominal coefficient is the unchanged CPU binary64 ``a @ W``;
* rows that can enter the binary64 underflow range additionally receive a
  streamed, bounded-workspace V3-style componentwise penalty; and
* exactly one final scalar guard, the minimum of two independently sound
  guards, is published for each row.

Every returned object is explicitly marked ``proof_authority=False``.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Optional, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as _v3
from act.back_end.hybridz_tf.query_dual_scalar_guard import (
    outward_roundoff_parameters as _f64_roundoff_parameters,
)


_SCHEMA = "act.query_dual_scalar_guard_v51.experimental.v1"
_PLATFORM_SCHEMA = "act.query_dual_scalar_guard_v51.platform.v1"
_F64 = np.dtype(np.float64)
_BOOL = np.dtype(np.bool_)
_F64_TINY = float(np.finfo(np.float64).tiny)
_F64_NORMAL_FREXP_EXPONENT = -1021
_DEFAULT_TILE_WIDTH = 256


class QueryDualScalarGuardV51Error(RuntimeError):
    """Fail-closed experimental error with a stable code."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualScalarGuardV51Error(code, message)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        _fail("INVALID_LAYOUT", "only C-contiguous arrays may be hashed")
    if array.dtype == _F64 and array.dtype.isnative:
        canonical = array.astype(np.dtype("<f8"), copy=False)
        dtype_name = "<f8"
    elif array.dtype == _BOOL:
        canonical = array.astype(np.dtype("u1"), copy=False)
        dtype_name = "|b1"
    else:
        _fail("INVALID_DTYPE", "only native binary64/bool arrays may be hashed")
    header = json.dumps(
        {"dtype": dtype_name, "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_copy(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=dtype
    ).reshape(contiguous.shape)
    if result.flags.writeable or result.flags.owndata:
        _fail("INVALID_STORAGE", "failed to construct bytes-backed storage")
    return result


def _is_bytes_backed(value: np.ndarray) -> bool:
    current: Any = value
    while isinstance(current, np.ndarray):
        if current.flags.writeable:
            return False
        current = current.base
    return isinstance(current, bytes)


def _require_f64_c_array(
    value: Any, *, name: str, ndim: int
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _F64 or not array.dtype.isnative:
        _fail("INVALID_DTYPE", f"{name} must be native binary64")
    if array.ndim != ndim:
        _fail("SHAPE_MISMATCH", f"{name} must have rank {ndim}")
    if not array.flags.c_contiguous:
        _fail("INVALID_LAYOUT", f"{name} must be C-contiguous")
    if not np.all(np.isfinite(array)):
        _fail("NONFINITE", f"{name} contains a non-finite value")
    return array


def _validated_deadline(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    if isinstance(deadline, bool):
        _fail("INVALID_DEADLINE", "deadline must be a monotonic timestamp")
    try:
        result = float(deadline)
    except (TypeError, ValueError, OverflowError):
        _fail("INVALID_DEADLINE", "deadline must be a monotonic timestamp")
    if not math.isfinite(result):
        _fail("INVALID_DEADLINE", "deadline must be a monotonic timestamp")
    return result


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        _fail("DEADLINE_EXPIRED", "V5.1 Dense guard deadline expired")


@dataclass(frozen=True)
class V51Diagnostics:
    """Canonical immutable diagnostic payload."""

    items: Tuple[Tuple[str, str], ...]
    sha256: str

    def __post_init__(self) -> None:
        if self.items != tuple(sorted(self.items)):
            raise ValueError("diagnostic items must be sorted")
        if len({key for key, _ in self.items}) != len(self.items):
            raise ValueError("diagnostic keys must be unique")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.items
        ):
            raise TypeError("diagnostic keys and values must be strings")
        wanted = _canonical_digest(self.items)
        if not hmac.compare_digest(self.sha256, wanted):
            raise ValueError("diagnostic SHA-256 mismatch")

    def as_dict(self) -> Mapping[str, str]:
        return dict(self.items)


def _diagnostics(**values: Any) -> V51Diagnostics:
    items = tuple(
        sorted((str(key), str(value)) for key, value in values.items())
    )
    return V51Diagnostics(items=items, sha256=_canonical_digest(items))


def _longdouble_text(value: np.longdouble) -> str:
    """Canonical numeric text that excludes ABI padding bytes."""

    return np.format_float_scientific(
        np.longdouble(value),
        precision=40,
        unique=False,
        trim="k",
    )


@dataclass(frozen=True)
class WideDotParameters:
    """Directed long-double parameters for one nonnegative dot."""

    operations: int
    gamma_upper_text: str
    tau_upper_text: str
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("experimental parameters have no authority")


@dataclass(frozen=True)
class DenseV51Support:
    """Bytes-backed V5.1 support catalog entry for one Dense/frame pair."""

    support_upper: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    box_mass_upper: float
    weight_shape: Tuple[int, int]
    weight_sha256: str
    max_abs_sha256: str
    support_sha256: str
    binding: Tuple[Tuple[str, str], ...]
    weight_exponent_min: Optional[int]
    weight_exponent_max: Optional[int]
    support_exponent_min: Optional[int]
    support_exponent_max: Optional[int]
    max_abs_exponent_min: Optional[int]
    max_abs_exponent_max: Optional[int]
    global_underflow_risk: bool
    global_subnormal_operand: bool
    disjoint_box_mass: bool
    diagnostics: V51Diagnostics
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("experimental support has no authority")


@dataclass(frozen=True)
class DenseV51GuardResult:
    """Nominal coefficient and row-local V5.1 guard decision."""

    nominal: np.ndarray = field(repr=False, compare=False, hash=False)
    support_mass_upper: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    wide_guard: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    streamed_v3_guard: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    final_guard: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    active_mask: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    fallback_mask: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    fallback_reasons: Tuple[Tuple[str, ...], ...]
    tile_width: int
    diagnostics: V51Diagnostics
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("experimental guard has no authority")


def _ld_up(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.longdouble)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        _fail("NONFINITE", "invalid nonnegative long-double expression")
    result = np.nextafter(
        array, np.longdouble(math.inf), dtype=np.longdouble
    )
    if not np.all(np.isfinite(result)):
        _fail("NONFINITE", "long-double outward successor overflowed")
    return result


def _ld_down_nonnegative(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.longdouble)
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        _fail("NUMERIC_GUARD", "invalid positive long-double denominator")
    result = np.nextafter(
        array, np.longdouble(0.0), dtype=np.longdouble
    )
    if np.any(result <= 0):
        _fail("NUMERIC_GUARD", "long-double denominator rounded to zero")
    return result


def _ld_add_up(left: Any, right: Any) -> np.ndarray:
    return _ld_up(
        np.asarray(left, dtype=np.longdouble)
        + np.asarray(right, dtype=np.longdouble)
    )


def _ld_mul_up(left: Any, right: Any) -> np.ndarray:
    return _ld_up(
        np.asarray(left, dtype=np.longdouble)
        * np.asarray(right, dtype=np.longdouble)
    )


def _ld_div_up(numerator: Any, denominator_lower: Any) -> np.ndarray:
    denominator = np.asarray(denominator_lower, dtype=np.longdouble)
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        _fail("NUMERIC_GUARD", "invalid long-double division denominator")
    return _ld_up(
        np.asarray(numerator, dtype=np.longdouble) / denominator
    )


def _ceil_f64(value: Any) -> np.ndarray:
    """Return the least observed binary64 not below a wide enclosure."""

    wide = np.asarray(value, dtype=np.longdouble)
    if not np.all(np.isfinite(wide)) or np.any(wide < 0):
        _fail("NONFINITE", "invalid long-double value for binary64 ceiling")
    nearest = np.asarray(wide, dtype=np.float64)
    if not np.all(np.isfinite(nearest)):
        _fail("NONFINITE", "wide enclosure does not fit in binary64")
    below = np.asarray(nearest, dtype=np.longdouble) < wide
    if np.any(below):
        nearest = nearest.copy()
        nearest[below] = np.nextafter(
            nearest[below], np.float64(math.inf)
        )
    if not np.all(np.isfinite(nearest)):
        _fail("NONFINITE", "binary64 ceiling overflowed")
    return np.ascontiguousarray(nearest)


def _wide_parameters(width: int) -> Tuple[np.longdouble, np.longdouble, np.longdouble, WideDotParameters]:
    width = int(width)
    if width <= 0:
        _fail("SHAPE_MISMATCH", "dot width must be positive")
    operations = 2 * width + 2
    mantissa_bits = int(np.finfo(np.longdouble).nmant) + 1
    unit_roundoff = np.ldexp(np.longdouble(1.0), -mantissa_bits)
    eta = np.nextafter(
        np.longdouble(0.0),
        np.longdouble(math.inf),
        dtype=np.longdouble,
    )
    product = _ld_mul_up(
        np.longdouble(operations), unit_roundoff
    )
    if product.ndim != 0 or product >= np.longdouble(0.5):
        _fail("NUMERIC_GUARD", "long-double gamma is too large")
    denominator_lower = _ld_down_nonnegative(
        np.longdouble(1.0) - product
    )
    gamma_upper = _ld_div_up(product, denominator_lower)
    tau_numerator = _ld_mul_up(np.longdouble(operations), eta)
    tau_upper = _ld_div_up(tau_numerator, denominator_lower)
    params = WideDotParameters(
        operations=operations,
        gamma_upper_text=_longdouble_text(
            np.longdouble(gamma_upper)
        ),
        tau_upper_text=_longdouble_text(np.longdouble(tau_upper)),
    )
    return (
        np.longdouble(gamma_upper),
        np.longdouble(tau_upper),
        np.longdouble(denominator_lower),
        params,
    )


def _dot_up_longdouble_unchecked(
    left: np.ndarray, right: np.ndarray
) -> Tuple[np.ndarray, WideDotParameters]:
    if (
        left.ndim != 2
        or right.ndim != 1
        or left.shape[1] != right.size
        or left.shape[0] <= 0
        or right.size <= 0
    ):
        _fail("SHAPE_MISMATCH", "invalid nonnegative dot operands")
    if np.any(left < 0) or np.any(right < 0):
        _fail("NEGATIVE_MASS", "negative operand in nonnegative dot")
    gamma_upper, tau_upper, _, params = _wide_parameters(right.size)
    nominal = (
        np.asarray(left, dtype=np.longdouble)
        @ np.asarray(right, dtype=np.longdouble)
    )
    if not np.all(np.isfinite(nominal)) or np.any(nominal < 0):
        _fail("NONFINITE", "long-double nonnegative dot is invalid")
    numerator = _ld_add_up(nominal, tau_upper)
    denominator_lower = _ld_down_nonnegative(
        np.longdouble(1.0) - gamma_upper
    )
    wide_upper = _ld_div_up(numerator, denominator_lower)
    result = _ceil_f64(wide_upper)
    exact_zero = ~np.any(
        (left != 0.0) & (right.reshape(1, -1) != 0.0), axis=1
    )
    result[exact_zero] = 0.0
    if np.any(result < 0) or not np.all(np.isfinite(result)):
        _fail("NUMERIC_GUARD", "invalid wide nonnegative-dot result")
    return result, params


def dot_up_longdouble(left: Any, right: Any) -> np.ndarray:
    """Compute pre-registered ``DotUpL`` and return a binary64 enclosure."""

    check_v51_platform()
    left_array = _require_f64_c_array(left, name="left", ndim=2)
    right_array = _require_f64_c_array(right, name="right", ndim=1)
    result, _ = _dot_up_longdouble_unchecked(left_array, right_array)
    return result


def _fraction_dot_is_enclosed(
    left: np.ndarray, right: np.ndarray, upper: np.ndarray
) -> bool:
    from fractions import Fraction

    for row, stored in zip(left, upper):
        exact = sum(
            (
                Fraction.from_float(float(x))
                * Fraction.from_float(float(y))
                for x, y in zip(row, right)
            ),
            Fraction(0),
        )
        if Fraction.from_float(float(stored)) < exact:
            return False
    return True


@functools.lru_cache(maxsize=1)
def check_v51_platform() -> V51Diagnostics:
    """Gate wide arithmetic before any candidate result is published."""

    longdouble = np.finfo(np.longdouble)
    f64 = np.finfo(np.float64)
    if (
        int(longdouble.nmant) < int(f64.nmant) + 8
        or not longdouble.eps < f64.eps
    ):
        _fail(
            "NUMERIC_PLATFORM",
            "V5.1 requires longdouble with at least eight extra mantissa bits",
        )
    eta64 = np.nextafter(np.float64(0.0), np.float64(math.inf))
    eta_long = np.nextafter(
        np.longdouble(0.0),
        np.longdouble(math.inf),
        dtype=np.longdouble,
    )
    if not (eta64 > 0 and eta64 * np.float64(1.0) == eta64):
        _fail("NUMERIC_PLATFORM", "binary64 gradual-underflow probe failed")
    if not (
        eta_long > 0
        and eta_long * np.longdouble(1.0) == eta_long
    ):
        _fail("NUMERIC_PLATFORM", "longdouble gradual-underflow probe failed")
    one = np.longdouble(1.0)
    tie = np.ldexp(one, -(int(longdouble.nmant) + 1))
    if one + tie != one or not one + np.longdouble(3.0) * tie > one:
        _fail("NUMERIC_PLATFORM", "longdouble RN-even probe failed")

    spot_left = np.asarray(
        [
            [1.0e16, 1.0, 1.0e16],
            [eta64, 0.0, 1.0],
            [0.0, -0.0, 0.0],
        ],
        dtype=np.float64,
    )
    spot_right = np.asarray([1.0, 0.5, eta64], dtype=np.float64)
    spot, _ = _dot_up_longdouble_unchecked(spot_left, spot_right)
    if not _fraction_dot_is_enclosed(spot_left, spot_right, spot):
        _fail("NUMERIC_PLATFORM", "DotUpL Fraction spot audit failed")
    if spot[2] != 0.0:
        _fail("NUMERIC_PLATFORM", "DotUpL structural-zero audit failed")
    return _diagnostics(
        schema=_PLATFORM_SCHEMA,
        experimental=True,
        proof_authority=False,
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        byteorder=sys.byteorder,
        binary64_nmant=int(f64.nmant),
        longdouble_nmant=int(longdouble.nmant),
        longdouble_itemsize=np.dtype(np.longdouble).itemsize,
        longdouble_eps=str(longdouble.eps),
        longdouble_smallest_subnormal=str(eta_long),
        round_to_nearest_even=True,
        gradual_underflow=True,
        fraction_spot_rows=3,
        integration_gate="not-authoritative",
    )


def _canonical_binding(
    binding: Optional[Mapping[str, str]]
) -> Tuple[Tuple[str, str], ...]:
    if binding is None:
        return ()
    if not isinstance(binding, Mapping):
        _fail("INVALID_BINDING", "binding must be a string mapping")
    items = []
    for key, value in binding.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
        ):
            _fail("INVALID_BINDING", "binding entries must be nonempty strings")
        items.append((key, value))
    if len({key for key, _ in items}) != len(items):
        _fail("INVALID_BINDING", "binding keys must be unique")
    return tuple(sorted(items))


def _exponent_extrema(value: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    absolute = np.abs(np.asarray(value, dtype=np.float64))
    nonzero = absolute[absolute != 0.0]
    if nonzero.size == 0:
        return None, None
    _, exponents = np.frexp(nonzero)
    return int(np.min(exponents)), int(np.max(exponents))


def _has_subnormal(value: np.ndarray) -> bool:
    absolute = np.abs(np.asarray(value, dtype=np.float64))
    return bool(np.any((absolute > 0.0) & (absolute < _F64_TINY)))


def _product_may_be_subnormal(
    first_min: Optional[int], second_min: Optional[int]
) -> bool:
    if first_min is None or second_min is None:
        return False
    return first_min + second_min <= _F64_NORMAL_FREXP_EXPONENT


def prepare_dense_support_v51(
    weight: Any,
    predecessor_max_abs: Any,
    *,
    binding: Optional[Mapping[str, str]] = None,
    deadline: Optional[float] = None,
) -> DenseV51Support:
    """Precompute a bytes-backed wide support vector and box mass."""

    checked_deadline = _validated_deadline(deadline)
    _check_deadline(checked_deadline)
    platform_diagnostics = check_v51_platform()
    weight_array = _require_f64_c_array(weight, name="weight", ndim=2)
    max_abs = _require_f64_c_array(
        predecessor_max_abs, name="predecessor_max_abs", ndim=1
    )
    if (
        weight_array.shape[0] <= 0
        or weight_array.shape[1] <= 0
        or weight_array.shape[1] != max_abs.size
    ):
        _fail("SHAPE_MISMATCH", "Dense weight/max-abs dimensions disagree")
    if np.any(max_abs < 0):
        _fail("NEGATIVE_MASS", "predecessor max-abs must be nonnegative")

    absolute_weight = np.ascontiguousarray(
        np.abs(weight_array), dtype=np.float64
    )
    support_upper, support_params = _dot_up_longdouble_unchecked(
        absolute_weight, max_abs
    )
    ones = np.ones((1, max_abs.size), dtype=np.float64)
    box_mass, box_params = _dot_up_longdouble_unchecked(ones, max_abs)
    box_mass_upper = float(box_mass[0])
    _check_deadline(checked_deadline)

    weight_min, weight_max = _exponent_extrema(weight_array)
    max_abs_min, max_abs_max = _exponent_extrema(max_abs)
    support_min, support_max = _exponent_extrema(support_upper)
    global_subnormal = _has_subnormal(weight_array) or _has_subnormal(max_abs)
    global_underflow = _product_may_be_subnormal(
        weight_min, max_abs_min
    )
    nonzero_weight_column = np.any(weight_array != 0.0, axis=0)
    disjoint_box_mass = bool(
        np.any((max_abs != 0.0) & ~nonzero_weight_column)
    )
    support_upper = _immutable_copy(support_upper, _F64)
    canonical_binding = _canonical_binding(binding)
    weight_sha = _array_sha256(weight_array)
    max_abs_sha = _array_sha256(max_abs)
    support_sha = _array_sha256(support_upper)
    diagnostics = _diagnostics(
        schema=_SCHEMA,
        stage="dense-support-precompute",
        experimental=True,
        proof_authority=False,
        weight_shape=f"{weight_array.shape[0]}x{weight_array.shape[1]}",
        weight_sha256=weight_sha,
        max_abs_sha256=max_abs_sha,
        support_sha256=support_sha,
        box_mass_upper_hex=box_mass_upper.hex(),
        support_operations=support_params.operations,
        support_gamma_upper_text=support_params.gamma_upper_text,
        support_tau_upper_text=support_params.tau_upper_text,
        box_operations=box_params.operations,
        box_gamma_upper_text=box_params.gamma_upper_text,
        box_tau_upper_text=box_params.tau_upper_text,
        weight_exponent_min=weight_min,
        weight_exponent_max=weight_max,
        support_exponent_min=support_min,
        support_exponent_max=support_max,
        max_abs_exponent_min=max_abs_min,
        max_abs_exponent_max=max_abs_max,
        global_underflow_risk=global_underflow,
        global_subnormal_operand=global_subnormal,
        disjoint_box_mass=disjoint_box_mass,
        binding_sha256=_canonical_digest(canonical_binding),
        platform_sha256=platform_diagnostics.sha256,
        deadline_enforced=checked_deadline is not None,
        integration_gate="not-authoritative",
    )
    _check_deadline(checked_deadline)
    return DenseV51Support(
        support_upper=support_upper,
        box_mass_upper=box_mass_upper,
        weight_shape=(
            int(weight_array.shape[0]), int(weight_array.shape[1])
        ),
        weight_sha256=weight_sha,
        max_abs_sha256=max_abs_sha,
        support_sha256=support_sha,
        binding=canonical_binding,
        weight_exponent_min=weight_min,
        weight_exponent_max=weight_max,
        support_exponent_min=support_min,
        support_exponent_max=support_max,
        max_abs_exponent_min=max_abs_min,
        max_abs_exponent_max=max_abs_max,
        global_underflow_risk=global_underflow,
        global_subnormal_operand=global_subnormal,
        disjoint_box_mass=disjoint_box_mass,
        diagnostics=diagnostics,
    )


def _validate_support(
    support: DenseV51Support,
    weight: np.ndarray,
    *,
    platform_sha256: str,
) -> None:
    if not isinstance(support, DenseV51Support) or support.proof_authority:
        _fail("INVALID_SUPPORT", "invalid V5.1 support object")
    if (
        not isinstance(support.weight_shape, tuple)
        or len(support.weight_shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in support.weight_shape
        )
        or not isinstance(support.box_mass_upper, float)
        or not math.isfinite(support.box_mass_upper)
        or support.box_mass_upper < 0.0
        or not isinstance(support.binding, tuple)
        or support.binding != tuple(sorted(support.binding))
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(
                isinstance(value, str) and value for value in item
            )
            for item in support.binding
        )
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in (
                support.weight_sha256,
                support.max_abs_sha256,
                support.support_sha256,
            )
        )
        or not isinstance(support.diagnostics, V51Diagnostics)
    ):
        _fail("INVALID_SUPPORT", "malformed V5.1 support metadata")
    values = np.asarray(support.support_upper)
    if (
        values.dtype != _F64
        or not values.dtype.isnative
        or values.ndim != 1
        or values.size != support.weight_shape[0]
        or not values.flags.c_contiguous
        or not _is_bytes_backed(values)
        or np.any(values < 0)
        or not np.all(np.isfinite(values))
    ):
        _fail("INVALID_SUPPORT", "malformed V5.1 support vector")
    if tuple(weight.shape) != support.weight_shape:
        _fail("SHAPE_MISMATCH", "runtime Dense weight shape changed")
    if not hmac.compare_digest(_array_sha256(weight), support.weight_sha256):
        _fail("BINDING_MISMATCH", "runtime Dense weight hash changed")
    if not hmac.compare_digest(
        _array_sha256(values), support.support_sha256
    ):
        _fail("BINDING_MISMATCH", "V5.1 support vector hash changed")
    diagnostic_values = support.diagnostics.as_dict()
    if (
        diagnostic_values.get("proof_authority") != "False"
        or diagnostic_values.get("integration_gate") != "not-authoritative"
        or diagnostic_values.get("weight_sha256") != support.weight_sha256
        or diagnostic_values.get("max_abs_sha256")
        != support.max_abs_sha256
        or diagnostic_values.get("support_sha256") != support.support_sha256
        or diagnostic_values.get("box_mass_upper_hex")
        != support.box_mass_upper.hex()
        or diagnostic_values.get("platform_sha256") != platform_sha256
        or diagnostic_values.get("binding_sha256")
        != _canonical_digest(support.binding)
    ):
        _fail("BINDING_MISMATCH", "V5.1 support diagnostics changed")


def _zero_preserving_upper_sum(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or np.any(left < 0) or np.any(right < 0):
        _fail("NUMERIC_GUARD", "invalid zero-preserving upper sum")
    result = np.empty_like(left)
    left_zero = left == 0.0
    right_zero = right == 0.0
    result[left_zero] = right[left_zero]
    result[right_zero] = left[right_zero]
    both = ~left_zero & ~right_zero
    if np.any(both):
        wide = _ld_add_up(
            np.asarray(left[both], dtype=np.longdouble),
            np.asarray(right[both], dtype=np.longdouble),
        )
        result[both] = _ceil_f64(wide)
    return np.ascontiguousarray(result)


def _stream_v3_penalty(
    coefficients: np.ndarray,
    weight: np.ndarray,
    max_abs: np.ndarray,
    fallback_mask: np.ndarray,
    *,
    tile_width: int,
    deadline: Optional[float],
) -> np.ndarray:
    """Stream componentwise V3 radii in predecessor-column tiles."""

    rows = np.flatnonzero(fallback_mask)
    result = np.full(coefficients.shape[0], math.inf, dtype=np.float64)
    if rows.size == 0:
        return result
    selected = np.ascontiguousarray(coefficients[rows], dtype=np.float64)
    total = np.zeros(rows.size, dtype=np.float64)
    predecessor_width = int(weight.shape[1])
    for start in range(0, predecessor_width, tile_width):
        _check_deadline(deadline)
        end = min(start + tile_width, predecessor_width)
        weight_tile = np.ascontiguousarray(weight[:, start:end])
        max_abs_tile = np.ascontiguousarray(max_abs[start:end])
        _, radius = _v3._matrix_product_with_error(
            selected, weight_tile
        )
        if not np.any(radius):
            continue
        # The component radii are exactly the V3 Higham-plus-eta
        # construction.  Absorb each bounded tile with DotUpL rather than
        # paying a fresh binary64 row-dot allowance per tile; otherwise a
        # tiled fallback can become looser than the untiled V3 baseline
        # solely because it repeats the absorption underflow allowance.
        tile_penalty, _ = _dot_up_longdouble_unchecked(
            np.ascontiguousarray(radius), max_abs_tile
        )
        total = _zero_preserving_upper_sum(total, tile_penalty)
    result[rows] = total
    _check_deadline(deadline)
    return result


def _row_fallback_reasons(
    coefficients: np.ndarray,
    support: DenseV51Support,
    wide_guard: np.ndarray,
    active_mask: np.ndarray,
) -> Tuple[Tuple[Tuple[str, ...], ...], np.ndarray]:
    absolute = np.abs(coefficients)
    coefficient_subnormal = np.any(
        (absolute > 0.0) & (absolute < _F64_TINY), axis=1
    )
    _, coefficient_exponents = np.frexp(absolute)
    exponent_sentinel = np.iinfo(np.int32).max
    row_exponent_min = np.min(
        np.where(
            absolute != 0.0,
            coefficient_exponents,
            exponent_sentinel,
        ),
        axis=1,
    )
    row_exponent_min = np.where(
        row_exponent_min == exponent_sentinel,
        0,
        row_exponent_min,
    )
    support_values = np.asarray(support.support_upper)
    support_subnormal_values = (
        (support_values > 0.0) & (support_values < _F64_TINY)
    )
    used_subnormal_support = np.any(
        (coefficients != 0.0)
        & support_subnormal_values.reshape(1, -1),
        axis=1,
    )
    reasons = []
    fallback = np.zeros(coefficients.shape[0], dtype=np.bool_)
    for query_index in range(coefficients.shape[0]):
        row_reasons = []
        if not active_mask[query_index]:
            reasons.append(())
            continue
        row_min = int(row_exponent_min[query_index])
        if support.global_subnormal_operand:
            row_reasons.append("layer_subnormal_operand")
        if support.global_underflow_risk:
            row_reasons.append("support_product_underflow_risk")
        if support.disjoint_box_mass:
            row_reasons.append("disjoint_box_mass")
        if coefficient_subnormal[query_index]:
            row_reasons.append("coefficient_subnormal")
        if used_subnormal_support[query_index]:
            row_reasons.append("support_subnormal")
        if _product_may_be_subnormal(
            row_min, support.weight_exponent_min
        ):
            row_reasons.append("nominal_product_underflow_risk")
        if _product_may_be_subnormal(
            row_min, support.support_exponent_min
        ):
            row_reasons.append("support_mass_underflow_risk")
        if 0.0 < wide_guard[query_index] < _F64_TINY:
            row_reasons.append("guard_subnormal")
        unique = tuple(sorted(set(row_reasons)))
        reasons.append(unique)
        fallback[query_index] = bool(unique)
    return tuple(reasons), fallback


def dense_support_compressed_guard_v51(
    coefficients: Any,
    weight: Any,
    predecessor_max_abs: Any,
    support: DenseV51Support,
    *,
    tile_width: int = _DEFAULT_TILE_WIDTH,
    deadline: Optional[float] = None,
) -> DenseV51GuardResult:
    """Return byte-identical nominal coefficients and one final row guard."""

    checked_deadline = _validated_deadline(deadline)
    _check_deadline(checked_deadline)
    platform_diagnostics = check_v51_platform()
    coefficient_array = _require_f64_c_array(
        coefficients, name="coefficients", ndim=2
    )
    weight_array = _require_f64_c_array(weight, name="weight", ndim=2)
    max_abs = _require_f64_c_array(
        predecessor_max_abs, name="predecessor_max_abs", ndim=1
    )
    if (
        coefficient_array.shape[0] <= 0
        or coefficient_array.shape[1] <= 0
        or coefficient_array.shape[1] != weight_array.shape[0]
        or max_abs.size != weight_array.shape[1]
    ):
        _fail("SHAPE_MISMATCH", "Dense V5.1 dimensions disagree")
    if np.any(max_abs < 0):
        _fail("NEGATIVE_MASS", "predecessor max-abs must be nonnegative")
    if (
        isinstance(tile_width, bool)
        or not isinstance(tile_width, int)
        or tile_width <= 0
    ):
        _fail("INVALID_TILE", "tile width must be a positive integer")
    _validate_support(
        support, weight_array, platform_sha256=platform_diagnostics.sha256
    )
    if not hmac.compare_digest(
        _array_sha256(max_abs), support.max_abs_sha256
    ):
        _fail("BINDING_MISMATCH", "runtime predecessor max-abs changed")

    nominal = np.asarray(
        coefficient_array @ weight_array, dtype=np.float64
    )
    if (
        nominal.shape
        != (coefficient_array.shape[0], weight_array.shape[1])
        or not np.all(np.isfinite(nominal))
    ):
        _fail("NONFINITE", "Dense nominal matrix product is non-finite")
    nominal = np.ascontiguousarray(nominal)
    _check_deadline(checked_deadline)

    absolute_coefficients = np.ascontiguousarray(
        np.abs(coefficient_array), dtype=np.float64
    )
    support_mass, mass_params = _dot_up_longdouble_unchecked(
        absolute_coefficients, np.asarray(support.support_upper)
    )
    nominal_parameters = _f64_roundoff_parameters(
        2 * int(coefficient_array.shape[1]) + 2
    )
    gamma_term = _ld_mul_up(
        np.longdouble(nominal_parameters.gamma_upper),
        np.asarray(support_mass, dtype=np.longdouble),
    )
    tau_term = _ld_mul_up(
        np.longdouble(nominal_parameters.tau_upper),
        np.longdouble(support.box_mass_upper),
    )
    wide_guard = _ceil_f64(_ld_add_up(gamma_term, tau_term))
    active_mask = np.any(
        (coefficient_array != 0.0)
        & (
            np.asarray(support.support_upper).reshape(1, -1)
            != 0.0
        ),
        axis=1,
    )
    wide_guard[~active_mask] = 0.0
    if np.any(wide_guard < 0) or not np.all(np.isfinite(wide_guard)):
        _fail("NUMERIC_GUARD", "invalid V5.1 wide guard")

    fallback_reasons, fallback_mask = _row_fallback_reasons(
        coefficient_array, support, wide_guard, active_mask
    )
    streamed_guard = _stream_v3_penalty(
        coefficient_array,
        weight_array,
        max_abs,
        fallback_mask,
        tile_width=tile_width,
        deadline=checked_deadline,
    )
    streamed_guard[~fallback_mask] = wide_guard[~fallback_mask]
    final_guard = wide_guard.copy()
    if np.any(fallback_mask):
        final_guard[fallback_mask] = np.minimum(
            wide_guard[fallback_mask], streamed_guard[fallback_mask]
        )
    final_guard[~active_mask] = 0.0
    if np.any(final_guard < 0) or not np.all(np.isfinite(final_guard)):
        _fail("NUMERIC_GUARD", "invalid V5.1 final guard")
    if np.any(final_guard > wide_guard):
        _fail("NUMERIC_GUARD", "fallback increased a V5.1 guard")
    _check_deadline(checked_deadline)

    nominal = _immutable_copy(nominal, _F64)
    support_mass = _immutable_copy(support_mass, _F64)
    wide_guard = _immutable_copy(wide_guard, _F64)
    streamed_guard = _immutable_copy(streamed_guard, _F64)
    final_guard = _immutable_copy(final_guard, _F64)
    active_mask = _immutable_copy(active_mask, _BOOL)
    fallback_mask = _immutable_copy(fallback_mask, _BOOL)
    reason_digest = _canonical_digest(fallback_reasons)
    diagnostics = _diagnostics(
        schema=_SCHEMA,
        stage="dense-runtime-guard",
        experimental=True,
        proof_authority=False,
        coefficient_shape=(
            f"{coefficient_array.shape[0]}x{coefficient_array.shape[1]}"
        ),
        weight_shape=f"{weight_array.shape[0]}x{weight_array.shape[1]}",
        coefficients_sha256=_array_sha256(coefficient_array),
        weight_sha256=support.weight_sha256,
        max_abs_sha256=support.max_abs_sha256,
        support_sha256=support.support_sha256,
        support_catalog_sha256=support.diagnostics.sha256,
        support_mass_sha256=_array_sha256(support_mass),
        nominal_sha256=_array_sha256(nominal),
        wide_guard_sha256=_array_sha256(wide_guard),
        streamed_v3_guard_sha256=_array_sha256(streamed_guard),
        final_guard_sha256=_array_sha256(final_guard),
        active_mask_sha256=_array_sha256(active_mask),
        fallback_mask_sha256=_array_sha256(fallback_mask),
        fallback_reasons_sha256=reason_digest,
        active_rows=int(np.count_nonzero(active_mask)),
        fallback_rows=int(np.count_nonzero(fallback_mask)),
        query_rows=int(coefficient_array.shape[0]),
        tile_width=tile_width,
        mass_operations=mass_params.operations,
        mass_gamma_upper_text=mass_params.gamma_upper_text,
        mass_tau_upper_text=mass_params.tau_upper_text,
        nominal_operations=nominal_parameters.operations,
        nominal_gamma_upper_hex=nominal_parameters.gamma_upper.hex(),
        nominal_tau_upper_hex=nominal_parameters.tau_upper.hex(),
        platform_sha256=platform_diagnostics.sha256,
        deadline_enforced=checked_deadline is not None,
        integration_gate="not-authoritative",
    )
    _check_deadline(checked_deadline)
    return DenseV51GuardResult(
        nominal=nominal,
        support_mass_upper=support_mass,
        wide_guard=wide_guard,
        streamed_v3_guard=streamed_guard,
        final_guard=final_guard,
        active_mask=active_mask,
        fallback_mask=fallback_mask,
        fallback_reasons=fallback_reasons,
        tile_width=tile_width,
        diagnostics=diagnostics,
    )


__all__ = [
    "DenseV51GuardResult",
    "DenseV51Support",
    "QueryDualScalarGuardV51Error",
    "V51Diagnostics",
    "WideDotParameters",
    "check_v51_platform",
    "dense_support_compressed_guard_v51",
    "dot_up_longdouble",
    "prepare_dense_support_v51",
]
