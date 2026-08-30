#===- query_dual_scalar_guard.py - Dense scalar guard candidate -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
"""Experimental support-compressed roundoff guards for Dense replay.

This module is intentionally independent from ``query_dual_replay.py``.  It
does not participate in the authoritative replay path and every public result
has ``proof_authority=False``.  It is a production *candidate* whose remaining
integration obligations include a pinned BLAS operation lemma, session-level
platform gating, box/frame binding, and receipt crosswalks.

For a Dense reverse step

``nominal = fl(coefficients @ weight)``,

let ``m_j`` be a sound componentwise bound on the predecessor magnitude.  A
standard binary64 dot-product model gives

``|nominal_qj - exact_qj| <= gamma_k * sum_i |a_qi W_ij| + tau_k``,

where ``gamma_k = ku / (1-ku)`` and, including gradual underflow,
``tau_k = k*eta / (1-ku)``.  The weighted coefficient error is therefore at
most

``G_q = gamma_k * P_q + tau_k * B``,

with the support-compressed quantities

``s_i = sum_j |W_ij| m_j``,
``B = sum_j m_j``, and
``P_q = sum_i |a_qi| s_i``.

The helpers below compute outward binary64 upper bounds for ``s``, ``B``,
``P``, and ``G`` while retaining the exact same CPU binary64 nominal
``coefficients @ weight`` used by the componentwise V3 implementation.
"""

from __future__ import annotations

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


_SCHEMA = "act.query_dual_scalar_guard.experimental.v1"
_PLATFORM_SCHEMA = "act.query_dual_scalar_guard.platform.v1"
_U = float(2.0**-53)
_ETA = float(np.nextafter(np.float64(0.0), np.float64(math.inf)))
_F64 = np.dtype(np.float64)
_OUTWARD_EXTRA_MANTISSA_BITS = 8


class QueryDualScalarGuardError(RuntimeError):
    """Fail-closed candidate error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualScalarGuardError(code, message)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class HashableDiagnostics:
    """Canonical, Python-hashable diagnostics with a stable SHA-256."""

    items: Tuple[Tuple[str, str], ...]
    sha256: str

    def __post_init__(self) -> None:
        if tuple(sorted(self.items)) != self.items:
            raise ValueError("diagnostic items must be canonically sorted")
        keys = tuple(key for key, _ in self.items)
        if len(set(keys)) != len(keys):
            raise ValueError("diagnostic keys must be unique")
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in self.items):
            raise TypeError("diagnostic keys and values must be strings")
        wanted = _canonical_digest(self.items)
        if not isinstance(self.sha256, str) or not hmac.compare_digest(
            self.sha256, wanted
        ):
            raise ValueError("diagnostic SHA-256 does not match its items")

    def as_dict(self) -> Mapping[str, str]:
        """Return a fresh dictionary suitable for a future receipt."""

        return dict(self.items)


def _diagnostics(**values: Any) -> HashableDiagnostics:
    items = tuple(sorted((str(key), str(value)) for key, value in values.items()))
    return HashableDiagnostics(items=items, sha256=_canonical_digest(items))


@dataclass(frozen=True)
class RoundoffParameters:
    """Outward binary64 parameters for one conservative operation count."""

    operations: int
    gamma_upper: float
    tau_upper: float
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("experimental roundoff parameters have no authority")


@dataclass(frozen=True)
class DenseSupport:
    """Precomputed outward ``s`` and ``B`` bound to weights and max-abs."""

    support_upper: np.ndarray = field(repr=False, compare=False, hash=False)
    box_mass_upper: float
    weight_shape: Tuple[int, int]
    weight_sha256: str
    max_abs_sha256: str
    support_sha256: str
    binding: Tuple[Tuple[str, str], ...]
    diagnostics: HashableDiagnostics
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("experimental Dense support has no authority")


@dataclass(frozen=True)
class DenseScalarGuardResult:
    """CPU binary64 nominal and one outward scalar guard per query."""

    nominal: np.ndarray = field(repr=False, compare=False, hash=False)
    scalar_guard: np.ndarray = field(repr=False, compare=False, hash=False)
    support_mass_upper: np.ndarray = field(
        repr=False, compare=False, hash=False
    )
    diagnostics: HashableDiagnostics
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("experimental scalar guard result has no authority")


def _require_f64_c_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
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


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != _F64 or not array.dtype.isnative:
        _fail("INVALID_DTYPE", "only native binary64 arrays may be hashed")
    if not array.flags.c_contiguous:
        _fail("INVALID_LAYOUT", "only C-contiguous arrays may be hashed")
    canonical = array.astype(np.dtype("<f8"), copy=False)
    header = json.dumps(
        {"dtype": "<f8", "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_f64_copy(value: np.ndarray) -> np.ndarray:
    """Return a C-order binary64 view backed by immutable ``bytes``."""

    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    immutable = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.float64
    ).reshape(contiguous.shape)
    if immutable.flags.writeable or immutable.flags.owndata:
        _fail("INVALID_SUPPORT", "failed to create immutable binary64 storage")
    return immutable


def _is_immutable_bytes_backed(value: np.ndarray) -> bool:
    current: Any = value
    while isinstance(current, np.ndarray):
        if current.flags.writeable:
            return False
        current = current.base
    return isinstance(current, bytes)


def _has_wide_longdouble() -> bool:
    return bool(
        np.finfo(np.longdouble).nmant
        >= (
            np.finfo(np.float64).nmant
            + _OUTWARD_EXTRA_MANTISSA_BITS
        )
        and np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    )


def check_scalar_guard_platform() -> HashableDiagnostics:
    """Fail closed unless the local CPU satisfies the candidate contract.

    This probe is deliberately repeated by precompute and runtime.  It checks
    the scalar and a 1x1 BLAS subnormal path, round-to-nearest-even, native
    binary64, and a wider ``longdouble`` used only for short outward
    expressions.  A production integration must additionally pin the actual
    large-GEMM operation lemma; this local probe alone does not do so.
    """

    if (
        np.dtype(np.float64).itemsize != 8
        or int(np.finfo(np.float64).nmant) != 52
    ):
        _fail("NUMERIC_PLATFORM", "native numpy float64 is not IEEE binary64")
    if not _has_wide_longdouble():
        _fail(
            "NUMERIC_PLATFORM",
            "scalar guard requires longdouble wider than binary64",
        )

    eta = np.nextafter(np.float64(0.0), np.float64(math.inf))
    tiny = np.float64(np.finfo(np.float64).tiny)
    half_tiny = np.float64(tiny * np.float64(0.5))
    eta_product = np.float64(eta * np.float64(1.0))
    eta_dot = float(
        np.asarray([eta], dtype=np.float64)
        @ np.asarray([1.0], dtype=np.float64)
    )
    eta_matmul = float(
        (
            np.asarray([[eta]], dtype=np.float64)
            @ np.asarray([[1.0]], dtype=np.float64)
        )[0, 0]
    )
    if (
        eta <= 0.0
        or half_tiny <= 0.0
        or eta_product != eta
        or eta_dot != float(eta)
        or eta_matmul != float(eta)
    ):
        _fail(
            "NUMERIC_PLATFORM",
            "gradual-underflow probe failed (FTZ/DAZ is unsafe)",
        )

    half_ulp = np.float64(2.0**-53)
    above_half_ulp = np.nextafter(half_ulp, np.float64(math.inf))
    if (
        np.float64(1.0) + half_ulp != np.float64(1.0)
        or np.float64(1.0) + above_half_ulp == np.float64(1.0)
    ):
        _fail("NUMERIC_PLATFORM", "round-to-nearest-even probe failed")

    return _diagnostics(
        schema=_PLATFORM_SCHEMA,
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        byteorder=sys.byteorder,
        binary64_nmant=int(np.finfo(np.float64).nmant),
        longdouble_nmant=int(np.finfo(np.longdouble).nmant),
        required_extra_mantissa_bits=_OUTWARD_EXTRA_MANTISSA_BITS,
        longdouble_eps=str(np.finfo(np.longdouble).eps),
        gradual_underflow=True,
        round_to_nearest_even=True,
        blas_subnormal_dot=True,
        blas_operation_lemma="pending-integration-gate",
    )


def _longdouble_to_f64_up(value: Any, *, where: str) -> Any:
    if not _has_wide_longdouble():
        _fail(
            "NUMERIC_PLATFORM",
            "outward conversion requires longdouble wider than binary64",
        )
    extended = np.asarray(value, dtype=np.longdouble)
    if not np.all(np.isfinite(extended)) or np.any(extended < 0):
        _fail("NONFINITE", f"invalid outward expression at {where}")
    narrowed = np.asarray(extended, dtype=np.float64)
    if not np.all(np.isfinite(narrowed)):
        _fail("NONFINITE", f"outward binary64 conversion overflowed at {where}")
    outward = np.nextafter(narrowed, np.float64(math.inf))
    if not np.all(np.isfinite(outward)):
        _fail("NONFINITE", f"outward successor overflowed at {where}")
    if outward.ndim == 0:
        return float(outward)
    return np.ascontiguousarray(outward)


def _operation_count(width: int) -> int:
    width = int(width)
    if width <= 0:
        _fail("SHAPE_MISMATCH", "dot-product width must be positive")
    return 2 * width + 2


def outward_roundoff_parameters(operations: int) -> RoundoffParameters:
    """Return outward ``gamma_k`` and ``tau_k=k*eta/(1-ku)``."""

    if isinstance(operations, bool) or not isinstance(operations, int):
        _fail("NUMERIC_GUARD", "operation count must be an integer")
    if operations <= 0:
        _fail("NUMERIC_GUARD", "operation count must be positive")
    product = np.longdouble(operations) * np.longdouble(_U)
    if not np.isfinite(product) or product >= np.longdouble(0.5):
        _fail("NUMERIC_GUARD", "operation count is too large")
    denominator = np.longdouble(1.0) - product
    gamma = _longdouble_to_f64_up(
        product / denominator, where="gamma_k"
    )
    tau = _longdouble_to_f64_up(
        (
            np.longdouble(operations)
            * np.longdouble(_ETA)
            / denominator
        ),
        where="tau_k",
    )
    if not (0.0 <= gamma < 1.0) or not (0.0 < tau < math.inf):
        _fail("NUMERIC_GUARD", "invalid outward roundoff parameters")
    return RoundoffParameters(
        operations=operations,
        gamma_upper=float(gamma),
        tau_upper=float(tau),
    )


def _nonnegative_dot_upper(
    left: np.ndarray,
    right: np.ndarray,
    *,
    where: str,
) -> Tuple[np.ndarray, np.ndarray, RoundoffParameters]:
    if (
        left.ndim != 2
        or right.ndim != 1
        or left.shape[1] != right.size
        or left.shape[0] <= 0
        or right.size <= 0
    ):
        _fail("SHAPE_MISMATCH", f"invalid nonnegative dot at {where}")
    if np.any(left < 0.0) or np.any(right < 0.0):
        _fail("NEGATIVE_MASS", f"negative operand at {where}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        _fail("NONFINITE", f"non-finite operand at {where}")

    parameters = outward_roundoff_parameters(_operation_count(right.size))
    nominal = np.asarray(left @ right, dtype=np.float64)
    if (
        nominal.ndim != 1
        or nominal.size != left.shape[0]
        or not np.all(np.isfinite(nominal))
        or np.any(nominal < 0.0)
    ):
        _fail("NONFINITE", f"invalid nominal nonnegative dot at {where}")

    exact_zero = ~np.any(
        (left != 0.0) & (right.reshape(1, -1) != 0.0), axis=1
    )
    numerator = (
        np.asarray(nominal, dtype=np.longdouble)
        + np.longdouble(parameters.tau_upper)
    )
    denominator = (
        np.longdouble(1.0) - np.longdouble(parameters.gamma_upper)
    )
    if denominator <= 0.0:
        _fail("NUMERIC_GUARD", f"nonpositive enclosure denominator at {where}")
    upper = np.asarray(
        _longdouble_to_f64_up(
            numerator / denominator,
            where=f"{where} upper",
        ),
        dtype=np.float64,
    )
    upper[exact_zero] = 0.0
    if np.any(upper < 0.0) or not np.all(np.isfinite(upper)):
        _fail("NUMERIC_GUARD", f"invalid nonnegative upper at {where}")
    return (
        np.ascontiguousarray(nominal),
        np.ascontiguousarray(upper),
        parameters,
    )


def _canonical_binding(
    binding: Optional[Mapping[str, str]],
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
            _fail("INVALID_BINDING", "binding keys and values must be nonempty strings")
        items.append((key, value))
    return tuple(sorted(items))


def _validated_deadline(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    if isinstance(deadline, bool):
        _fail("INVALID_DEADLINE", "deadline must be a finite monotonic time")
    try:
        value = float(deadline)
    except (TypeError, ValueError, OverflowError):
        _fail("INVALID_DEADLINE", "deadline must be a finite monotonic time")
    if not math.isfinite(value):
        _fail("INVALID_DEADLINE", "deadline must be a finite monotonic time")
    return value


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        _fail("DEADLINE_EXPIRED", "Dense scalar-guard deadline expired")


def prepare_dense_support(
    weight: Any,
    predecessor_max_abs: Any,
    *,
    binding: Optional[Mapping[str, str]] = None,
    deadline: Optional[float] = None,
) -> DenseSupport:
    """Precompute outward ``s`` and ``B`` for one frozen Dense/frame pair."""

    checked_deadline = _validated_deadline(deadline)
    _check_deadline(checked_deadline)
    platform_diagnostics = check_scalar_guard_platform()
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
    if np.any(max_abs < 0.0):
        _fail("NEGATIVE_MASS", "predecessor_max_abs must be nonnegative")

    abs_weight = np.ascontiguousarray(np.abs(weight_array), dtype=np.float64)
    _, support_upper, support_parameters = _nonnegative_dot_upper(
        abs_weight,
        max_abs,
        where="Dense support s",
    )
    ones = np.ones((1, max_abs.size), dtype=np.float64)
    _, box_mass, box_parameters = _nonnegative_dot_upper(
        ones,
        max_abs,
        where="Dense box mass B",
    )
    _check_deadline(checked_deadline)
    box_mass_upper = float(box_mass[0])
    if box_mass_upper < 0.0 or not math.isfinite(box_mass_upper):
        _fail("NUMERIC_GUARD", "invalid Dense box mass B")

    support_upper = _immutable_f64_copy(support_upper)
    weight_sha256 = _array_sha256(weight_array)
    max_abs_sha256 = _array_sha256(max_abs)
    support_sha256 = _array_sha256(support_upper)
    canonical_binding = _canonical_binding(binding)
    binding_sha256 = _canonical_digest(canonical_binding)
    diagnostics = _diagnostics(
        schema=_SCHEMA,
        stage="dense-support-precompute",
        experimental=True,
        proof_authority=False,
        box_semantics="predecessor-componentwise-max-abs",
        weight_shape=f"{weight_array.shape[0]}x{weight_array.shape[1]}",
        weight_sha256=weight_sha256,
        max_abs_sha256=max_abs_sha256,
        support_sha256=support_sha256,
        box_mass_upper_hex=box_mass_upper.hex(),
        box_mass_sha256=_array_sha256(
            np.asarray([box_mass_upper], dtype=np.float64)
        ),
        support_operations=support_parameters.operations,
        support_gamma_upper_hex=support_parameters.gamma_upper.hex(),
        support_tau_upper_hex=support_parameters.tau_upper.hex(),
        box_operations=box_parameters.operations,
        box_gamma_upper_hex=box_parameters.gamma_upper.hex(),
        box_tau_upper_hex=box_parameters.tau_upper.hex(),
        binding_sha256=binding_sha256,
        binding_items=len(canonical_binding),
        platform_sha256=platform_diagnostics.sha256,
        deadline_enforced=checked_deadline is not None,
        integration_gate="not-authoritative",
    )
    _check_deadline(checked_deadline)
    return DenseSupport(
        support_upper=support_upper,
        box_mass_upper=box_mass_upper,
        weight_shape=(int(weight_array.shape[0]), int(weight_array.shape[1])),
        weight_sha256=weight_sha256,
        max_abs_sha256=max_abs_sha256,
        support_sha256=support_sha256,
        binding=canonical_binding,
        diagnostics=diagnostics,
    )


def _validate_support(
    support: DenseSupport,
    weight: np.ndarray,
    *,
    platform_sha256: str,
) -> None:
    if not isinstance(support, DenseSupport) or support.proof_authority:
        _fail("INVALID_SUPPORT", "invalid experimental Dense support object")
    if (
        not isinstance(support.weight_shape, tuple)
        or len(support.weight_shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in support.weight_shape
        )
        or not isinstance(support.diagnostics, HashableDiagnostics)
        or not isinstance(support.binding, tuple)
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in support.binding
        )
        or tuple(sorted(support.binding)) != support.binding
        or not all(
            isinstance(value, str) and len(value) == 64
            for value in (
                support.weight_sha256,
                support.max_abs_sha256,
                support.support_sha256,
            )
        )
        or not isinstance(support.box_mass_upper, float)
    ):
        _fail("INVALID_SUPPORT", "malformed Dense support metadata")
    values = np.asarray(support.support_upper)
    if (
        values.dtype != _F64
        or not values.dtype.isnative
        or values.ndim != 1
        or values.size != support.weight_shape[0]
        or not values.flags.c_contiguous
        or not _is_immutable_bytes_backed(values)
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
    ):
        _fail("INVALID_SUPPORT", "malformed Dense support vector")
    if (
        not math.isfinite(support.box_mass_upper)
        or support.box_mass_upper < 0.0
    ):
        _fail("INVALID_SUPPORT", "malformed Dense box mass")
    if tuple(weight.shape) != support.weight_shape:
        _fail("SHAPE_MISMATCH", "runtime Dense weight shape changed")
    if not hmac.compare_digest(_array_sha256(weight), support.weight_sha256):
        _fail("BINDING_MISMATCH", "runtime Dense weight hash changed")
    if not hmac.compare_digest(
        _array_sha256(values), support.support_sha256
    ):
        _fail("BINDING_MISMATCH", "Dense support vector hash changed")
    diagnostic_values = support.diagnostics.as_dict()
    if (
        diagnostic_values.get("proof_authority") != "False"
        or diagnostic_values.get("integration_gate") != "not-authoritative"
        or diagnostic_values.get("weight_sha256") != support.weight_sha256
        or diagnostic_values.get("max_abs_sha256") != support.max_abs_sha256
        or diagnostic_values.get("support_sha256") != support.support_sha256
        or diagnostic_values.get("box_mass_upper_hex")
        != support.box_mass_upper.hex()
        or diagnostic_values.get("box_mass_sha256")
        != _array_sha256(
            np.asarray([support.box_mass_upper], dtype=np.float64)
        )
        or diagnostic_values.get("binding_sha256")
        != _canonical_digest(support.binding)
        or diagnostic_values.get("platform_sha256") != platform_sha256
    ):
        _fail("BINDING_MISMATCH", "Dense support diagnostics changed")


def dense_support_compressed_guard(
    coefficients: Any,
    weight: Any,
    support: DenseSupport,
    *,
    deadline: Optional[float] = None,
) -> DenseScalarGuardResult:
    """Compute CPU f64 ``a@W`` and its support-compressed scalar guard."""

    checked_deadline = _validated_deadline(deadline)
    _check_deadline(checked_deadline)
    platform_diagnostics = check_scalar_guard_platform()
    coefficient_array = _require_f64_c_array(
        coefficients, name="coefficients", ndim=2
    )
    weight_array = _require_f64_c_array(weight, name="weight", ndim=2)
    _validate_support(
        support,
        weight_array,
        platform_sha256=platform_diagnostics.sha256,
    )
    if (
        coefficient_array.shape[0] <= 0
        or coefficient_array.shape[1] <= 0
        or coefficient_array.shape[1] != weight_array.shape[0]
    ):
        _fail("SHAPE_MISMATCH", "Dense coefficient/weight dimensions disagree")

    nominal = np.asarray(coefficient_array @ weight_array, dtype=np.float64)
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
    _, support_mass_upper, mass_parameters = _nonnegative_dot_upper(
        absolute_coefficients,
        support.support_upper,
        where="runtime support mass P",
    )
    _check_deadline(checked_deadline)
    nominal_parameters = outward_roundoff_parameters(
        _operation_count(coefficient_array.shape[1])
    )
    if nominal_parameters.operations != mass_parameters.operations:
        _fail("NUMERIC_GUARD", "nominal and support-mass operation counts differ")

    active_query = np.any(
        (coefficient_array != 0.0)
        & (support.support_upper.reshape(1, -1) != 0.0),
        axis=1,
    )
    guard_expression = (
        np.longdouble(nominal_parameters.gamma_upper)
        * np.asarray(support_mass_upper, dtype=np.longdouble)
        + np.longdouble(nominal_parameters.tau_upper)
        * np.longdouble(support.box_mass_upper)
    )
    scalar_guard = np.asarray(
        _longdouble_to_f64_up(
            guard_expression,
            where="Dense scalar guard G",
        ),
        dtype=np.float64,
    )
    scalar_guard[~active_query] = 0.0
    if np.any(scalar_guard < 0.0) or not np.all(np.isfinite(scalar_guard)):
        _fail("NUMERIC_GUARD", "invalid Dense scalar guard G")

    nominal.setflags(write=False)
    support_mass_upper.setflags(write=False)
    scalar_guard = np.ascontiguousarray(scalar_guard)
    scalar_guard.setflags(write=False)
    diagnostics = _diagnostics(
        schema=_SCHEMA,
        stage="dense-runtime-guard",
        experimental=True,
        proof_authority=False,
        coefficient_shape=(
            f"{coefficient_array.shape[0]}x{coefficient_array.shape[1]}"
        ),
        weight_shape=f"{weight_array.shape[0]}x{weight_array.shape[1]}",
        nominal_operations=nominal_parameters.operations,
        nominal_gamma_upper_hex=nominal_parameters.gamma_upper.hex(),
        nominal_tau_upper_hex=nominal_parameters.tau_upper.hex(),
        coefficients_sha256=_array_sha256(coefficient_array),
        weight_sha256=support.weight_sha256,
        max_abs_sha256=support.max_abs_sha256,
        support_sha256=support.support_sha256,
        support_precompute_diagnostics_sha256=support.diagnostics.sha256,
        support_mass_sha256=_array_sha256(support_mass_upper),
        nominal_sha256=_array_sha256(nominal),
        scalar_guard_sha256=_array_sha256(scalar_guard),
        zero_guard_queries=int(np.count_nonzero(~active_query)),
        query_count=int(coefficient_array.shape[0]),
        binding_sha256=_canonical_digest(support.binding),
        platform_sha256=platform_diagnostics.sha256,
        deadline_enforced=checked_deadline is not None,
        integration_gate="not-authoritative",
    )
    _check_deadline(checked_deadline)
    return DenseScalarGuardResult(
        nominal=nominal,
        scalar_guard=scalar_guard,
        support_mass_upper=support_mass_upper,
        diagnostics=diagnostics,
    )


__all__ = [
    "DenseScalarGuardResult",
    "DenseSupport",
    "HashableDiagnostics",
    "QueryDualScalarGuardError",
    "RoundoffParameters",
    "check_scalar_guard_platform",
    "dense_support_compressed_guard",
    "outward_roundoff_parameters",
    "prepare_dense_support",
]
