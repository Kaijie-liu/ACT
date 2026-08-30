#!/usr/bin/env python3
"""Controlled CPU/BLAS contract candidate for query-dual V5.1.

The contract is intentionally non-authoritative.  It records and tests the
numeric platform that a future sealed V5.1 session must revalidate at commit.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Optional, Sequence, Tuple

import numpy as np
from threadpoolctl import threadpool_info


SCHEMA = "act.query_dual_blas_contract.experimental.v1"
_F64 = np.dtype(np.float64)
_ALLOWED_BLAS = frozenset({"mkl"})
_FALSE_ENV = frozenset({"0", "FALSE", "NO", "OFF"})


class QueryDualBlasContractError(RuntimeError):
    """Stable fail-closed error for a candidate platform contract."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualBlasContractError(code, message)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _deep_freeze(item)
                for key, item in sorted(
                    value.items(), key=lambda pair: str(pair[0])
                )
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_identity() -> Mapping[str, Any]:
    """Return stable CPU fields, excluding dynamic MHz counters."""

    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return {
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    records = [
        block
        for block in path.read_text(
            encoding="ascii", errors="replace"
        ).strip().split("\n\n")
        if block.strip()
    ]
    first = {}
    for line in records[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            first[key.strip()] = value.strip()
    stable_names = (
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "microcode",
        "address sizes",
    )
    identity = {
        name: first.get(name, "") for name in stable_names
    }
    identity["flags"] = sorted(first.get("flags", "").split())
    identity["logical_processors"] = len(records)
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "<f8", "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _check_deadline(deadline: Optional[float], *, where: str) -> None:
    if deadline is None:
        return
    if (
        isinstance(deadline, bool)
        or not math.isfinite(float(deadline))
    ):
        _fail("INVALID_DEADLINE", "deadline must be finite monotonic time")
    if time.monotonic() >= float(deadline):
        _fail("DEADLINE_EXPIRED", f"deadline expired {where}")


def _fegetround() -> int:
    try:
        library = ctypes.CDLL("libm.so.6")
        function = library.fegetround
        function.argtypes = ()
        function.restype = ctypes.c_int
        return int(function())
    except (AttributeError, OSError) as exc:
        raise QueryDualBlasContractError(
            "NUMERIC_PLATFORM", f"cannot query fenv rounding mode: {exc}"
        ) from exc


def _dynamic_threads_disabled() -> Mapping[str, str]:
    values = {
        "MKL_DYNAMIC": os.environ.get("MKL_DYNAMIC", ""),
        "OMP_DYNAMIC": os.environ.get("OMP_DYNAMIC", ""),
    }
    if any(value.strip().upper() not in _FALSE_ENV for value in values.values()):
        _fail(
            "DYNAMIC_THREADS",
            "MKL_DYNAMIC and OMP_DYNAMIC must both be explicitly false",
        )
    return values


def _blas_backend(required_threads: int) -> Mapping[str, Any]:
    if (
        isinstance(required_threads, bool)
        or not isinstance(required_threads, int)
        or required_threads <= 0
    ):
        _fail("INVALID_THREADS", "required_threads must be a positive integer")
    # Materialize one NumPy GEMM before querying loaded thread pools.
    probe = (
        np.ones((2, 3), dtype=np.float64)
        @ np.ones((3, 2), dtype=np.float64)
    )
    if not np.array_equal(probe, np.full((2, 2), 3.0)):
        _fail("NUMERIC_PLATFORM", "BLAS activation probe changed")
    pools = [
        dict(item)
        for item in threadpool_info()
        if item.get("user_api") == "blas"
    ]
    allowed = [
        item
        for item in pools
        if item.get("internal_api") in _ALLOWED_BLAS
    ]
    if len(allowed) != 1:
        _fail(
            "UNSUPPORTED_BLAS",
            f"expected one allowlisted NumPy BLAS, found {len(allowed)}",
        )
    selected = allowed[0]
    if int(selected.get("num_threads", -1)) != required_threads:
        _fail(
            "THREAD_MISMATCH",
            "active BLAS thread count differs from the sealed count",
        )
    filepath = Path(str(selected.get("filepath", ""))).resolve()
    if not filepath.is_file():
        _fail("UNSUPPORTED_BLAS", "BLAS library path is not a file")
    if not selected.get("version"):
        _fail("UNSUPPORTED_BLAS", "BLAS version is unavailable")
    return {
        "selected": {
            **selected,
            "filepath": str(filepath),
            "library_sha256": _file_sha256(filepath),
        },
        "all_blas_pools": pools,
    }


def _fraction_matrix_product(
    left: np.ndarray,
    right: np.ndarray,
) -> Tuple[Tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(
                (
                    Fraction.from_float(float(left[row, inner]))
                    * Fraction.from_float(float(right[inner, column]))
                    for inner in range(left.shape[1])
                ),
                Fraction(0),
            )
            for column in range(right.shape[1])
        )
        for row in range(left.shape[0])
    )


def _matrix_kernel_probe() -> Mapping[str, Any]:
    """Audit a nontrivial MKL GEMM with cancellation and subnormal lanes."""

    eta = np.nextafter(np.float64(0.0), np.float64(math.inf))
    left = np.zeros((17, 31), dtype=np.float64)
    right = np.zeros((31, 13), dtype=np.float64)
    left[0, :3] = np.asarray([1.0e16, 1.0, -1.0e16])
    right[:3, 0] = 1.0
    left[1, 3] = eta
    right[3, 1] = 1.0
    left[2, 4] = -eta
    right[4, 2] = 1.0
    rng = np.random.default_rng(20260728)
    left[3:, 5:] = (
        rng.integers(-8, 9, size=(14, 26)).astype(np.float64) / 8.0
    )
    right[5:, 3:] = (
        rng.integers(-8, 9, size=(26, 10)).astype(np.float64) / 8.0
    )
    actual = np.asarray(left @ right, dtype=np.float64)
    if (
        actual.dtype != _F64
        or actual.shape != (17, 13)
        or not np.all(np.isfinite(actual))
    ):
        _fail("MATRIX_KERNEL", "nontrivial BLAS result is invalid")
    if actual[1, 1] != eta or actual[2, 2] != -eta:
        _fail(
            "GRADUAL_UNDERFLOW",
            "nontrivial BLAS kernel flushed a subnormal lane",
        )

    exact = _fraction_matrix_product(left, right)
    operations = 2 * left.shape[1] + 2
    unit = Fraction(1, 2**53)
    eta_fraction = Fraction(1, 2**1074)
    denominator = Fraction(1) - operations * unit
    gamma = operations * unit / denominator
    tau = operations * eta_fraction / denominator
    maximum_error = Fraction(0)
    maximum_bound = Fraction(0)
    maximum_ratio = Fraction(0)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            stored = Fraction.from_float(float(actual[row, column]))
            error = abs(stored - exact[row][column])
            mass = sum(
                (
                    abs(
                        Fraction.from_float(float(left[row, inner]))
                        * Fraction.from_float(float(right[inner, column]))
                    )
                    for inner in range(left.shape[1])
                ),
                Fraction(0),
            )
            bound = gamma * mass + tau
            if error > bound:
                _fail(
                    "MATRIX_KERNEL",
                    f"BLAS error bound failed at ({row},{column})",
                )
            maximum_error = max(maximum_error, error)
            maximum_bound = max(maximum_bound, bound)
            if bound:
                maximum_ratio = max(maximum_ratio, error / bound)
    return {
        "shape": [[17, 31], [31, 13]],
        "operations_per_dot": operations,
        "left_sha256": _array_sha256(left),
        "right_sha256": _array_sha256(right),
        "actual_sha256": _array_sha256(actual),
        "cancellation_actual_hex": float(actual[0, 0]).hex(),
        "cancellation_exact": str(exact[0][0]),
        "positive_subnormal_hex": float(actual[1, 1]).hex(),
        "negative_subnormal_hex": float(actual[2, 2]).hex(),
        "maximum_error": str(maximum_error),
        "maximum_bound": str(maximum_bound),
        "maximum_error_over_bound": str(maximum_ratio),
        "fraction_cells_checked": int(actual.size),
        "lemma": (
            "arbitrary binary64 dot reduction with no more than "
            "2*n+2 rounded multiply/add/FMA-equivalent operations"
        ),
    }


def _scalar_platform_probe() -> Mapping[str, Any]:
    if (
        np.dtype(np.float64).itemsize != 8
        or int(np.finfo(np.float64).nmant) != 52
        or not np.dtype(np.float64).isnative
    ):
        _fail("NUMERIC_PLATFORM", "NumPy float64 is not native binary64")
    if (
        int(np.finfo(np.longdouble).nmant)
        < int(np.finfo(np.float64).nmant) + 8
    ):
        _fail("NUMERIC_PLATFORM", "longdouble is not sufficiently wider")
    if _fegetround() != 0:
        _fail("ROUNDING_MODE", "fenv is not FE_TONEAREST")
    eta = np.nextafter(np.float64(0.0), np.float64(math.inf))
    if eta <= 0.0 or np.float64(eta * np.float64(1.0)) != eta:
        _fail("GRADUAL_UNDERFLOW", "scalar subnormal probe failed")
    half = np.float64(2.0**-53)
    above = np.nextafter(half, np.float64(math.inf))
    if (
        np.float64(1.0) + half != np.float64(1.0)
        or np.float64(1.0) + above == np.float64(1.0)
    ):
        _fail("ROUNDING_MODE", "round-to-nearest-even probe failed")
    return {
        "fegetround": 0,
        "rounding": "FE_TONEAREST",
        "binary64_nmant": int(np.finfo(np.float64).nmant),
        "longdouble_nmant": int(np.finfo(np.longdouble).nmant),
        "longdouble_eps": str(np.finfo(np.longdouble).eps),
        "gradual_underflow": True,
        "round_to_nearest_even": True,
    }


@dataclass(frozen=True)
class QueryDualBlasContract:
    """Immutable research receipt for the current CPU numeric platform."""

    required_threads: int
    content_sha256: str
    receipt: Mapping[str, Any] = field(repr=False, compare=False)
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("the BLAS contract candidate has no authority")


def probe_query_dual_blas_contract(
    *,
    required_threads: int,
    deadline: Optional[float] = None,
) -> QueryDualBlasContract:
    """Probe and bind one exact CPU/BLAS configuration."""

    _check_deadline(deadline, where="before BLAS contract")
    dynamic = _dynamic_threads_disabled()
    scalar = _scalar_platform_probe()
    _check_deadline(deadline, where="before BLAS backend")
    backend = _blas_backend(required_threads)
    _check_deadline(deadline, where="before matrix kernel")
    matrix = _matrix_kernel_probe()
    _check_deadline(deadline, where="after matrix kernel")
    source_path = Path(__file__).resolve()
    body = {
        "schema": SCHEMA,
        "status": "controlled_candidate",
        "proof_authority": False,
        "device": "cpu",
        "nominal_dtype": "IEEE-754-binary64",
        "required_threads": int(required_threads),
        "dynamic_thread_environment": dynamic,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "scalar_platform": scalar,
        "blas": backend,
        "matrix_kernel": matrix,
        "implementation_path": str(source_path),
        "implementation_sha256": _file_sha256(source_path),
        "cpu_identity": _cpu_identity(),
        "operation_lemma_status": "controlled_fraction_audited_candidate",
        "commit_recheck_required": True,
    }
    content_sha = _json_sha256(body)
    receipt_body = dict(body)
    receipt_body["content_sha256"] = content_sha
    receipt_body["receipt_sha256"] = _json_sha256(receipt_body)
    return QueryDualBlasContract(
        required_threads=int(required_threads),
        content_sha256=content_sha,
        receipt=_deep_freeze(receipt_body),
    )


def validate_query_dual_blas_contract(
    value: Any,
    *,
    recheck_current_platform: bool = False,
    deadline: Optional[float] = None,
) -> bool:
    """Validate receipt integrity and optionally re-probe the live platform."""

    try:
        if (
            not isinstance(value, QueryDualBlasContract)
            or value.proof_authority is not False
        ):
            return False
        receipt = dict(value.receipt)
        claimed_receipt = str(receipt.pop("receipt_sha256"))
        claimed_content = str(receipt.pop("content_sha256"))
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("proof_authority") is not False
            or int(receipt.get("required_threads"))
            != value.required_threads
            or not hmac.compare_digest(
                _json_sha256(receipt), claimed_content
            )
            or not hmac.compare_digest(
                claimed_content, value.content_sha256
            )
        ):
            return False
        receipt_with_content = dict(receipt)
        receipt_with_content["content_sha256"] = claimed_content
        if not hmac.compare_digest(
            _json_sha256(receipt_with_content), claimed_receipt
        ):
            return False
        if recheck_current_platform:
            fresh = probe_query_dual_blas_contract(
                required_threads=value.required_threads,
                deadline=deadline,
            )
            fresh_body = dict(fresh.receipt)
            old_body = dict(value.receipt)
            for key in ("content_sha256", "receipt_sha256"):
                fresh_body.pop(key, None)
                old_body.pop(key, None)
            if _canonical_value(fresh_body) != _canonical_value(old_body):
                return False
        return True
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualBlasContractError,
    ):
        return False


__all__ = [
    "QueryDualBlasContract",
    "QueryDualBlasContractError",
    "SCHEMA",
    "probe_query_dual_blas_contract",
    "validate_query_dual_blas_contract",
]
