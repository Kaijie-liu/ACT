#!/usr/bin/env python3
# ===- property_correlated_plane_bundle.py - correlated plane bundles --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Toy-only, proof-neutral correlated-prefix affine plane bundles.

For one rival, every member of a bundle is an independently justified affine
upper plane ``a_k z + beta_k`` over one immutable prefix frame ``P``.  The
sound bundle upper is

``max_{z in P} min_k (a_k z + beta_k)``.

The candidate LP searches the dual simplex over planes together with the
prefix-row dual.  SciPy output has no authority.  :func:`check_bundle_candidate`
reconstructs the candidate from the original stored binary64 frame with exact
``Fraction`` arithmetic and exports an outward binary64 upper.

Bundles are rival-separable.  Batched execution shares only an immutable
prefix scan; no multiplier or selector is shared across rivals.  This module
is deliberately disconnected from Operator-HZ, the solver, the verifier, and
all benchmark configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


_FRAME_SCHEMA = "act.correlated_plane_prefix_frame.v1"
_PLANE_SCHEMA = "act.correlated_affine_upper_plane.v1"
_BUNDLE_SCHEMA = "act.rival_correlated_plane_bundle.v1"
_CANDIDATE_SCHEMA = "act.correlated_plane_bundle_candidate.v1"
_CHECK_SCHEMA = "act.correlated_plane_bundle_check.v1"
_BATCH_SCHEMA = "act.correlated_plane_bundle_batch.v1"

_CHECK_RECEIPT_KEYS = frozenset(
    {
        "schema", "rival_id", "frame_digest", "bundle_digest",
        "candidate_digest", "exact_upper", "outward_upper",
        "longdouble_nominal", "exact_plane_weights",
        "exact_prefix_ub_dual", "exact_prefix_eq_dual", "checker_source",
        "numeric_lagrangian_reconstruction",
        "candidate_solver_status_has_authority", "plane_validity_authority",
        "prefix_provenance_authority", "proof_authority",
        "verdict_authority", "receipt_sha256",
    }
)
_BATCH_RECEIPT_KEYS = frozenset(
    {
        "schema", "frame_digest", "ordered_rival_ids",
        "ordered_bundle_digests", "ordered_candidate_digests",
        "ordered_check_receipts", "shared_prefix_scan_count",
        "rival_simplex_shared", "all_nonpositive",
        "all_nonpositive_is_diagnostic_only", "plane_validity_authority",
        "prefix_provenance_authority", "proof_authority",
        "verdict_authority", "receipt_sha256",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_schema(value: Any, expected: str) -> bool:
    return type(value) is str and value == expected


def _is_optional_sha256(value: Any) -> bool:
    return type(value) is str and (value == "" or _is_sha256(value))


def _is_finite_longdouble_text(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        return bool(np.isfinite(np.longdouble(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _float_payload(values: np.ndarray) -> list[str]:
    return [float(value).hex() for value in values.reshape(-1)]


def _fraction(value: Any) -> Fraction:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("non-finite binary64 value")
    numerator, denominator = scalar.as_integer_ratio()
    return Fraction(numerator, denominator)


def _readonly_f64(values: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )


def _readonly_i64(values: Any, *, name: str) -> np.ndarray:
    source = tuple(values)
    if any(type(value) is not int for value in source):
        raise ValueError(f"{name} must contain exact builtin ints")
    array = np.array(source, dtype=np.int64, copy=True).reshape(-1)
    if np.any(array < 0) or np.unique(array).size != array.size:
        raise ValueError(f"{name} must contain unique nonnegative ids")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int64)


def _readonly_csr(values: Any) -> sp.csr_matrix:
    matrix = sp.csr_matrix(values, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not matrix.has_canonical_format:
        raise ValueError("prefix matrix must be canonical CSR")
    if matrix.nnz and not np.all(np.isfinite(matrix.data)):
        raise ValueError("prefix matrix contains non-finite coefficients")
    data = np.frombuffer(matrix.data.tobytes(), dtype=matrix.data.dtype)
    indices = np.frombuffer(matrix.indices.tobytes(), dtype=matrix.indices.dtype)
    indptr = np.frombuffer(matrix.indptr.tobytes(), dtype=matrix.indptr.dtype)
    frozen = sp.csr_matrix((data, indices, indptr), shape=matrix.shape, copy=False)
    # SciPy may normalize views during construction.  Rebind the exact
    # bytes-backed arrays so setflags(write=True) cannot reopen them.
    frozen.data = data
    frozen.indices = indices
    frozen.indptr = indptr
    return frozen


def _has_immutable_bytes_base(array: np.ndarray) -> bool:
    current: Any = array
    while type(current) is np.ndarray:
        current = current.base
    return type(current) is bytes


def _is_live_vector(
    value: Any,
    *,
    dtype: np.dtype[Any],
) -> bool:
    return (
        type(value) is np.ndarray
        and value.dtype == dtype
        and value.ndim == 1
        and value.flags.c_contiguous
        and not value.flags.writeable
        and _has_immutable_bytes_base(value)
    )


def _is_live_csr(value: Any) -> bool:
    return (
        type(value) is sp.csr_matrix
        and value.dtype == np.dtype(np.float64)
        and value.ndim == 2
        and value.has_canonical_format
        and _is_live_vector(value.data, dtype=np.dtype(np.float64))
        and value.indices.dtype in (np.dtype(np.int32), np.dtype(np.int64))
        and _is_live_vector(value.indices, dtype=value.indices.dtype)
        and value.indptr.dtype == value.indices.dtype
        and _is_live_vector(value.indptr, dtype=value.indptr.dtype)
        and (not value.nnz or np.all(np.isfinite(value.data)))
    )


def _recursive_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("receipt keys must be exact builtin strings")
        return MappingProxyType(
            {key: _recursive_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_recursive_freeze(item) for item in value)
    return value


def _receipt_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("receipt keys must be exact builtin strings")
            result[key] = _receipt_snapshot(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_receipt_snapshot(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError("receipt contains a non-canonical value")


def _is_receipt_fraction_pair(value: Any) -> bool:
    return (
        type(value) is list
        and len(value) == 2
        and type(value[0]) is int
        and type(value[1]) is int
        and value[1] > 0
    )


def _receipt_fields_have_exact_types(body: Mapping[str, Any], schema: str) -> bool:
    if schema == _CHECK_SCHEMA:
        return (
            type(body.get("rival_id")) is int
            and body["rival_id"] >= 0
            and all(
                _is_sha256(body.get(key))
                for key in ("frame_digest", "bundle_digest", "candidate_digest")
            )
            and _is_receipt_fraction_pair(body.get("exact_upper"))
            and type(body.get("outward_upper")) is str
            and type(body.get("longdouble_nominal")) is str
            and all(
                type(body.get(key)) is list
                and all(_is_receipt_fraction_pair(pair) for pair in body[key])
                for key in (
                    "exact_plane_weights",
                    "exact_prefix_ub_dual",
                    "exact_prefix_eq_dual",
                )
            )
            and body.get("checker_source")
            == "original_stored_binary64_fraction_reconstruction"
            and type(body.get("checker_source")) is str
            and body.get("numeric_lagrangian_reconstruction")
            == "sound_conditioned_on_supplied_binary64_frame_and_planes"
            and type(body.get("numeric_lagrangian_reconstruction")) is str
            and body.get("candidate_solver_status_has_authority") is False
            and body.get("plane_validity_authority") is False
            and body.get("prefix_provenance_authority") is False
            and body.get("proof_authority") is False
            and body.get("verdict_authority") is False
        )
    if schema == _BATCH_SCHEMA:
        return (
            _is_sha256(body.get("frame_digest"))
            and type(body.get("ordered_rival_ids")) is list
            and bool(body["ordered_rival_ids"])
            and all(type(value) is int and value >= 0 for value in body["ordered_rival_ids"])
            and all(
                type(body.get(key)) is list
                and all(_is_sha256(value) for value in body[key])
                for key in (
                    "ordered_bundle_digests",
                    "ordered_candidate_digests",
                    "ordered_check_receipts",
                )
            )
            and type(body.get("shared_prefix_scan_count")) is int
            and body["shared_prefix_scan_count"] == 1
            and body.get("rival_simplex_shared") is False
            and type(body.get("all_nonpositive")) is bool
            and body.get("all_nonpositive_is_diagnostic_only") is True
            and body.get("plane_validity_authority") is False
            and body.get("prefix_provenance_authority") is False
            and body.get("proof_authority") is False
            and body.get("verdict_authority") is False
        )
    return False


def _checked_receipt(
    value: Any,
    *,
    schema: str,
    exact_keys: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("receipt must be a mapping")
    body = _receipt_snapshot(value)
    observed_keys = frozenset(body)
    receipt_digest = body.pop("receipt_sha256", None)
    if (
        observed_keys != exact_keys
        or not _is_exact_schema(body.get("schema"), schema)
        or not _is_sha256(receipt_digest)
        or not _receipt_fields_have_exact_types(body, schema)
        or _digest(body) != receipt_digest
        or body.get("proof_authority") is not False
        or body.get("verdict_authority") is not False
    ):
        raise ValueError("receipt schema, digest, or authority mismatch")
    body["receipt_sha256"] = receipt_digest
    return _recursive_freeze(body)


def _frame_payload(
    A_ub: sp.csr_matrix,
    b_ub: np.ndarray,
    A_eq: sp.csr_matrix,
    b_eq: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    stable_ids: np.ndarray,
    ub_row_keys: Tuple[str, ...],
    eq_row_keys: Tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema": _FRAME_SCHEMA,
        "ub_shape": [int(A_ub.shape[0]), int(A_ub.shape[1])],
        "ub_indptr": [int(value) for value in A_ub.indptr],
        "ub_indices": [int(value) for value in A_ub.indices],
        "ub_data": _float_payload(A_ub.data),
        "b_ub": _float_payload(b_ub),
        "eq_shape": [int(A_eq.shape[0]), int(A_eq.shape[1])],
        "eq_indptr": [int(value) for value in A_eq.indptr],
        "eq_indices": [int(value) for value in A_eq.indices],
        "eq_data": _float_payload(A_eq.data),
        "b_eq": _float_payload(b_eq),
        "lb": _float_payload(lower),
        "ub": _float_payload(upper),
        "stable_var_ids": [int(value) for value in stable_ids],
        "stable_ub_row_keys": list(ub_row_keys),
        "stable_eq_row_keys": list(eq_row_keys),
    }


def _plane_payload(plane: "AffineUpperPlane") -> dict[str, Any]:
    return {
        "schema": _PLANE_SCHEMA,
        "plane_id": plane.plane_id,
        "rival_id": plane.rival_id,
        "property_digest": plane.property_digest,
        "prefix_digest": plane.prefix_digest,
        "stop_digest": plane.stop_digest,
        "producer_receipt_digest": plane.producer_receipt_digest,
        "coefficients": _float_payload(plane.coefficients),
        "intercept": float(plane.intercept).hex(),
        "proof_authority": False,
        "verdict_authority": False,
    }


@dataclass(frozen=True)
class SparsePrefixFrame:
    """Immutable original ``A_ub z<=b_ub``, ``A_eq z=b_eq``, and box."""

    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    A_eq: sp.csr_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    stable_var_ids: np.ndarray
    stable_ub_row_keys: Tuple[str, ...]
    stable_eq_row_keys: Tuple[str, ...]
    semantic_digest: str = ""
    schema: str = _FRAME_SCHEMA

    def __post_init__(self) -> None:
        A_ub = _readonly_csr(self.A_ub)
        b_ub = _readonly_f64(self.b_ub, ndim=1, name="b_ub")
        A_eq = _readonly_csr(self.A_eq)
        b_eq = _readonly_f64(self.b_eq, ndim=1, name="b_eq")
        lower = _readonly_f64(self.lb, ndim=1, name="lb")
        upper = _readonly_f64(self.ub, ndim=1, name="ub")
        stable_ids = _readonly_i64(self.stable_var_ids, name="stable_var_ids")
        ub_row_keys = tuple(self.stable_ub_row_keys)
        eq_row_keys = tuple(self.stable_eq_row_keys)
        if any(type(value) is not str for value in ub_row_keys + eq_row_keys):
            raise ValueError("stable row keys must be exact builtin strings")
        if (
            not _is_exact_schema(self.schema, _FRAME_SCHEMA)
            or not _is_optional_sha256(self.semantic_digest)
            or
            A_ub.shape != (b_ub.size, lower.size)
            or A_eq.shape != (b_eq.size, lower.size)
            or upper.size != lower.size
            or stable_ids.size != lower.size
            or len(ub_row_keys) != b_ub.size
            or len(eq_row_keys) != b_eq.size
            or len(set(ub_row_keys + eq_row_keys)) != len(ub_row_keys) + len(eq_row_keys)
            or any(not key for key in ub_row_keys + eq_row_keys)
            or np.any(lower > upper)
        ):
            raise ValueError("prefix frame shape, bounds, ids, or row keys mismatch")
        payload = _frame_payload(
            A_ub,
            b_ub,
            A_eq,
            b_eq,
            lower,
            upper,
            stable_ids,
            ub_row_keys,
            eq_row_keys,
        )
        observed = _digest(payload)
        if self.semantic_digest != "" and self.semantic_digest != observed:
            raise ValueError("supplied prefix semantic digest is stale")
        object.__setattr__(self, "A_ub", A_ub)
        object.__setattr__(self, "b_ub", b_ub)
        object.__setattr__(self, "A_eq", A_eq)
        object.__setattr__(self, "b_eq", b_eq)
        object.__setattr__(self, "lb", lower)
        object.__setattr__(self, "ub", upper)
        object.__setattr__(self, "stable_var_ids", stable_ids)
        object.__setattr__(self, "stable_ub_row_keys", ub_row_keys)
        object.__setattr__(self, "stable_eq_row_keys", eq_row_keys)
        object.__setattr__(self, "semantic_digest", observed)


@dataclass(frozen=True)
class AffineUpperPlane:
    """One binary64 affine upper plane bound to a rival and prefix frame."""

    plane_id: str
    rival_id: int
    property_digest: str
    prefix_digest: str
    stop_digest: str
    coefficients: np.ndarray
    intercept: float
    producer_receipt_digest: str
    plane_digest: str = ""
    proof_authority: bool = False
    verdict_authority: bool = False
    schema: str = _PLANE_SCHEMA

    def __post_init__(self) -> None:
        coefficients = _readonly_f64(
            self.coefficients, ndim=1, name="plane coefficients"
        )
        if (
            not _is_exact_schema(self.schema, _PLANE_SCHEMA)
            or type(self.plane_id) is not str
            or not self.plane_id
            or type(self.rival_id) is not int
            or self.rival_id < 0
            or any(
                not _is_sha256(value)
                for value in (
                    self.property_digest,
                    self.prefix_digest,
                    self.stop_digest,
                    self.producer_receipt_digest,
                )
            )
            or not math.isfinite(float(self.intercept))
            or not _is_optional_sha256(self.plane_digest)
            or self.proof_authority is not False
            or self.verdict_authority is not False
        ):
            raise ValueError("affine plane identity, digest, or authority mismatch")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))
        payload = _plane_payload(self)
        observed = _digest(payload)
        if self.plane_digest != "" and self.plane_digest != observed:
            raise ValueError("supplied plane digest is stale")
        object.__setattr__(self, "plane_digest", observed)


@dataclass(frozen=True)
class RivalPlaneBundle:
    """Two to four planes for exactly one rival and one prefix frame."""

    rival_id: int
    property_digest: str
    prefix_digest: str
    planes: Tuple[AffineUpperPlane, ...]
    bundle_digest: str = ""
    proof_authority: bool = False
    verdict_authority: bool = False
    schema: str = _BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        planes = tuple(self.planes)
        if (
            not _is_exact_schema(self.schema, _BUNDLE_SCHEMA)
            or type(self.rival_id) is not int
            or self.rival_id < 0
            or type(self.property_digest) is not str
            or not _is_sha256(self.property_digest)
            or not _is_sha256(self.prefix_digest)
            or not 2 <= len(planes) <= 4
            or len({plane.plane_id for plane in planes}) != len(planes)
            or any(type(plane) is not AffineUpperPlane for plane in planes)
            or any(
                plane.rival_id != self.rival_id
                or plane.property_digest != self.property_digest
                or plane.prefix_digest != self.prefix_digest
                for plane in planes
            )
            or len({plane.stop_digest for plane in planes}) < 1
            or not _is_optional_sha256(self.bundle_digest)
            or self.proof_authority is not False
            or self.verdict_authority is not False
        ):
            raise ValueError("rival plane bundle identity, size, or authority mismatch")
        widths = {plane.coefficients.size for plane in planes}
        if len(widths) != 1:
            raise ValueError("all bundle planes must use one prefix width")
        payload = {
            "schema": _BUNDLE_SCHEMA,
            "rival_id": self.rival_id,
            "property_digest": self.property_digest,
            "prefix_digest": self.prefix_digest,
            "plane_digests": [plane.plane_digest for plane in planes],
            "proof_authority": False,
            "verdict_authority": False,
        }
        observed = _digest(payload)
        if self.bundle_digest != "" and self.bundle_digest != observed:
            raise ValueError("supplied bundle digest is stale")
        object.__setattr__(self, "planes", planes)
        object.__setattr__(self, "bundle_digest", observed)


@dataclass(frozen=True)
class PlaneBundleCandidate:
    """Untrusted SciPy simplex/prefix-dual proposal."""

    rival_id: int
    frame_digest: str
    bundle_digest: str
    plane_weights: np.ndarray
    prefix_ub_dual: np.ndarray
    prefix_eq_dual: np.ndarray
    candidate_support: float
    solver_status: int
    solver_message: str
    candidate_digest: str
    elapsed_seconds: float
    proof_authority: bool = False
    verdict_authority: bool = False
    schema: str = _CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        weights = _readonly_f64(
            self.plane_weights, ndim=1, name="candidate plane weights"
        )
        ub_dual = _readonly_f64(
            self.prefix_ub_dual, ndim=1, name="candidate prefix ub dual"
        )
        eq_dual = _readonly_f64(
            self.prefix_eq_dual, ndim=1, name="candidate prefix eq dual"
        )
        if (
            not _is_exact_schema(self.schema, _CANDIDATE_SCHEMA)
            or type(self.rival_id) is not int
            or self.rival_id < 0
            or any(
                not _is_sha256(value)
                for value in (
                    self.frame_digest,
                    self.bundle_digest,
                    self.candidate_digest,
                )
            )
            or type(self.solver_status) is not int
            or type(self.solver_message) is not str
            or not math.isfinite(float(self.candidate_support))
            or not math.isfinite(float(self.elapsed_seconds))
            or float(self.elapsed_seconds) < 0.0
            or np.any(weights < 0.0)
            or np.any(ub_dual < 0.0)
            or weights.sum() <= 0.0
            or self.proof_authority is not False
            or self.verdict_authority is not False
        ):
            raise ValueError("candidate identity, scalar, or authority mismatch")
        object.__setattr__(self, "plane_weights", weights)
        object.__setattr__(self, "prefix_ub_dual", ub_dual)
        object.__setattr__(self, "prefix_eq_dual", eq_dual)
        object.__setattr__(self, "candidate_support", float(self.candidate_support))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        payload = _candidate_payload(
            rival_id=self.rival_id,
            frame_digest=self.frame_digest,
            bundle_digest=self.bundle_digest,
            plane_weights=weights,
            prefix_ub_dual=ub_dual,
            prefix_eq_dual=eq_dual,
            candidate_support=float(self.candidate_support),
            solver_status=self.solver_status,
            solver_message=self.solver_message,
        )
        if _digest(payload) != self.candidate_digest:
            raise ValueError("candidate digest does not bind its numeric fields")


@dataclass(frozen=True)
class CheckedPlaneBundleUpper:
    """Exact original-frame reconstruction of one untrusted candidate."""

    rival_id: int
    frame_digest: str
    bundle_digest: str
    candidate_digest: str
    exact_numerator: int
    exact_denominator: int
    outward_upper: float
    longdouble_nominal: str
    exact_plane_weights: Tuple[Tuple[int, int], ...]
    exact_prefix_ub_dual: Tuple[Tuple[int, int], ...]
    exact_prefix_eq_dual: Tuple[Tuple[int, int], ...]
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    verdict_authority: bool = False
    schema: str = _CHECK_SCHEMA

    def __post_init__(self) -> None:
        weight_pairs = tuple(tuple(value) for value in self.exact_plane_weights)
        ub_pairs = tuple(tuple(value) for value in self.exact_prefix_ub_dual)
        eq_pairs = tuple(tuple(value) for value in self.exact_prefix_eq_dual)
        all_pairs = weight_pairs + ub_pairs + eq_pairs
        pair_shape_valid = all(
            len(pair) == 2
            and type(pair[0]) is int
            and type(pair[1]) is int
            and pair[1] > 0
            and (Fraction(pair[0], pair[1]).numerator,
                 Fraction(pair[0], pair[1]).denominator) == pair
            for pair in all_pairs
        )
        if (
            not _is_exact_schema(self.schema, _CHECK_SCHEMA)
            or type(self.rival_id) is not int
            or self.rival_id < 0
            or any(
                not _is_sha256(value)
                for value in (
                    self.frame_digest,
                    self.bundle_digest,
                    self.candidate_digest,
                )
            )
            or type(self.exact_numerator) is not int
            or type(self.exact_denominator) is not int
            or self.exact_denominator <= 0
            or type(self.outward_upper) is not float
            or not math.isfinite(self.outward_upper)
            or not _is_finite_longdouble_text(self.longdouble_nominal)
            or not pair_shape_valid
            or self.proof_authority is not False
            or self.verdict_authority is not False
        ):
            raise ValueError("checked result identity, numeric, or authority mismatch")
        exact = Fraction(self.exact_numerator, self.exact_denominator)
        if (
            (exact.numerator, exact.denominator)
            != (self.exact_numerator, self.exact_denominator)
            or not weight_pairs
            or sum((Fraction(*pair) for pair in weight_pairs), Fraction(0))
            != Fraction(1)
            or any(Fraction(*pair) < 0 for pair in weight_pairs + ub_pairs)
            or _fraction(self.outward_upper) < exact
        ):
            raise ValueError("outward upper rounds below the exact upper")
        receipt = _checked_receipt(
            self.receipt,
            schema=_CHECK_SCHEMA,
            exact_keys=_CHECK_RECEIPT_KEYS,
        )
        if (
            receipt.get("rival_id") != self.rival_id
            or receipt.get("frame_digest") != self.frame_digest
            or receipt.get("bundle_digest") != self.bundle_digest
            or receipt.get("candidate_digest") != self.candidate_digest
            or tuple(receipt.get("exact_upper", ()))
            != (self.exact_numerator, self.exact_denominator)
            or receipt.get("outward_upper") != float(self.outward_upper).hex()
            or receipt.get("longdouble_nominal") != self.longdouble_nominal
            or tuple(
                tuple(pair) for pair in receipt.get("exact_plane_weights", ())
            ) != weight_pairs
            or tuple(
                tuple(pair) for pair in receipt.get("exact_prefix_ub_dual", ())
            ) != ub_pairs
            or tuple(
                tuple(pair) for pair in receipt.get("exact_prefix_eq_dual", ())
            ) != eq_pairs
            or receipt.get("checker_source")
            != "original_stored_binary64_fraction_reconstruction"
            or receipt.get("numeric_lagrangian_reconstruction")
            != "sound_conditioned_on_supplied_binary64_frame_and_planes"
            or receipt.get("candidate_solver_status_has_authority") is not False
            or receipt.get("plane_validity_authority") is not False
            or receipt.get("prefix_provenance_authority") is not False
        ):
            raise ValueError("checked result receipt does not bind the result")
        object.__setattr__(self, "outward_upper", float(self.outward_upper))
        object.__setattr__(self, "exact_plane_weights", weight_pairs)
        object.__setattr__(self, "exact_prefix_ub_dual", ub_pairs)
        object.__setattr__(self, "exact_prefix_eq_dual", eq_pairs)
        object.__setattr__(self, "receipt", receipt)

    @property
    def exact_upper(self) -> Fraction:
        return Fraction(self.exact_numerator, self.exact_denominator)


def _rechecked_candidate(value: PlaneBundleCandidate) -> PlaneBundleCandidate:
    if type(value) is not PlaneBundleCandidate:
        raise TypeError("candidate must use the exact candidate type")
    return PlaneBundleCandidate(
        rival_id=value.rival_id,
        frame_digest=value.frame_digest,
        bundle_digest=value.bundle_digest,
        plane_weights=value.plane_weights,
        prefix_ub_dual=value.prefix_ub_dual,
        prefix_eq_dual=value.prefix_eq_dual,
        candidate_support=value.candidate_support,
        solver_status=value.solver_status,
        solver_message=value.solver_message,
        candidate_digest=value.candidate_digest,
        elapsed_seconds=value.elapsed_seconds,
        proof_authority=value.proof_authority,
        verdict_authority=value.verdict_authority,
        schema=value.schema,
    )


def _rechecked_upper(
    value: CheckedPlaneBundleUpper,
) -> CheckedPlaneBundleUpper:
    if type(value) is not CheckedPlaneBundleUpper:
        raise TypeError("checked result must use the exact checked type")
    return CheckedPlaneBundleUpper(
        rival_id=value.rival_id,
        frame_digest=value.frame_digest,
        bundle_digest=value.bundle_digest,
        candidate_digest=value.candidate_digest,
        exact_numerator=value.exact_numerator,
        exact_denominator=value.exact_denominator,
        outward_upper=value.outward_upper,
        longdouble_nominal=value.longdouble_nominal,
        exact_plane_weights=value.exact_plane_weights,
        exact_prefix_ub_dual=value.exact_prefix_ub_dual,
        exact_prefix_eq_dual=value.exact_prefix_eq_dual,
        receipt=value.receipt,
        proof_authority=value.proof_authority,
        verdict_authority=value.verdict_authority,
        schema=value.schema,
    )


@dataclass(frozen=True)
class PlaneBundleBatchResult:
    """Ordered rival-separable candidate/check pairs over one prefix scan."""

    rival_ids: Tuple[int, ...]
    candidates: Tuple[PlaneBundleCandidate, ...]
    checked: Tuple[CheckedPlaneBundleUpper, ...]
    all_nonpositive: bool
    shared_prefix_scan_count: int
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    verdict_authority: bool = False
    schema: str = _BATCH_SCHEMA

    def __post_init__(self) -> None:
        rival_ids = tuple(self.rival_ids)
        candidates = tuple(self.candidates)
        checked = tuple(self.checked)
        if any(type(value) is not PlaneBundleCandidate for value in candidates):
            raise ValueError("batch candidates must use the exact candidate type")
        if any(type(value) is not CheckedPlaneBundleUpper for value in checked):
            raise ValueError("batch checks must use the exact checked type")
        candidates = tuple(_rechecked_candidate(value) for value in candidates)
        checked = tuple(_rechecked_upper(value) for value in checked)
        if (
            not _is_exact_schema(self.schema, _BATCH_SCHEMA)
            or not rival_ids
            or any(type(value) is not int or value < 0 for value in rival_ids)
            or len(set(rival_ids)) != len(rival_ids)
            or len(candidates) != len(rival_ids)
            or len(checked) != len(rival_ids)
            or tuple(value.rival_id for value in candidates) != rival_ids
            or tuple(value.rival_id for value in checked) != rival_ids
            or any(
                candidate.frame_digest != result.frame_digest
                or candidate.bundle_digest != result.bundle_digest
                or candidate.candidate_digest != result.candidate_digest
                or len(result.exact_plane_weights) != candidate.plane_weights.size
                or len(result.exact_prefix_ub_dual)
                != candidate.prefix_ub_dual.size
                or len(result.exact_prefix_eq_dual)
                != candidate.prefix_eq_dual.size
                for candidate, result in zip(candidates, checked)
            )
            or type(self.all_nonpositive) is not bool
            or self.all_nonpositive
            is not all(result.exact_upper <= 0 for result in checked)
            or type(self.shared_prefix_scan_count) is not int
            or self.shared_prefix_scan_count != 1
            or self.proof_authority is not False
            or self.verdict_authority is not False
        ):
            raise ValueError("batch result shape, identity, or authority mismatch")
        receipt = _checked_receipt(
            self.receipt,
            schema=_BATCH_SCHEMA,
            exact_keys=_BATCH_RECEIPT_KEYS,
        )
        if (
            len({candidate.frame_digest for candidate in candidates}) != 1
            or receipt.get("frame_digest") != candidates[0].frame_digest
            or tuple(receipt.get("ordered_rival_ids", ())) != rival_ids
            or tuple(receipt.get("ordered_bundle_digests", ()))
            != tuple(candidate.bundle_digest for candidate in candidates)
            or tuple(receipt.get("ordered_candidate_digests", ()))
            != tuple(candidate.candidate_digest for candidate in candidates)
            or tuple(receipt.get("ordered_check_receipts", ()))
            != tuple(result.receipt["receipt_sha256"] for result in checked)
            or receipt.get("shared_prefix_scan_count") != 1
            or receipt.get("rival_simplex_shared") is not False
            or receipt.get("all_nonpositive") is not self.all_nonpositive
            or receipt.get("all_nonpositive_is_diagnostic_only") is not True
            or receipt.get("plane_validity_authority") is not False
            or receipt.get("prefix_provenance_authority") is not False
        ):
            raise ValueError("batch receipt does not bind the result")
        object.__setattr__(self, "rival_ids", rival_ids)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "checked", checked)
        object.__setattr__(self, "receipt", receipt)


def _live_frame_digest(frame: SparsePrefixFrame) -> str:
    return _digest(
        _frame_payload(
            frame.A_ub,
            frame.b_ub,
            frame.A_eq,
            frame.b_eq,
            frame.lb,
            frame.ub,
            frame.stable_var_ids,
            frame.stable_ub_row_keys,
            frame.stable_eq_row_keys,
        )
    )


def _snapshot_frame(frame: SparsePrefixFrame) -> SparsePrefixFrame:
    if type(frame) is not SparsePrefixFrame:
        raise TypeError("frame must use the exact immutable type")
    vectors = (frame.b_ub, frame.b_eq, frame.lb, frame.ub)
    if (
        not _is_exact_schema(frame.schema, _FRAME_SCHEMA)
        or not _is_sha256(frame.semantic_digest)
        or not _is_live_csr(frame.A_ub)
        or not _is_live_csr(frame.A_eq)
        or any(
            not _is_live_vector(value, dtype=np.dtype(np.float64))
            for value in vectors
        )
        or not _is_live_vector(
            frame.stable_var_ids, dtype=np.dtype(np.int64)
        )
        or np.any(frame.stable_var_ids < 0)
        or np.unique(frame.stable_var_ids).size != frame.stable_var_ids.size
        or type(frame.stable_ub_row_keys) is not tuple
        or type(frame.stable_eq_row_keys) is not tuple
        or any(
            type(value) is not str
            for value in frame.stable_ub_row_keys + frame.stable_eq_row_keys
        )
    ):
        raise ValueError("live frame violates the exact snapshot contract")
    return SparsePrefixFrame(
        A_ub=frame.A_ub,
        b_ub=frame.b_ub,
        A_eq=frame.A_eq,
        b_eq=frame.b_eq,
        lb=frame.lb,
        ub=frame.ub,
        stable_var_ids=tuple(frame.stable_var_ids.tolist()),
        stable_ub_row_keys=frame.stable_ub_row_keys,
        stable_eq_row_keys=frame.stable_eq_row_keys,
        semantic_digest=frame.semantic_digest,
        schema=frame.schema,
    )


def _snapshot_bundle(
    frame: SparsePrefixFrame, bundle: RivalPlaneBundle
) -> RivalPlaneBundle:
    if type(bundle) is not RivalPlaneBundle:
        raise TypeError("bundle must use the exact immutable type")
    if type(bundle.planes) is not tuple:
        raise ValueError("live bundle planes must remain an exact tuple")
    planes = []
    for plane in bundle.planes:
        if type(plane) is not AffineUpperPlane:
            raise TypeError("bundle planes must use the exact immutable type")
        planes.append(
            AffineUpperPlane(
                plane_id=plane.plane_id,
                rival_id=plane.rival_id,
                property_digest=plane.property_digest,
                prefix_digest=plane.prefix_digest,
                stop_digest=plane.stop_digest,
                coefficients=plane.coefficients,
                intercept=plane.intercept,
                producer_receipt_digest=plane.producer_receipt_digest,
                plane_digest=plane.plane_digest,
                proof_authority=plane.proof_authority,
                verdict_authority=plane.verdict_authority,
                schema=plane.schema,
            )
        )
    snapshot = RivalPlaneBundle(
        rival_id=bundle.rival_id,
        property_digest=bundle.property_digest,
        prefix_digest=bundle.prefix_digest,
        planes=tuple(planes),
        bundle_digest=bundle.bundle_digest,
        proof_authority=bundle.proof_authority,
        verdict_authority=bundle.verdict_authority,
        schema=bundle.schema,
    )
    _validate_frame_bundle(
        frame,
        snapshot,
        prechecked_frame_digest=frame.semantic_digest,
    )
    return snapshot


def _validate_frame_bundle(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
    *,
    prechecked_frame_digest: str | None = None,
) -> None:
    if type(frame) is not SparsePrefixFrame or type(bundle) is not RivalPlaneBundle:
        raise TypeError("frame and bundle must use exact immutable types")
    if bundle.prefix_digest != frame.semantic_digest:
        raise ValueError("bundle prefix digest does not match the live frame")
    if any(
        plane.coefficients.size != frame.A_ub.shape[1]
        for plane in bundle.planes
    ):
        raise ValueError("bundle plane width does not match prefix frame")
    live_frame = (
        _live_frame_digest(frame)
        if prechecked_frame_digest is None
        else prechecked_frame_digest
    )
    if live_frame != frame.semantic_digest:
        raise ValueError("live prefix frame no longer matches its semantic digest")
    if any(_digest(_plane_payload(plane)) != plane.plane_digest for plane in bundle.planes):
        raise ValueError("live plane no longer matches its semantic digest")
    live_bundle = _digest(
        {
            "schema": _BUNDLE_SCHEMA,
            "rival_id": bundle.rival_id,
            "property_digest": bundle.property_digest,
            "prefix_digest": bundle.prefix_digest,
            "plane_digests": [plane.plane_digest for plane in bundle.planes],
            "proof_authority": False,
            "verdict_authority": False,
        }
    )
    if live_bundle != bundle.bundle_digest:
        raise ValueError("live bundle no longer matches its semantic digest")


def _candidate_payload(
    *,
    rival_id: int,
    frame_digest: str,
    bundle_digest: str,
    plane_weights: np.ndarray,
    prefix_ub_dual: np.ndarray,
    prefix_eq_dual: np.ndarray,
    candidate_support: float,
    solver_status: int,
    solver_message: str,
) -> dict[str, Any]:
    return {
        "schema": _CANDIDATE_SCHEMA,
        "rival_id": rival_id,
        "frame_digest": frame_digest,
        "bundle_digest": bundle_digest,
        "plane_weights": _float_payload(plane_weights),
        "prefix_ub_dual": _float_payload(prefix_ub_dual),
        "prefix_eq_dual": _float_payload(prefix_eq_dual),
        "candidate_support": float(candidate_support).hex(),
        "solver_status": int(solver_status),
        "solver_message": str(solver_message),
        "proof_authority": False,
        "verdict_authority": False,
    }


def _build_candidate_lp(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, list[tuple[float | None, float | None]]]:
    """Build the candidate LP without a dense copy of any prefix matrix."""

    plane_count = len(bundle.planes)
    ub_row_count, width = frame.A_ub.shape
    eq_row_count = frame.A_eq.shape[0]
    plane_matrix = sp.vstack(
        [sp.csr_matrix(plane.coefficients.reshape(1, -1)) for plane in bundle.planes],
        format="csr",
    )
    intercepts = np.fromiter(
        (plane.intercept for plane in bundle.planes),
        dtype=np.float64,
        count=plane_count,
    )
    # Variables are lambda[K], mu[U]>=0, nu[E] free, r+[N], r-[N].
    objective = np.concatenate(
        [intercepts, frame.b_ub, frame.b_eq, frame.ub, -frame.lb]
    )
    top = sp.hstack(
        [
            plane_matrix.transpose(),
            -frame.A_ub.transpose(),
            -frame.A_eq.transpose(),
            -sp.eye(width, format="csr", dtype=np.float64),
            sp.eye(width, format="csr", dtype=np.float64),
        ],
        format="csr",
    )
    bottom = sp.hstack(
        [
            sp.csr_matrix(np.ones((1, plane_count), dtype=np.float64)),
            sp.csr_matrix((1, ub_row_count + eq_row_count + 2 * width)),
        ],
        format="csr",
    )
    equality = sp.vstack([top, bottom], format="csr")
    equality_rhs = np.zeros(width + 1, dtype=np.float64)
    equality_rhs[-1] = 1.0
    bounds = (
        [(0.0, None)] * (plane_count + ub_row_count)
        + [(None, None)] * eq_row_count
        + [(0.0, None)] * (2 * width)
    )
    return objective, equality, equality_rhs, bounds


def _propose_plane_bundle_dual(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
    *,
    timeout_seconds: float = 1.0,
    prechecked_frame_digest: str | None = None,
) -> PlaneBundleCandidate:
    _validate_frame_bundle(
        frame, bundle, prechecked_frame_digest=prechecked_frame_digest
    )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.01 <= float(timeout_seconds) <= 10.0
    ):
        raise ValueError("timeout_seconds must lie in [0.01, 10]")
    started = time.monotonic()
    plane_count = len(bundle.planes)
    ub_row_count = frame.A_ub.shape[0]
    eq_row_count = frame.A_eq.shape[0]
    eq_offset = plane_count + ub_row_count
    objective, equality, equality_rhs, bounds = _build_candidate_lp(frame, bundle)
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={"time_limit": float(timeout_seconds)},
    )
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        raise ValueError(
            "candidate simplex/prefix dual was not produced: " + str(result.message)
        )
    weights = _readonly_f64(
        result.x[:plane_count], ndim=1, name="candidate plane weights"
    )
    prefix_ub_dual = _readonly_f64(
        result.x[plane_count : plane_count + ub_row_count],
        ndim=1,
        name="candidate prefix ub dual",
    )
    prefix_eq_dual = _readonly_f64(
        result.x[eq_offset : eq_offset + eq_row_count],
        ndim=1,
        name="candidate prefix eq dual",
    )
    if (
        np.any(weights < 0.0)
        or np.any(prefix_ub_dual < 0.0)
        or weights.sum() <= 0.0
    ):
        raise ValueError("candidate dual violates nonnegative simplex shape")
    payload = _candidate_payload(
        rival_id=bundle.rival_id,
        frame_digest=frame.semantic_digest,
        bundle_digest=bundle.bundle_digest,
        plane_weights=weights,
        prefix_ub_dual=prefix_ub_dual,
        prefix_eq_dual=prefix_eq_dual,
        candidate_support=float(result.fun),
        solver_status=int(result.status),
        solver_message=str(result.message),
    )
    return PlaneBundleCandidate(
        rival_id=bundle.rival_id,
        frame_digest=frame.semantic_digest,
        bundle_digest=bundle.bundle_digest,
        plane_weights=weights,
        prefix_ub_dual=prefix_ub_dual,
        prefix_eq_dual=prefix_eq_dual,
        candidate_support=float(result.fun),
        solver_status=int(result.status),
        solver_message=str(result.message),
        candidate_digest=_digest(payload),
        elapsed_seconds=float(time.monotonic() - started),
    )


def propose_plane_bundle_dual(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
    *,
    timeout_seconds: float = 1.0,
) -> PlaneBundleCandidate:
    """Use SciPy to propose a simplex/prefix dual; never trust its status."""

    frame_snapshot = _snapshot_frame(frame)
    bundle_snapshot = _snapshot_bundle(frame_snapshot, bundle)
    return _propose_plane_bundle_dual(
        frame_snapshot,
        bundle_snapshot,
        timeout_seconds=timeout_seconds,
        prechecked_frame_digest=frame_snapshot.semantic_digest,
    )


def _validate_candidate(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
    candidate: PlaneBundleCandidate,
    *,
    prechecked_frame_digest: str | None = None,
) -> None:
    _validate_frame_bundle(
        frame, bundle, prechecked_frame_digest=prechecked_frame_digest
    )
    if type(candidate) is not PlaneBundleCandidate:
        raise TypeError("candidate must use the exact candidate type")
    weights = np.asarray(candidate.plane_weights, dtype=np.float64)
    prefix_ub_dual = np.asarray(candidate.prefix_ub_dual, dtype=np.float64)
    prefix_eq_dual = np.asarray(candidate.prefix_eq_dual, dtype=np.float64)
    if (
        candidate.rival_id != bundle.rival_id
        or candidate.frame_digest != frame.semantic_digest
        or candidate.bundle_digest != bundle.bundle_digest
        or weights.shape != (len(bundle.planes),)
        or prefix_ub_dual.shape != (frame.A_ub.shape[0],)
        or prefix_eq_dual.shape != (frame.A_eq.shape[0],)
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(prefix_ub_dual))
        or not np.all(np.isfinite(prefix_eq_dual))
        or np.any(weights < 0.0)
        or np.any(prefix_ub_dual < 0.0)
        or weights.sum() <= 0.0
        or not math.isfinite(candidate.candidate_support)
        or candidate.proof_authority is not False
        or candidate.verdict_authority is not False
    ):
        raise ValueError("candidate identity, shape, numeric value, or authority mismatch")
    payload = _candidate_payload(
        rival_id=candidate.rival_id,
        frame_digest=candidate.frame_digest,
        bundle_digest=candidate.bundle_digest,
        plane_weights=weights,
        prefix_ub_dual=prefix_ub_dual,
        prefix_eq_dual=prefix_eq_dual,
        candidate_support=candidate.candidate_support,
        solver_status=candidate.solver_status,
        solver_message=candidate.solver_message,
    )
    if _digest(payload) != candidate.candidate_digest:
        raise ValueError("candidate digest does not bind the live numeric arrays")


def _exact_candidate_support(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
    candidate: PlaneBundleCandidate,
) -> tuple[
    Fraction,
    Tuple[Fraction, ...],
    Tuple[Fraction, ...],
    Tuple[Fraction, ...],
]:
    weights_raw = tuple(_fraction(value) for value in candidate.plane_weights)
    weight_sum = sum(weights_raw, Fraction(0))
    if weight_sum <= 0:
        raise ValueError("candidate has an empty plane simplex")
    weights = tuple(value / weight_sum for value in weights_raw)
    ub_dual = tuple(_fraction(value) for value in candidate.prefix_ub_dual)
    eq_dual = tuple(_fraction(value) for value in candidate.prefix_eq_dual)
    width = frame.A_ub.shape[1]
    combined = [Fraction(0) for _ in range(width)]
    constant = Fraction(0)
    for weight, plane in zip(weights, bundle.planes):
        constant += weight * _fraction(plane.intercept)
        for column, coefficient in enumerate(plane.coefficients):
            combined[column] += weight * _fraction(coefficient)
    for row, multiplier in enumerate(ub_dual):
        constant += multiplier * _fraction(frame.b_ub[row])
        start = int(frame.A_ub.indptr[row])
        end = int(frame.A_ub.indptr[row + 1])
        for offset in range(start, end):
            column = int(frame.A_ub.indices[offset])
            combined[column] -= multiplier * _fraction(
                frame.A_ub.data[offset]
            )
    for row, multiplier in enumerate(eq_dual):
        constant += multiplier * _fraction(frame.b_eq[row])
        start = int(frame.A_eq.indptr[row])
        end = int(frame.A_eq.indptr[row + 1])
        for offset in range(start, end):
            column = int(frame.A_eq.indices[offset])
            combined[column] -= multiplier * _fraction(
                frame.A_eq.data[offset]
            )
    support = constant
    for column, coefficient in enumerate(combined):
        endpoint = frame.ub[column] if coefficient >= 0 else frame.lb[column]
        support += coefficient * _fraction(endpoint)
    return support, weights, ub_dual, eq_dual


def _outward_binary64(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise OverflowError("exact bundle support is not finite in binary64")
    if _fraction(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise OverflowError("outward rounding exceeds finite binary64")
    return result


def _snapshot_candidate(candidate: PlaneBundleCandidate) -> PlaneBundleCandidate:
    return _rechecked_candidate(candidate)


def _check_bundle_candidate(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
    candidate: PlaneBundleCandidate,
    *,
    prechecked_frame_digest: str | None = None,
) -> CheckedPlaneBundleUpper:
    # The public dataclass copies inputs, and this second private snapshot
    # prevents a caller that forcibly re-enables writes from racing validation
    # against the exact replay below.
    candidate = _snapshot_candidate(candidate)
    _validate_candidate(
        frame,
        bundle,
        candidate,
        prechecked_frame_digest=prechecked_frame_digest,
    )
    exact, weights, ub_dual, eq_dual = _exact_candidate_support(
        frame, bundle, candidate
    )
    outward = _outward_binary64(exact)

    # A separately evaluated long-double nominal is diagnostic only.  The
    # exact stored-float reconstruction above is the authority for `outward`.
    weight_ld = np.asarray([float(value) for value in weights], dtype=np.longdouble)
    ub_dual_ld = np.asarray(candidate.prefix_ub_dual, dtype=np.longdouble)
    eq_dual_ld = np.asarray(candidate.prefix_eq_dual, dtype=np.longdouble)
    plane_ld = np.asarray(
        [plane.coefficients for plane in bundle.planes], dtype=np.longdouble
    )
    residual = weight_ld @ plane_ld
    if frame.A_ub.shape[0]:
        residual -= np.asarray(
            frame.A_ub.transpose() @ ub_dual_ld, dtype=np.longdouble
        ).reshape(-1)
    if frame.A_eq.shape[0]:
        residual -= np.asarray(
            frame.A_eq.transpose() @ eq_dual_ld, dtype=np.longdouble
        ).reshape(-1)
    nominal = np.dot(
        weight_ld,
        np.asarray([plane.intercept for plane in bundle.planes], dtype=np.longdouble),
    )
    nominal += np.dot(
        ub_dual_ld, np.asarray(frame.b_ub, dtype=np.longdouble)
    )
    nominal += np.dot(
        eq_dual_ld, np.asarray(frame.b_eq, dtype=np.longdouble)
    )
    nominal += np.dot(
        np.maximum(residual, np.longdouble(0)),
        np.asarray(frame.ub, dtype=np.longdouble),
    )
    nominal += np.dot(
        np.minimum(residual, np.longdouble(0)),
        np.asarray(frame.lb, dtype=np.longdouble),
    )
    if not np.isfinite(nominal):
        raise ValueError("long-double bundle support is non-finite")
    exact_weight_pairs = tuple(
        (value.numerator, value.denominator) for value in weights
    )
    exact_ub_dual_pairs = tuple(
        (value.numerator, value.denominator) for value in ub_dual
    )
    exact_eq_dual_pairs = tuple(
        (value.numerator, value.denominator) for value in eq_dual
    )
    receipt_body = {
        "schema": _CHECK_SCHEMA,
        "rival_id": bundle.rival_id,
        "frame_digest": frame.semantic_digest,
        "bundle_digest": bundle.bundle_digest,
        "candidate_digest": candidate.candidate_digest,
        "exact_upper": [exact.numerator, exact.denominator],
        "outward_upper": outward.hex(),
        "longdouble_nominal": str(nominal),
        "exact_plane_weights": [list(value) for value in exact_weight_pairs],
        "exact_prefix_ub_dual": [
            list(value) for value in exact_ub_dual_pairs
        ],
        "exact_prefix_eq_dual": [
            list(value) for value in exact_eq_dual_pairs
        ],
        "checker_source": "original_stored_binary64_fraction_reconstruction",
        "numeric_lagrangian_reconstruction": (
            "sound_conditioned_on_supplied_binary64_frame_and_planes"
        ),
        "candidate_solver_status_has_authority": False,
        "plane_validity_authority": False,
        "prefix_provenance_authority": False,
        "proof_authority": False,
        "verdict_authority": False,
    }
    receipt_body["receipt_sha256"] = _digest(receipt_body)
    return CheckedPlaneBundleUpper(
        rival_id=bundle.rival_id,
        frame_digest=frame.semantic_digest,
        bundle_digest=bundle.bundle_digest,
        candidate_digest=candidate.candidate_digest,
        exact_numerator=exact.numerator,
        exact_denominator=exact.denominator,
        outward_upper=outward,
        longdouble_nominal=str(nominal),
        exact_plane_weights=exact_weight_pairs,
        exact_prefix_ub_dual=exact_ub_dual_pairs,
        exact_prefix_eq_dual=exact_eq_dual_pairs,
        receipt=MappingProxyType(receipt_body),
    )


def check_bundle_candidate(
    frame: SparsePrefixFrame,
    bundle: RivalPlaneBundle,
    candidate: PlaneBundleCandidate,
) -> CheckedPlaneBundleUpper:
    """Independently reconstruct a candidate over original stored floats."""

    frame_snapshot = _snapshot_frame(frame)
    bundle_snapshot = _snapshot_bundle(frame_snapshot, bundle)
    return _check_bundle_candidate(
        frame_snapshot,
        bundle_snapshot,
        candidate,
        prechecked_frame_digest=frame_snapshot.semantic_digest,
    )


def solve_plane_bundles_rival_separable(
    frame: SparsePrefixFrame,
    bundles: Sequence[RivalPlaneBundle],
    *,
    timeout_seconds_per_rival: float = 1.0,
) -> PlaneBundleBatchResult:
    """Solve/check ordered rival lanes while sharing no numeric dual state."""

    frame_snapshot = _snapshot_frame(frame)
    ordered = tuple(bundles)
    if not ordered or any(type(bundle) is not RivalPlaneBundle for bundle in ordered):
        raise ValueError("bundles must be a nonempty exact bundle sequence")
    if any(bundle.prefix_digest != frame_snapshot.semantic_digest for bundle in ordered):
        raise ValueError("every rival lane must bind the one shared prefix")
    ordered_snapshots = tuple(
        _snapshot_bundle(frame_snapshot, bundle) for bundle in ordered
    )
    rival_ids = tuple(bundle.rival_id for bundle in ordered_snapshots)
    if len(set(rival_ids)) != len(rival_ids):
        raise ValueError("rival lanes must have unique stable rival ids")
    prechecked_frame_digest = frame_snapshot.semantic_digest
    candidates = tuple(
        _propose_plane_bundle_dual(
            frame_snapshot,
            bundle,
            timeout_seconds=timeout_seconds_per_rival,
            prechecked_frame_digest=prechecked_frame_digest,
        )
        for bundle in ordered_snapshots
    )
    checked = tuple(
        _check_bundle_candidate(
            frame_snapshot,
            bundle,
            candidate,
            prechecked_frame_digest=prechecked_frame_digest,
        )
        for bundle, candidate in zip(ordered_snapshots, candidates)
    )
    receipt_body = {
        "schema": _BATCH_SCHEMA,
        "frame_digest": frame_snapshot.semantic_digest,
        "ordered_rival_ids": list(rival_ids),
        "ordered_bundle_digests": [
            bundle.bundle_digest for bundle in ordered_snapshots
        ],
        "ordered_candidate_digests": [
            candidate.candidate_digest for candidate in candidates
        ],
        "ordered_check_receipts": [
            result.receipt["receipt_sha256"] for result in checked
        ],
        "shared_prefix_scan_count": 1,
        "rival_simplex_shared": False,
        "all_nonpositive": all(result.exact_upper <= 0 for result in checked),
        "all_nonpositive_is_diagnostic_only": True,
        "plane_validity_authority": False,
        "prefix_provenance_authority": False,
        "proof_authority": False,
        "verdict_authority": False,
    }
    receipt_body["receipt_sha256"] = _digest(receipt_body)
    return PlaneBundleBatchResult(
        rival_ids=rival_ids,
        candidates=candidates,
        checked=checked,
        all_nonpositive=bool(receipt_body["all_nonpositive"]),
        shared_prefix_scan_count=1,
        receipt=MappingProxyType(receipt_body),
    )


def exact_single_plane_support(
    frame: SparsePrefixFrame, plane: AffineUpperPlane
) -> Fraction:
    """Small exact oracle used by controlled tests; no verdict authority."""

    if type(frame) is not SparsePrefixFrame or type(plane) is not AffineUpperPlane:
        raise TypeError("frame and plane must use exact immutable types")
    if plane.prefix_digest != frame.semantic_digest:
        raise ValueError("plane/frame digest mismatch")
    # Solve the one-plane LP by enumerating its exact dual through SciPy and
    # replaying that proposal with the same original-frame Fraction checker.
    clone = AffineUpperPlane(
        plane_id=plane.plane_id + ":duplicate_for_single_oracle",
        rival_id=plane.rival_id,
        property_digest=plane.property_digest,
        prefix_digest=plane.prefix_digest,
        stop_digest=plane.stop_digest,
        coefficients=plane.coefficients,
        intercept=plane.intercept,
        producer_receipt_digest=plane.producer_receipt_digest,
    )
    bundle = RivalPlaneBundle(
        rival_id=plane.rival_id,
        property_digest=plane.property_digest,
        prefix_digest=plane.prefix_digest,
        planes=(plane, clone),
    )
    return check_bundle_candidate(
        frame, bundle, propose_plane_bundle_dual(frame, bundle)
    ).exact_upper
