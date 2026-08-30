"""Independent numeric primal certificates for native split HZ blocks.

This module deliberately has a narrow authority boundary.  It can certify
that one immutable binary64 point is feasible for the *supplied* split
numeric frame and return a downward-rounded value of the exact objective
sealed in a live solver-private preformed envelope.  Every authority-bearing
numeric input must be strictly read-only on entry and remain the same object
and read-only through authorization.  This closes ordinary writable-input
ABA, but cannot stop a hostile owner alias from re-enabling writes, mutating,
and restoring bytes concurrently.  A one-use owner wrapper must exclude such
mutation.  The naked API does not establish that the supplied constraint
blocks came from the parent named by the envelope; that binding must be
supplied by a higher-level live-HZ wrapper.

No solver status, solver objective, or solver feasibility flag is consumed.
The sparse blocks remain split throughout; ``hstack`` and ``vstack`` are not
used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
import time
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    build_objective_binding,
    verify_objective_binding,
)
from act.back_end.solver.solver_hz import (
    _hz_longdouble_certificate_platform,
    _hz_read_exact_objective_binding_material_from_factor_envelope,
)


_SCHEMA = "act.hybridz_preformed_split_primal_lower.v1"
_STABLE_IDS_HASH_PREFIX = b"act.hz.preformed.stable_ids.v1\0"
_FRAME_HASH_PREFIX = b"act.hybridz.preformed_split_primal_frame.v1\0"


@dataclass(frozen=True)
class PreformedSplitPrimalCertificateCaps:
    """Hard resource ceilings for one independent replay."""

    max_columns: int = 1_000_000
    max_rows: int = 1_000_000
    max_constraint_nnz: int = 4_000_000
    max_exact_objective_terms: int = 250_000
    max_exact_equality_rows: int = 250_000
    max_exact_equality_nnz: int = 500_000
    max_exact_upper_rows: int = 16_384
    max_exact_upper_nnz: int = 500_000
    chunk_elements: int = 65_536


DEFAULT_PREFORMED_SPLIT_PRIMAL_CAPS = PreformedSplitPrimalCertificateCaps()


class _Deadline(TimeoutError):
    pass


class _CapExceeded(ValueError):
    pass


def _deadline(deadline: Optional[float], stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise _Deadline(stage)


def _canonical_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one canonical lowercase SHA-256")
    return value


def _caps(value: Any) -> PreformedSplitPrimalCertificateCaps:
    if value is None:
        result = DEFAULT_PREFORMED_SPLIT_PRIMAL_CAPS
    elif type(value) is PreformedSplitPrimalCertificateCaps:
        result = value
    elif isinstance(value, Mapping):
        expected = tuple(asdict(DEFAULT_PREFORMED_SPLIT_PRIMAL_CAPS))
        if set(value) != set(expected):
            raise ValueError("caps mapping must contain exactly the known fields")
        result = PreformedSplitPrimalCertificateCaps(
            **{name: value[name] for name in expected}
        )
    else:
        raise ValueError("caps must be the frozen cap record or a strict mapping")
    for name, item in asdict(result).items():
        if type(item) is not int or item <= 0:
            raise ValueError(f"caps.{name} must be a strict positive integer")
    return result


def _dense_f64(
    values: Any,
    *,
    size: Optional[int],
    name: str,
    require_readonly: bool = False,
) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.dtype(np.float64)
        or values.ndim != 1
        or not values.flags.c_contiguous
        or not values.flags.aligned
        or (require_readonly and values.flags.writeable)
        or (size is not None and int(values.size) != int(size))
    ):
        raise ValueError(
            f"{name} must be an aligned contiguous one-dimensional "
            f"binary64 array{' that is strictly readonly' if require_readonly else ''}"
        )
    return values


def _candidate_snapshot(
    values: Any,
    *,
    size: int,
    name: str,
    deadline: Optional[float],
    chunk: int,
) -> Tuple[np.ndarray, str, int]:
    source = _dense_f64(values, size=size, name=name)
    if source.flags.writeable:
        raise ValueError(f"{name} must be readonly before snapshot")
    packed = source.tobytes(order="C")
    snapshot = np.frombuffer(packed, dtype="<f8")
    if snapshot.flags.writeable or not snapshot.flags.c_contiguous:
        raise AssertionError("immutable candidate snapshot postcondition failed")
    digest = hashlib.sha256()
    digest.update(b"act.hybridz.primal_candidate_snapshot.v1\0")
    digest.update(name.encode("ascii") + b"\0")
    digest.update(np.asarray(snapshot.shape, dtype="<i8").tobytes())
    for start in range(0, int(snapshot.size), chunk):
        _deadline(deadline, f"snapshot_{name}")
        local = snapshot[start : start + chunk]
        if not np.all(np.isfinite(local)):
            raise ValueError(f"{name} contains a non-finite value")
        digest.update(memoryview(local.view(np.uint8)))
    return snapshot, digest.hexdigest(), len(packed)


def _stable_ids(
    values: Any,
    *,
    size: int,
    name: str,
    deadline: Optional[float],
    chunk: int,
) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.dtype(np.int64)
        or values.ndim != 1
        or int(values.size) != int(size)
        or not values.flags.c_contiguous
        or not values.flags.aligned
        or values.flags.writeable
    ):
        raise ValueError(
            f"{name} must be an aligned contiguous readonly int64 vector"
        )
    for start in range(0, int(values.size), chunk):
        _deadline(deadline, f"validate_{name}")
        if np.any(values[start : start + chunk] < 0):
            raise ValueError(f"{name} contains a negative stable id")
    return values


def _strictly_sorted_unique(
    values: np.ndarray,
    *,
    name: str,
    deadline: Optional[float],
    chunk: int,
) -> Tuple[np.ndarray, bool]:
    """Return a sorted vector, copying only for a non-monotone input."""

    monotone = True
    previous = None
    for start in range(0, int(values.size), chunk):
        _deadline(deadline, f"unique_{name}")
        local = values[start : start + chunk]
        if local.size:
            if previous is not None and int(local[0]) <= previous:
                monotone = False
            if local.size > 1 and np.any(local[1:] <= local[:-1]):
                monotone = False
            previous = int(local[-1])
    if monotone:
        return values, False
    ordered = np.sort(values, kind="quicksort")
    for start in range(1, int(ordered.size), chunk):
        _deadline(deadline, f"sorted_unique_{name}")
        stop = min(int(ordered.size), start + chunk)
        if np.any(ordered[start:stop] == ordered[start - 1 : stop - 1]):
            raise ValueError(f"{name} contains duplicate stable ids")
    return ordered, True


def _reject_stable_id_intersection(
    left: np.ndarray,
    right: np.ndarray,
    *,
    deadline: Optional[float],
    chunk: int,
) -> None:
    if not left.size or not right.size:
        return
    small, large = (left, right) if left.size <= right.size else (right, left)
    for start in range(0, int(small.size), chunk):
        _deadline(deadline, "stable_id_disjointness")
        local = small[start : start + chunk]
        positions = np.searchsorted(large, local)
        inside = positions < large.size
        if np.any(inside):
            if np.any(large[positions[inside]] == local[inside]):
                raise ValueError(
                    "continuous and binary stable ids must be disjoint"
                )


def _hash_named_array(
    digest: Any,
    *,
    name: str,
    values: np.ndarray,
    canonical_dtype: np.dtype,
    deadline: Optional[float],
    chunk_bytes: int,
) -> None:
    digest.update(name.encode("ascii") + b"\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.dtype(canonical_dtype).str.encode("ascii") + b"\0")
    raw = values.view(np.uint8).reshape(-1)
    for start in range(0, int(raw.size), chunk_bytes):
        _deadline(deadline, f"hash_{name}")
        digest.update(memoryview(raw[start : start + chunk_bytes]))


def _stable_ids_sha256(
    continuous_ids: np.ndarray,
    binary_ids: np.ndarray,
    *,
    deadline: Optional[float],
    chunk_bytes: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(_STABLE_IDS_HASH_PREFIX)
    _hash_named_array(
        digest,
        name="continuous_col_ids",
        values=continuous_ids,
        canonical_dtype=np.dtype("<i8"),
        deadline=deadline,
        chunk_bytes=chunk_bytes,
    )
    _hash_named_array(
        digest,
        name="binary_col_ids",
        values=binary_ids,
        canonical_dtype=np.dtype("<i8"),
        deadline=deadline,
        chunk_bytes=chunk_bytes,
    )
    return digest.hexdigest()


def _csr(
    matrix: Any,
    *,
    rows: int,
    columns: int,
    name: str,
    deadline: Optional[float],
    chunk: int,
) -> sp.csr_matrix:
    if (
        not sp.isspmatrix_csr(matrix)
        or matrix.dtype != np.dtype(np.float64)
        or matrix.shape != (int(rows), int(columns))
        or not matrix.has_canonical_format
        or matrix.indptr.ndim != 1
        or matrix.indices.ndim != 1
        or matrix.data.ndim != 1
        or int(matrix.indptr.size) != int(rows) + 1
        or int(matrix.indices.size) != int(matrix.data.size)
        or matrix.indptr.dtype.kind != "i"
        or matrix.indices.dtype.kind != "i"
        or any(
            values.flags.writeable
            or not values.flags.c_contiguous
            or not values.flags.aligned
            for values in (matrix.indptr, matrix.indices, matrix.data)
        )
    ):
        raise ValueError(
            f"{name} must be canonical binary64 CSR with the exact shape "
            "and strictly readonly aligned contiguous data/indices/indptr"
        )
    for start in range(0, int(matrix.nnz), chunk):
        _deadline(deadline, f"validate_{name}")
        if not np.all(np.isfinite(matrix.data[start : start + chunk])):
            raise ValueError(f"{name} contains non-finite coefficients")
    return matrix


def _authority_input_identity_records(
    *,
    matrices: Tuple[Tuple[str, sp.csr_matrix], ...],
    arrays: Tuple[Tuple[str, np.ndarray], ...],
) -> Tuple[Tuple[Any, ...], ...]:
    records = []
    for name, matrix in matrices:
        records.append(
            (
                "csr",
                name,
                matrix,
                id(matrix),
                matrix.data,
                id(matrix.data),
                matrix.indices,
                id(matrix.indices),
                matrix.indptr,
                id(matrix.indptr),
            )
        )
    for name, values in arrays:
        records.append(("array", name, values, id(values)))
    return tuple(records)


def _recheck_authority_input_identity_and_readonly(
    records: Tuple[Tuple[Any, ...], ...],
) -> None:
    """Reject replacement or write re-enablement before authorization."""

    for record in records:
        if record[0] == "csr":
            (
                _,
                name,
                matrix,
                matrix_identity,
                data,
                data_identity,
                indices,
                indices_identity,
                indptr,
                indptr_identity,
            ) = record
            if (
                id(matrix) != matrix_identity
                or matrix.data is not data
                or id(matrix.data) != data_identity
                or matrix.indices is not indices
                or id(matrix.indices) != indices_identity
                or matrix.indptr is not indptr
                or id(matrix.indptr) != indptr_identity
                or any(
                    values.flags.writeable
                    or not values.flags.c_contiguous
                    or not values.flags.aligned
                    for values in (data, indices, indptr)
                )
            ):
                raise PermissionError(
                    f"{name} identity or readonly contract changed"
                )
        else:
            _, name, values, identity = record
            if (
                id(values) != identity
                or values.flags.writeable
                or not values.flags.c_contiguous
                or not values.flags.aligned
            ):
                raise PermissionError(
                    f"{name} identity or readonly contract changed"
                )


def _hash_csr(
    digest: Any,
    *,
    name: str,
    matrix: sp.csr_matrix,
    deadline: Optional[float],
    chunk_bytes: int,
) -> None:
    digest.update(name.encode("ascii") + b"\0")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    for suffix, values in (
        ("indptr", matrix.indptr),
        ("indices", matrix.indices),
        ("data", matrix.data),
    ):
        _hash_named_array(
            digest,
            name=f"{name}_{suffix}",
            values=values,
            canonical_dtype=values.dtype,
            deadline=deadline,
            chunk_bytes=chunk_bytes,
        )


def _frame_sha256(
    *,
    matrices: Tuple[Tuple[str, sp.csr_matrix], ...],
    arrays: Tuple[Tuple[str, np.ndarray], ...],
    deadline: Optional[float],
    chunk_bytes: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(_FRAME_HASH_PREFIX)
    for name, matrix in matrices:
        _hash_csr(
            digest,
            name=name,
            matrix=matrix,
            deadline=deadline,
            chunk_bytes=chunk_bytes,
        )
    for name, values in arrays:
        _hash_named_array(
            digest,
            name=name,
            values=values,
            canonical_dtype=values.dtype,
            deadline=deadline,
            chunk_bytes=chunk_bytes,
        )
    return digest.hexdigest()


def _roundoff_guard(mass: np.longdouble, operations: int) -> np.longdouble:
    dtype = np.longdouble
    if mass < 0 or not np.isfinite(mass) or operations < 0:
        raise ValueError("invalid long-double row mass")
    scaled = dtype(operations) * dtype(np.finfo(dtype).eps)
    if scaled >= dtype(1.0) / dtype(64.0):
        raise ValueError("long-double row exceeds gamma regime")
    if mass == 0:
        return dtype(0)
    factor = np.nextafter(dtype(16) * scaled, dtype(np.inf))
    guard = np.nextafter(factor * mass, dtype(np.inf))
    if not np.isfinite(guard):
        raise ValueError("long-double row guard overflowed")
    return guard


def _longdouble_row_interval(
    *,
    row: int,
    continuous_matrix: sp.csr_matrix,
    binary_matrix: sp.csr_matrix,
    continuous_candidate: np.ndarray,
    binary_candidate: np.ndarray,
    rhs: float,
    deadline: Optional[float],
    chunk: int,
) -> Tuple[np.longdouble, np.longdouble, np.longdouble]:
    dtype = np.longdouble
    estimate = -dtype(rhs)
    mass = np.nextafter(np.abs(dtype(rhs)), dtype(np.inf))
    terms = 1
    reductions = 0
    for matrix, candidate in (
        (continuous_matrix, continuous_candidate),
        (binary_matrix, binary_candidate),
    ):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        while start < stop:
            _deadline(deadline, "longdouble_upper_rows")
            chunk_stop = min(stop, start + chunk)
            indices = matrix.indices[start:chunk_stop]
            products = matrix.data[start:chunk_stop].astype(dtype)
            products *= candidate[indices].astype(dtype)
            if not np.all(np.isfinite(products)):
                raise ValueError("long-double row product overflowed")
            local_sum = np.sum(products, dtype=dtype)
            np.abs(products, out=products)
            positive = products > 0
            if np.any(positive):
                products[positive] = np.nextafter(
                    products[positive], dtype(np.inf)
                )
            local_mass = np.sum(products, dtype=dtype)
            if local_mass > 0:
                local_mass = np.nextafter(local_mass, dtype(np.inf))
            estimate = estimate + local_sum
            mass = np.nextafter(mass + local_mass, dtype(np.inf))
            terms += int(products.size)
            reductions += 1
            start = chunk_stop
    # Products, within-chunk reductions, and cross-chunk additions all round.
    # ``4*terms + 4*reductions + 32`` intentionally dominates their actual
    # operation count; the 16x gamma margin in ``_roundoff_guard`` then also
    # covers the outward mass construction above.
    guard = _roundoff_guard(
        mass,
        4 * terms + 4 * reductions + 32,
    )
    lower = np.nextafter(estimate - guard, dtype(-np.inf))
    upper = np.nextafter(estimate + guard, dtype(np.inf))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("long-double row interval is non-finite")
    return lower, upper, guard


def _exact_row_value(
    *,
    row: int,
    continuous_matrix: sp.csr_matrix,
    binary_matrix: sp.csr_matrix,
    continuous_candidate: np.ndarray,
    binary_candidate: np.ndarray,
    deadline: Optional[float],
) -> Fraction:
    value = Fraction(0)
    visited = 0
    for matrix, candidate in (
        (continuous_matrix, continuous_candidate),
        (binary_matrix, binary_candidate),
    ):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        for position in range(start, stop):
            if visited % 4096 == 0:
                _deadline(deadline, "exact_constraint_rows")
            value += Fraction.from_float(float(matrix.data[position])) * (
                Fraction.from_float(float(candidate[int(matrix.indices[position])]))
            )
            visited += 1
    return value


def _validate_bounds(
    *,
    candidate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    name: str,
    deadline: Optional[float],
    chunk: int,
) -> None:
    for start in range(0, int(candidate.size), chunk):
        _deadline(deadline, f"bounds_{name}")
        stop = min(int(candidate.size), start + chunk)
        local = candidate[start:stop]
        local_lower = lower[start:stop]
        local_upper = upper[start:stop]
        if (
            not np.all(np.isfinite(local_lower))
            or not np.all(np.isfinite(local_upper))
            or np.any(local_lower > local_upper)
        ):
            raise ValueError(f"{name} bounds are invalid")
        if np.any(local < local_lower) or np.any(local > local_upper):
            raise ValueError(f"{name} candidate violates an exact bound")


def _term_positions(
    terms: Tuple[Tuple[int, Fraction], ...],
    stable_ids: np.ndarray,
    *,
    name: str,
    deadline: Optional[float],
    chunk: int,
) -> np.ndarray:
    count = len(terms)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    term_ids = np.fromiter(
        (stable_id for stable_id, _ in terms),
        dtype=np.int64,
        count=count,
    )
    positions = np.full(count, -1, dtype=np.int64)
    for start in range(0, int(stable_ids.size), chunk):
        _deadline(deadline, f"map_{name}_objective_terms")
        stop = min(int(stable_ids.size), start + chunk)
        local = stable_ids[start:stop]
        slots = np.searchsorted(term_ids, local)
        inside = slots < count
        if np.any(inside):
            local_offsets = np.flatnonzero(inside)
            candidate_slots = slots[inside]
            matches = term_ids[candidate_slots] == local[inside]
            if np.any(matches):
                positions[candidate_slots[matches]] = (
                    start + local_offsets[matches]
                )
    if np.any(positions < 0):
        raise ValueError(f"{name} objective term stable id is absent")
    return positions


def _outward_float64_lower(value: Fraction) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        return -np.inf if value < 0 else float(np.finfo(np.float64).max)
    if rounded == np.inf:
        return float(np.finfo(np.float64).max)
    if rounded == -np.inf:
        return -np.inf
    if Fraction.from_float(rounded) > value:
        rounded = float(np.nextafter(rounded, -np.inf))
    if math.isfinite(rounded) and Fraction.from_float(rounded) > value:
        raise AssertionError("downward binary64 objective rounding failed")
    return rounded


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def certify_preformed_split_primal_lower(
    *,
    objective_envelope,
    expected_parent_semantic_digest,
    expected_objective_id,
    expected_objective_binding_sha256,
    continuous_col_ids,
    binary_col_ids,
    Auc,
    Aub,
    Ac,
    Ab,
    ub,
    b,
    continuous_lb,
    continuous_ub,
    binary_lb,
    binary_ub,
    continuous_candidate,
    binary_candidate,
    deadline,
    caps=None,
):
    """Certify split-frame feasibility and an exact-objective lower bound.

    Successful return is ``(lower, receipt)``.  Every invalid, expired,
    over-cap, infeasible, stale, or forged input returns ``(None, receipt)``.
    ``proof_authority`` on success applies only to the supplied numeric frame;
    ``parent_binding_authority`` and ``verdict_authority`` are always false.
    """

    receipt = {
        "schema": _SCHEMA,
        "status": "not_started",
        "route": "split_csr_primal_exact_fallback_no_stack_v1",
        "proof_authority": False,
        "numeric_frame_authority": False,
        "lower_certificate_authority": False,
        "parent_binding_authority": False,
        "verdict_authority": False,
        "pcoh_authority": False,
        "solver_status_authority": False,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "assembled_sparse_nnz": 0,
        "input_sparse_nnz": None,
        "candidate_snapshot_bytes": None,
        "candidate_snapshot_sha256": None,
        "numeric_frame_sha256_pre": None,
        "numeric_frame_sha256_post": None,
        "numeric_frame_unchanged": False,
        "stable_ids_sha256": None,
        "stable_ids_cross_bound": False,
        "objective_binding_cross_bound": False,
        "objective_term_position_strategy": (
            "term_capped_searchsorted_scan_no_full_python_id_map_v1"
        ),
        "stable_id_sort_copy_bytes": None,
        "upper_rows_interval_verified": 0,
        "upper_rows_exact_replayed": 0,
        "upper_exact_replay_nnz": 0,
        "equality_rows_exact_replayed": 0,
        "equality_exact_replay_nnz": 0,
        "maximum_row_roundoff_guard": None,
        "bounds_exact_binary64_comparison": False,
        "binary_relaxation_bounds_only": True,
        "binary_integrality_required": False,
        "objective_exact_fraction": None,
        "lower": None,
        "lower_float64_rounding": "toward_negative_infinity_v1",
        "frame_trust_boundary": (
            "strict_readonly_identity_recheck_and_pre_post_sha_"
            "parent_semantic_binding_absent_v1"
        ),
        "proof_authority_scope": "supplied_numeric_frame_only",
        "numeric_authority_requires_strict_readonly_inputs": True,
        "authority_input_identity_rechecked": False,
        "authority_input_readonly_rechecked": False,
        "hostile_concurrent_aba_resistance": False,
        "trusted_no_concurrent_mutation_required": True,
        "one_use_owner_integration_required_for_parent_authority": True,
        "analytical_temporary_workspace_bytes_ceiling": None,
        "analytical_workspace_excludes_borrowed_inputs_and_envelope": True,
    }
    try:
        configured = _caps(caps)
        receipt["caps"] = asdict(configured)
        if (
            deadline is not None
            and (
                isinstance(deadline, (bool, np.bool_))
                or not np.isscalar(deadline)
                or not np.isfinite(float(deadline))
            )
        ):
            raise ValueError("deadline must be None or one finite scalar")
        deadline = None if deadline is None else float(deadline)
        _deadline(deadline, "entry")

        parent = _canonical_sha256(
            expected_parent_semantic_digest,
            name="expected_parent_semantic_digest",
        )
        expected_binding = _canonical_sha256(
            expected_objective_binding_sha256,
            name="expected_objective_binding_sha256",
        )
        if type(expected_objective_id) is not str:
            raise ValueError("expected_objective_id must be a strict string")

        # This private accessor is the live process-local registry and
        # identity-seal check.  Rebuild the public core binding independently
        # before using any exact term as authority.
        center, continuous_terms, binary_terms, sealed_binding = (
            _hz_read_exact_objective_binding_material_from_factor_envelope(
                objective_envelope,
                expected_parent_semantic_digest=parent,
                expected_objective_id=expected_objective_id,
            )
        )
        binding = build_objective_binding(
            objective_id=expected_objective_id,
            parent_semantic_digest=parent,
            center=center,
            continuous_terms=continuous_terms,
            binary_terms=binary_terms,
        )
        if (
            not verify_objective_binding(binding)
            or binding.objective_binding_sha256 != sealed_binding
            or sealed_binding != expected_binding
            or getattr(objective_envelope, "objective_binding_sha256", None)
            != expected_binding
        ):
            raise PermissionError("exact objective binding cross-check failed")
        receipt["objective_binding_cross_bound"] = True
        receipt["objective_binding_sha256"] = expected_binding
        receipt["objective_envelope_sha256"] = getattr(
            objective_envelope, "envelope_sha256", None
        )

        upper_rhs = _dense_f64(
            ub, size=None, name="ub", require_readonly=True
        )
        equality_rhs = _dense_f64(
            b, size=None, name="b", require_readonly=True
        )
        continuous_lower = _dense_f64(
            continuous_lb,
            size=None,
            name="continuous_lb",
            require_readonly=True,
        )
        continuous_upper = _dense_f64(
            continuous_ub,
            size=int(continuous_lower.size),
            name="continuous_ub",
            require_readonly=True,
        )
        binary_lower = _dense_f64(
            binary_lb,
            size=None,
            name="binary_lb",
            require_readonly=True,
        )
        binary_upper = _dense_f64(
            binary_ub,
            size=int(binary_lower.size),
            name="binary_ub",
            require_readonly=True,
        )
        n_continuous = int(continuous_lower.size)
        n_binary = int(binary_lower.size)
        n_upper = int(upper_rhs.size)
        n_equality = int(equality_rhs.size)
        if n_continuous + n_binary > configured.max_columns:
            raise _CapExceeded("column cap exceeded")
        if n_upper + n_equality > configured.max_rows:
            raise _CapExceeded("row cap exceeded")

        continuous_ids = _stable_ids(
            continuous_col_ids,
            size=n_continuous,
            name="continuous_col_ids",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        binary_ids = _stable_ids(
            binary_col_ids,
            size=n_binary,
            name="binary_col_ids",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        continuous_sorted, continuous_copied = _strictly_sorted_unique(
            continuous_ids,
            name="continuous_col_ids",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        binary_sorted, binary_copied = _strictly_sorted_unique(
            binary_ids,
            name="binary_col_ids",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        receipt["stable_id_sort_copy_bytes"] = int(
            (continuous_sorted.nbytes if continuous_copied else 0)
            + (binary_sorted.nbytes if binary_copied else 0)
        )
        _reject_stable_id_intersection(
            continuous_sorted,
            binary_sorted,
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        del continuous_sorted, binary_sorted
        stable_sha = _stable_ids_sha256(
            continuous_ids,
            binary_ids,
            deadline=deadline,
            chunk_bytes=8 * configured.chunk_elements,
        )
        if stable_sha != getattr(objective_envelope, "stable_ids_sha256", None):
            raise PermissionError("stable ids do not match the sealed envelope")
        receipt["stable_ids_sha256"] = stable_sha
        receipt["stable_ids_cross_bound"] = True

        matrices = (
            (
                "Auc",
                _csr(
                    Auc,
                    rows=n_upper,
                    columns=n_continuous,
                    name="Auc",
                    deadline=deadline,
                    chunk=configured.chunk_elements,
                ),
            ),
            (
                "Aub",
                _csr(
                    Aub,
                    rows=n_upper,
                    columns=n_binary,
                    name="Aub",
                    deadline=deadline,
                    chunk=configured.chunk_elements,
                ),
            ),
            (
                "Ac",
                _csr(
                    Ac,
                    rows=n_equality,
                    columns=n_continuous,
                    name="Ac",
                    deadline=deadline,
                    chunk=configured.chunk_elements,
                ),
            ),
            (
                "Ab",
                _csr(
                    Ab,
                    rows=n_equality,
                    columns=n_binary,
                    name="Ab",
                    deadline=deadline,
                    chunk=configured.chunk_elements,
                ),
            ),
        )
        Auc_live, Aub_live, Ac_live, Ab_live = (
            item[1] for item in matrices
        )
        input_nnz = sum(int(matrix.nnz) for _, matrix in matrices)
        if input_nnz > configured.max_constraint_nnz:
            raise _CapExceeded("constraint nonzero cap exceeded")
        receipt["input_sparse_nnz"] = int(input_nnz)
        receipt["block_shapes"] = {
            name: [int(value) for value in matrix.shape]
            for name, matrix in matrices
        }

        dense_frame = (
            ("ub", upper_rhs),
            ("b", equality_rhs),
            ("continuous_lb", continuous_lower),
            ("continuous_ub", continuous_upper),
            ("binary_lb", binary_lower),
            ("binary_ub", binary_upper),
            ("continuous_col_ids", continuous_ids),
            ("binary_col_ids", binary_ids),
        )
        authority_identity_records = _authority_input_identity_records(
            matrices=matrices,
            arrays=dense_frame,
        )
        receipt["authority_input_identity_count"] = len(
            authority_identity_records
        )
        frame_pre = _frame_sha256(
            matrices=matrices,
            arrays=dense_frame,
            deadline=deadline,
            chunk_bytes=8 * configured.chunk_elements,
        )
        receipt["numeric_frame_sha256_pre"] = frame_pre

        continuous_point, continuous_point_sha, continuous_snapshot_bytes = (
            _candidate_snapshot(
                continuous_candidate,
                size=n_continuous,
                name="continuous_candidate",
                deadline=deadline,
                chunk=configured.chunk_elements,
            )
        )
        binary_point, binary_point_sha, binary_snapshot_bytes = (
            _candidate_snapshot(
                binary_candidate,
                size=n_binary,
                name="binary_candidate",
                deadline=deadline,
                chunk=configured.chunk_elements,
            )
        )
        receipt["candidate_snapshot_bytes"] = int(
            continuous_snapshot_bytes + binary_snapshot_bytes
        )
        receipt["candidate_snapshot_sha256"] = hashlib.sha256(
            (continuous_point_sha + binary_point_sha).encode("ascii")
        ).hexdigest()

        _validate_bounds(
            candidate=continuous_point,
            lower=continuous_lower,
            upper=continuous_upper,
            name="continuous",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        _validate_bounds(
            candidate=binary_point,
            lower=binary_lower,
            upper=binary_upper,
            name="binary",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        receipt["bounds_exact_binary64_comparison"] = True

        if n_equality > configured.max_exact_equality_rows:
            raise _CapExceeded("exact equality row cap exceeded")
        equality_nnz = int(Ac_live.nnz + Ab_live.nnz)
        if equality_nnz > configured.max_exact_equality_nnz:
            raise _CapExceeded("exact equality nonzero cap exceeded")
        for row in range(n_equality):
            if row % 64 == 0:
                _deadline(deadline, "equality_rows")
            exact = _exact_row_value(
                row=row,
                continuous_matrix=Ac_live,
                binary_matrix=Ab_live,
                continuous_candidate=continuous_point,
                binary_candidate=binary_point,
                deadline=deadline,
            )
            if exact != Fraction.from_float(float(equality_rhs[row])):
                receipt["status"] = "infeasible:equality_violation"
                receipt["violating_row"] = int(row)
                return None, receipt
            receipt["equality_rows_exact_replayed"] += 1
        receipt["equality_exact_replay_nnz"] = equality_nnz

        platform_ok, platform_reason = _hz_longdouble_certificate_platform()
        receipt["platform"] = platform_reason
        if not platform_ok:
            raise ValueError("long-double interval platform unsupported")
        maximum_guard = np.longdouble(0)
        exact_upper_rows = 0
        exact_upper_nnz = 0
        for row in range(n_upper):
            if row % 64 == 0:
                _deadline(deadline, "upper_rows")
            lower_residual, upper_residual, guard = _longdouble_row_interval(
                row=row,
                continuous_matrix=Auc_live,
                binary_matrix=Aub_live,
                continuous_candidate=continuous_point,
                binary_candidate=binary_point,
                rhs=float(upper_rhs[row]),
                deadline=deadline,
                chunk=configured.chunk_elements,
            )
            maximum_guard = max(maximum_guard, guard)
            if upper_residual < 0:
                receipt["upper_rows_interval_verified"] += 1
                continue
            if lower_residual > 0:
                receipt["status"] = "infeasible:upper_interval_violation"
                receipt["violating_row"] = int(row)
                receipt["violating_residual_lower"] = float(lower_residual)
                return None, receipt
            row_nnz = int(
                Auc_live.indptr[row + 1]
                - Auc_live.indptr[row]
                + Aub_live.indptr[row + 1]
                - Aub_live.indptr[row]
            )
            exact_upper_rows += 1
            exact_upper_nnz += row_nnz
            if exact_upper_rows > configured.max_exact_upper_rows:
                raise _CapExceeded("ambiguous exact upper row cap exceeded")
            if exact_upper_nnz > configured.max_exact_upper_nnz:
                raise _CapExceeded("ambiguous exact upper nonzero cap exceeded")
            exact = _exact_row_value(
                row=row,
                continuous_matrix=Auc_live,
                binary_matrix=Aub_live,
                continuous_candidate=continuous_point,
                binary_candidate=binary_point,
                deadline=deadline,
            )
            if exact > Fraction.from_float(float(upper_rhs[row])):
                receipt["status"] = "infeasible:upper_exact_violation"
                receipt["violating_row"] = int(row)
                return None, receipt
            receipt["upper_rows_exact_replayed"] += 1
        receipt["upper_exact_replay_nnz"] = int(exact_upper_nnz)
        receipt["maximum_row_roundoff_guard"] = float(maximum_guard)

        objective_terms = len(continuous_terms) + len(binary_terms)
        if objective_terms > configured.max_exact_objective_terms:
            raise _CapExceeded("exact objective term cap exceeded")
        receipt["analytical_temporary_workspace_bytes_ceiling"] = int(
            8 * (n_continuous + n_binary)
            + 16 * objective_terms
            + 96 * configured.chunk_elements
            + 2_097_152
        )
        continuous_positions = _term_positions(
            continuous_terms,
            continuous_ids,
            name="continuous",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        binary_positions = _term_positions(
            binary_terms,
            binary_ids,
            name="binary",
            deadline=deadline,
            chunk=configured.chunk_elements,
        )
        exact_objective = center
        for offset, ((_, coefficient), position) in enumerate(
            zip(continuous_terms, continuous_positions)
        ):
            if offset % 4096 == 0:
                _deadline(deadline, "continuous_objective")
            exact_objective += coefficient * Fraction.from_float(
                float(continuous_point[int(position)])
            )
        for offset, ((_, coefficient), position) in enumerate(
            zip(binary_terms, binary_positions)
        ):
            if offset % 4096 == 0:
                _deadline(deadline, "binary_objective")
            exact_objective += coefficient * Fraction.from_float(
                float(binary_point[int(position)])
            )
        lower = _outward_float64_lower(exact_objective)

        frame_post = _frame_sha256(
            matrices=matrices,
            arrays=dense_frame,
            deadline=deadline,
            chunk_bytes=8 * configured.chunk_elements,
        )
        receipt["numeric_frame_sha256_post"] = frame_post
        if frame_post != frame_pre:
            raise PermissionError("numeric frame changed during certification")
        # Stable ids are part of frame_sha, but retain the solver-envelope
        # domain-separated replay as a separately auditable cross-binding.
        stable_sha_post = _stable_ids_sha256(
            continuous_ids,
            binary_ids,
            deadline=deadline,
            chunk_bytes=8 * configured.chunk_elements,
        )
        if stable_sha_post != stable_sha:
            raise PermissionError("stable ids changed during certification")
        _recheck_authority_input_identity_and_readonly(
            authority_identity_records
        )
        receipt["authority_input_identity_rechecked"] = True
        receipt["authority_input_readonly_rechecked"] = True
        _deadline(deadline, "before_authorization")

        receipt.update(
            {
                "status": "verified_numeric_frame_primal_lower",
                "proof_authority": True,
                "numeric_frame_authority": True,
                "lower_certificate_authority": True,
                "numeric_frame_unchanged": True,
                "objective_exact_fraction": _rational_text(exact_objective),
                "objective_term_count": int(objective_terms),
                "lower": lower,
                "parent_binding_authority": False,
                "verdict_authority": False,
                "pcoh_authority": False,
                "solver_status_authority": False,
            }
        )
        return lower, receipt
    except _Deadline as exc:
        receipt["status"] = f"deadline_exhausted:{str(exc)[:120]}"
        return None, receipt
    except _CapExceeded as exc:
        receipt["status"] = f"cap_exceeded:{str(exc)[:120]}"
        return None, receipt
    except Exception as exc:
        receipt["status"] = f"invalid:{type(exc).__name__}:{str(exc)[:120]}"
        return None, receipt


__all__ = [
    "DEFAULT_PREFORMED_SPLIT_PRIMAL_CAPS",
    "PreformedSplitPrimalCertificateCaps",
    "certify_preformed_split_primal_lower",
]
