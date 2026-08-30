"""Toy-first strict optimum ordering for two supplied split LP frames.

The numeric theorem is intentionally small and explicit.  An independently
checked feasible parent point gives ``L_parent <= OPT_parent``.  An
independently replayed fresh dual gives ``OPT_fresh <= U_fresh``.  Therefore
the exact comparison ``U_fresh < L_parent`` proves
``OPT_fresh < OPT_parent`` for the two supplied numeric frames.

This module does *not* prove that the fresh constraints are a sound extension
of a live parent HybridZ object.  It also does not trust a candidate solver's
status, objective, primal-feasibility telemetry, or upper bound.  Candidate
receipts are cross-bound for diagnostics and native-close hygiene only; the
public candidate dataclass has no private provenance seal.  All verifier,
property, PCOH, ground-truth, and live-parent authority remains false.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import struct
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    build_objective_binding,
    verify_objective_binding,
)
from act.back_end.hybridz_tf.preformed_split_primal_certificate import (
    PreformedSplitPrimalCertificateCaps,
    _authority_input_identity_records,
    _csr,
    _dense_f64,
    _frame_sha256,
    _recheck_authority_input_identity_and_readonly,
    _reject_stable_id_intersection,
    _stable_ids,
    _stable_ids_sha256,
    _strictly_sorted_unique,
    certify_preformed_split_primal_lower,
)
from act.back_end.hybridz_tf.split_constraint_generation_candidate import (
    SplitConstraintGenerationCandidate,
    _array_sha256 as _candidate_array_sha256,
    _canonical_json_sha256 as _candidate_json_sha256,
    _validate_csr as _candidate_validate_csr,
)
from act.back_end.solver.solver_hz import (
    _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope,
    _hz_read_exact_objective_binding_material_from_factor_envelope,
    _hz_validate_preformed_factor_objective_envelope,
)


_SCHEMA = "act.hybridz.strict_supplied_split_lp_ordering.v1"
_DESCRIPTOR_SCHEMA = (
    "act.hybridz.strict_supplied_split_lp_ordering_descriptor.v1"
)
_CHUNK = 65_536


class _Deadline(TimeoutError):
    pass


class _CertificateError(ValueError):
    pass


@dataclass(frozen=True)
class SplitRelaxedLPFrame:
    """Native continuous/binary, upper/equality split constraint blocks."""

    Auc: sp.csr_matrix
    Aub: sp.csr_matrix
    Ac: sp.csr_matrix
    Ab: sp.csr_matrix
    ub: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class PreformedSplitLPProblem:
    """One supplied numeric LP frame plus its sealed exact objective."""

    objective_envelope: Any
    expected_parent_semantic_digest: str
    expected_exact_objective_sha256: str
    expected_objective_binding_sha256: str
    continuous_col_ids: np.ndarray
    binary_col_ids: np.ndarray
    continuous_lb: np.ndarray
    continuous_ub: np.ndarray
    binary_lb: np.ndarray
    binary_ub: np.ndarray
    frame: SplitRelaxedLPFrame


@dataclass(frozen=True)
class StrictSuppliedSplitLPOrderingCertificate:
    """A numeric-frame-only strict ordering descriptor."""

    schema: str
    parent_lower: float
    fresh_upper: float
    exact_gap: Fraction
    parent_frame_sha256: str
    fresh_frame_sha256: str
    parent_objective_envelope_sha256: str
    fresh_objective_envelope_sha256: str
    strict_relaxed_lp_improvement_certified: bool
    strict_relaxed_lp_improvement_scope: str
    strict_supplied_frame_optimum_ordering_certified: bool
    numeric_frame_authority: bool
    proof_authority: bool
    verdict_authority: bool
    parent_binding_authority: bool
    sound_tightening_improvement_authority: bool
    hostile_concurrent_aba_resistance: bool
    trusted_no_concurrent_mutation_required: bool
    one_use_live_owner_required_for_sound_tightening: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedProblem:
    problem: PreformedSplitLPProblem
    objective_envelope: Any
    expected_parent_semantic_digest: str
    expected_exact_objective_sha256: str
    expected_objective_binding_sha256: str
    matrices: Tuple[Tuple[str, sp.csr_matrix], ...]
    dense_frame: Tuple[Tuple[str, np.ndarray], ...]
    identity_records: Tuple[Tuple[Any, ...], ...]
    frame_sha256: str
    continuous_ids: np.ndarray
    binary_ids: np.ndarray
    continuous_lower: np.ndarray
    continuous_upper: np.ndarray
    binary_lower: np.ndarray
    binary_upper: np.ndarray
    upper_rhs: np.ndarray
    equality_rhs: np.ndarray
    q_continuous: np.ndarray
    q_binary: np.ndarray
    exact_center: Fraction
    continuous_terms: Tuple[Tuple[int, Fraction], ...]
    binary_terms: Tuple[Tuple[int, Fraction], ...]
    sealed_fields: Mapping[str, Any]


def _deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise _Deadline(stage)


def _sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _CertificateError(f"{name} is not canonical SHA-256")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _CertificateError("authority payload is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _validate_deadline(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise _CertificateError("deadline must be finite monotonic time")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _CertificateError("deadline must be finite monotonic time") from exc
    if not math.isfinite(result):
        raise _CertificateError("deadline must be finite monotonic time")
    return result


def _validate_finite_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    name: str,
    deadline: float,
) -> None:
    for start in range(0, int(lower.size), _CHUNK):
        _deadline(deadline, f"validate_{name}_bounds")
        local_lower = lower[start : start + _CHUNK]
        local_upper = upper[start : start + _CHUNK]
        if (
            not np.all(np.isfinite(local_lower))
            or not np.all(np.isfinite(local_upper))
            or np.any(local_lower > local_upper)
        ):
            raise _CertificateError(f"{name} bounds are invalid")


def _validate_rhs(
    values: np.ndarray,
    *,
    name: str,
    deadline: float,
) -> None:
    for start in range(0, int(values.size), _CHUNK):
        _deadline(deadline, f"validate_{name}")
        if not np.all(np.isfinite(values[start : start + _CHUNK])):
            raise _CertificateError(f"{name} contains non-finite data")


def _prepare_problem(
    problem: Any,
    *,
    expected_objective_id: str,
    label: str,
    deadline: float,
) -> _PreparedProblem:
    if type(problem) is not PreformedSplitLPProblem:
        raise _CertificateError(f"{label} problem has the wrong type")
    if type(problem.frame) is not SplitRelaxedLPFrame:
        raise _CertificateError(f"{label} frame has the wrong type")
    parent_digest = _sha256(
        problem.expected_parent_semantic_digest,
        name=f"{label}_parent_semantic_digest",
    )
    exact_digest = _sha256(
        problem.expected_exact_objective_sha256,
        name=f"{label}_exact_objective_sha256",
    )
    binding_digest = _sha256(
        problem.expected_objective_binding_sha256,
        name=f"{label}_objective_binding_sha256",
    )
    validated = _hz_validate_preformed_factor_objective_envelope(
        problem.objective_envelope,
        expected_parent_semantic_digest=parent_digest,
        expected_exact_objective_sha256=exact_digest,
        expected_objective_binding_sha256=binding_digest,
    )
    sealed = validated[-1]
    n_continuous = int(sealed["_n_continuous"])
    n_binary = int(sealed["_n_binary"])
    continuous_ids = _stable_ids(
        problem.continuous_col_ids,
        size=n_continuous,
        name=f"{label}_continuous_col_ids",
        deadline=deadline,
        chunk=_CHUNK,
    )
    binary_ids = _stable_ids(
        problem.binary_col_ids,
        size=n_binary,
        name=f"{label}_binary_col_ids",
        deadline=deadline,
        chunk=_CHUNK,
    )
    continuous_sorted, _ = _strictly_sorted_unique(
        continuous_ids,
        name=f"{label}_continuous_col_ids",
        deadline=deadline,
        chunk=_CHUNK,
    )
    binary_sorted, _ = _strictly_sorted_unique(
        binary_ids,
        name=f"{label}_binary_col_ids",
        deadline=deadline,
        chunk=_CHUNK,
    )
    _reject_stable_id_intersection(
        continuous_sorted,
        binary_sorted,
        deadline=deadline,
        chunk=_CHUNK,
    )
    stable_digest = _stable_ids_sha256(
        continuous_ids,
        binary_ids,
        deadline=deadline,
        chunk_bytes=8 * _CHUNK,
    )
    if stable_digest != sealed["_stable_ids_sha256"]:
        raise _CertificateError(f"{label} stable ids do not match envelope")

    continuous_lower = _dense_f64(
        problem.continuous_lb,
        size=n_continuous,
        name=f"{label}_continuous_lb",
        require_readonly=True,
    )
    continuous_upper = _dense_f64(
        problem.continuous_ub,
        size=n_continuous,
        name=f"{label}_continuous_ub",
        require_readonly=True,
    )
    binary_lower = _dense_f64(
        problem.binary_lb,
        size=n_binary,
        name=f"{label}_binary_lb",
        require_readonly=True,
    )
    binary_upper = _dense_f64(
        problem.binary_ub,
        size=n_binary,
        name=f"{label}_binary_ub",
        require_readonly=True,
    )
    _validate_finite_bounds(
        continuous_lower,
        continuous_upper,
        name=f"{label}_continuous",
        deadline=deadline,
    )
    _validate_finite_bounds(
        binary_lower,
        binary_upper,
        name=f"{label}_binary",
        deadline=deadline,
    )

    upper_rhs = _dense_f64(
        problem.frame.ub,
        size=None,
        name=f"{label}_ub",
        require_readonly=True,
    )
    equality_rhs = _dense_f64(
        problem.frame.b,
        size=None,
        name=f"{label}_b",
        require_readonly=True,
    )
    _validate_rhs(upper_rhs, name=f"{label}_ub", deadline=deadline)
    _validate_rhs(equality_rhs, name=f"{label}_b", deadline=deadline)
    n_upper = int(upper_rhs.size)
    n_equality = int(equality_rhs.size)
    matrices = (
        (
            "Auc",
            _csr(
                problem.frame.Auc,
                rows=n_upper,
                columns=n_continuous,
                name=f"{label}_Auc",
                deadline=deadline,
                chunk=_CHUNK,
            ),
        ),
        (
            "Aub",
            _csr(
                problem.frame.Aub,
                rows=n_upper,
                columns=n_binary,
                name=f"{label}_Aub",
                deadline=deadline,
                chunk=_CHUNK,
            ),
        ),
        (
            "Ac",
            _csr(
                problem.frame.Ac,
                rows=n_equality,
                columns=n_continuous,
                name=f"{label}_Ac",
                deadline=deadline,
                chunk=_CHUNK,
            ),
        ),
        (
            "Ab",
            _csr(
                problem.frame.Ab,
                rows=n_equality,
                columns=n_binary,
                name=f"{label}_Ab",
                deadline=deadline,
                chunk=_CHUNK,
            ),
        ),
    )
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
    frame_sha = _frame_sha256(
        matrices=matrices,
        arrays=dense_frame,
        deadline=deadline,
        chunk_bytes=8 * _CHUNK,
    )
    identity_records = _authority_input_identity_records(
        matrices=matrices,
        arrays=dense_frame,
    )

    center, continuous_terms, binary_terms, accessor_binding = (
        _hz_read_exact_objective_binding_material_from_factor_envelope(
            problem.objective_envelope,
            expected_parent_semantic_digest=parent_digest,
            expected_objective_id=expected_objective_id,
        )
    )
    rebuilt = build_objective_binding(
        objective_id=expected_objective_id,
        parent_semantic_digest=parent_digest,
        center=center,
        continuous_terms=continuous_terms,
        binary_terms=binary_terms,
    )
    if (
        not verify_objective_binding(rebuilt)
        or rebuilt.objective_binding_sha256 != binding_digest
        or accessor_binding != binding_digest
    ):
        raise _CertificateError(f"{label} exact objective binding mismatch")
    _deadline(deadline, f"prepare_{label}_problem")
    return _PreparedProblem(
        problem=problem,
        objective_envelope=problem.objective_envelope,
        expected_parent_semantic_digest=parent_digest,
        expected_exact_objective_sha256=exact_digest,
        expected_objective_binding_sha256=binding_digest,
        matrices=matrices,
        dense_frame=dense_frame,
        identity_records=identity_records,
        frame_sha256=frame_sha,
        continuous_ids=continuous_ids,
        binary_ids=binary_ids,
        continuous_lower=continuous_lower,
        continuous_upper=continuous_upper,
        binary_lower=binary_lower,
        binary_upper=binary_upper,
        upper_rhs=upper_rhs,
        equality_rhs=equality_rhs,
        q_continuous=validated[0],
        q_binary=validated[2],
        exact_center=center,
        continuous_terms=continuous_terms,
        binary_terms=binary_terms,
        sealed_fields=sealed,
    )


def _sorted_copy(values: np.ndarray) -> np.ndarray:
    if values.size < 2 or np.all(values[1:] > values[:-1]):
        return values
    return np.sort(values, kind="quicksort")


def _require_subset(
    parent: np.ndarray,
    fresh: np.ndarray,
    *,
    name: str,
    deadline: float,
) -> int:
    parent_sorted = _sorted_copy(parent)
    fresh_sorted = _sorted_copy(fresh)
    for start in range(0, int(parent_sorted.size), _CHUNK):
        _deadline(deadline, f"objective_extension_{name}")
        local = parent_sorted[start : start + _CHUNK]
        positions = np.searchsorted(fresh_sorted, local)
        if np.any(positions >= fresh_sorted.size):
            raise _CertificateError(f"parent {name} ids are not a fresh subset")
        if np.any(fresh_sorted[positions] != local):
            raise _CertificateError(f"parent {name} ids are not a fresh subset")
    return int(fresh.size - parent.size)


def _verify_objective_extension(
    parent: _PreparedProblem,
    fresh: _PreparedProblem,
    *,
    deadline: float,
) -> Tuple[int, int]:
    if (
        parent.exact_center != fresh.exact_center
        or parent.continuous_terms != fresh.continuous_terms
        or parent.binary_terms != fresh.binary_terms
    ):
        raise _CertificateError(
            "parent and fresh exact objective semantics differ"
        )
    extra_continuous = _require_subset(
        parent.continuous_ids,
        fresh.continuous_ids,
        name="continuous",
        deadline=deadline,
    )
    extra_binary = _require_subset(
        parent.binary_ids,
        fresh.binary_ids,
        name="binary",
        deadline=deadline,
    )
    return extra_continuous, extra_binary


def _split_array_sha256(
    parts: Tuple[np.ndarray, ...],
    *,
    tag: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"act.ndarray.raw.v1\0")
    digest.update(tag.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.dtype(np.float64).str.encode("ascii"))
    digest.update(struct.pack(">Q", sum(int(part.size) for part in parts)))
    for part in parts:
        if (
            type(part) is not np.ndarray
            or part.dtype != np.dtype(np.float64)
            or part.ndim != 1
            or not part.flags.c_contiguous
        ):
            raise _CertificateError(f"{tag} split hash input is noncanonical")
        view = memoryview(part).cast("B")
        for start in range(0, len(view), 1 << 20):
            digest.update(view[start : start + (1 << 20)])
    return digest.hexdigest()


def _candidate_array(
    values: Any,
    *,
    size: int,
    name: str,
    deadline: float,
) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.dtype(np.float64)
        or values.ndim != 1
        or int(values.size) != int(size)
        or not values.flags.c_contiguous
        or not values.flags.aligned
        or values.flags.writeable
    ):
        raise _CertificateError(f"{name} candidate array is not canonical readonly")
    for start in range(0, int(values.size), _CHUNK):
        _deadline(deadline, f"validate_{name}")
        if not np.all(np.isfinite(values[start : start + _CHUNK])):
            raise _CertificateError(f"{name} candidate array is non-finite")
    return values


def _validate_candidate_receipt(
    candidate: Any,
    prepared: _PreparedProblem,
    *,
    label: str,
    deadline: float,
) -> Tuple[str, Tuple[Any, ...]]:
    if type(candidate) is not SplitConstraintGenerationCandidate:
        raise _CertificateError(f"{label} candidate has the wrong type")
    if (
        candidate.proof_authority is not False
        or candidate.verdict_authority is not False
        or candidate.primal_feasibility_authority is not False
    ):
        raise _CertificateError(f"{label} candidate authority flags are invalid")
    n_continuous = int(prepared.continuous_ids.size)
    n_binary = int(prepared.binary_ids.size)
    n_upper = int(prepared.upper_rhs.size)
    n_equality = int(prepared.equality_rhs.size)
    factor = _candidate_array(
        candidate.factor_primal,
        size=n_continuous + n_binary,
        name=f"{label}_factor_primal",
        deadline=deadline,
    )
    upper_dual = _candidate_array(
        candidate.upper_row_dual,
        size=n_upper,
        name=f"{label}_upper_dual",
        deadline=deadline,
    )
    equality_dual = _candidate_array(
        candidate.equality_row_dual,
        size=n_equality,
        name=f"{label}_equality_dual",
        deadline=deadline,
    )
    if not isinstance(candidate.receipt, Mapping):
        raise _CertificateError(f"{label} candidate receipt is missing")
    receipt = candidate.receipt
    claimed_receipt_sha = _sha256(
        receipt.get("receipt_sha256"),
        name=f"{label}_candidate_receipt_sha256",
    )
    raw_receipt = dict(receipt)
    raw_receipt.pop("receipt_sha256", None)
    if _candidate_json_sha256(raw_receipt) != claimed_receipt_sha:
        raise _CertificateError(f"{label} candidate receipt hash mismatch")
    if (
        receipt.get("schema")
        != "act.hybridz.split_constraint_generation_candidate.v1"
        or receipt.get("candidate_only") is not True
        or receipt.get("proof_authority") is not False
        or receipt.get("verdict_authority") is not False
        or receipt.get("primal_feasibility_authority") is not False
        or receipt.get("parent_binding_authority") is not False
        or receipt.get("native_model_closed_before_return") is not True
        or receipt.get("uses_sparse_hstack") is not False
        or receipt.get("uses_sparse_vstack") is not False
        or receipt.get("uses_dense_hstack") is not False
        or receipt.get("uses_dense_vstack") is not False
        or receipt.get("used_merged_sparse_frame") is not False
        or receipt.get("materialized_full_candidate_csr") is not False
        or receipt.get("n_continuous") != n_continuous
        or receipt.get("n_binary_relaxed") != n_binary
        or receipt.get("n_upper") != n_upper
        or receipt.get("n_equality") != n_equality
    ):
        raise _CertificateError(f"{label} candidate receipt contract mismatch")
    if (
        receipt.get("factor_primal_sha256")
        != _candidate_array_sha256(factor, tag="factor_primal_candidate")
        or receipt.get("upper_row_dual_sha256")
        != _candidate_array_sha256(
            upper_dual, tag="upper_row_dual_raw_minimization"
        )
        or receipt.get("equality_row_dual_sha256")
        != _candidate_array_sha256(
            equality_dual,
            tag="equality_row_dual_raw_minimization",
        )
    ):
        raise _CertificateError(f"{label} candidate array hash mismatch")

    source = receipt.get("provided_split_frame_binding")
    if not isinstance(source, Mapping):
        raise _CertificateError(f"{label} candidate frame binding is missing")
    if (
        source.get("schema")
        != "act.hybridz.provided_split_frame_binding.v1"
        or source.get("n_continuous") != n_continuous
        or source.get("n_binary_relaxed") != n_binary
        or source.get("n_upper") != n_upper
        or source.get("n_equality") != n_equality
    ):
        raise _CertificateError(f"{label} candidate frame dimensions mismatch")
    block_binding = source.get("blocks")
    if not isinstance(block_binding, Mapping) or set(block_binding) != {
        "Auc",
        "Aub",
        "Ac",
        "Ab",
    }:
        raise _CertificateError(f"{label} candidate block binding is malformed")
    for name, matrix in prepared.matrices:
        view = _candidate_validate_csr(
            matrix,
            name=name,
            rows=int(matrix.shape[0]),
            columns=int(matrix.shape[1]),
            deadline=deadline,
        )
        expected = {
            "shape": [view.rows, view.columns],
            "nnz": view.nnz,
            "sha256": view.sha256,
        }
        if block_binding.get(name) != expected:
            raise _CertificateError(f"{label} candidate {name} binding mismatch")
    dense = source.get("dense_sha256")
    if not isinstance(dense, Mapping) or set(dense) != {
        "ub",
        "b",
        "q",
        "lower_bounds",
        "upper_bounds",
        "seed_upper_duals",
        "seed_equality_duals",
    }:
        raise _CertificateError(f"{label} candidate dense binding is malformed")
    expected_dense = {
        "ub": _candidate_array_sha256(prepared.upper_rhs, tag="ub"),
        "b": _candidate_array_sha256(prepared.equality_rhs, tag="b"),
        "q": _split_array_sha256(
            (prepared.q_continuous, prepared.q_binary), tag="q"
        ),
        "lower_bounds": _split_array_sha256(
            (prepared.continuous_lower, prepared.binary_lower),
            tag="lower_bounds",
        ),
        "upper_bounds": _split_array_sha256(
            (prepared.continuous_upper, prepared.binary_upper),
            tag="upper_bounds",
        ),
    }
    if any(dense.get(name) != value for name, value in expected_dense.items()):
        raise _CertificateError(f"{label} candidate dense frame mismatch")
    _sha256(
        dense.get("seed_upper_duals"),
        name=f"{label}_seed_upper_duals_sha256",
    )
    _sha256(
        dense.get("seed_equality_duals"),
        name=f"{label}_seed_equality_duals_sha256",
    )
    provided_sha = _sha256(
        receipt.get("provided_split_frame_sha256"),
        name=f"{label}_provided_split_frame_sha256",
    )
    if _candidate_json_sha256(source) != provided_sha:
        raise _CertificateError(f"{label} provided frame hash mismatch")
    identity = (
        candidate,
        id(candidate),
        factor,
        id(factor),
        upper_dual,
        id(upper_dual),
        equality_dual,
        id(equality_dual),
        candidate.receipt,
        id(candidate.receipt),
    )
    return claimed_receipt_sha, identity


def _recheck_candidate_identity(identity: Tuple[Any, ...]) -> None:
    (
        candidate,
        candidate_id,
        factor,
        factor_id,
        upper_dual,
        upper_dual_id,
        equality_dual,
        equality_dual_id,
        receipt,
        receipt_id,
    ) = identity
    if (
        id(candidate) != candidate_id
        or candidate.factor_primal is not factor
        or id(candidate.factor_primal) != factor_id
        or candidate.upper_row_dual is not upper_dual
        or id(candidate.upper_row_dual) != upper_dual_id
        or candidate.equality_row_dual is not equality_dual
        or id(candidate.equality_row_dual) != equality_dual_id
        or candidate.receipt is not receipt
        or id(candidate.receipt) != receipt_id
        or factor.flags.writeable
        or upper_dual.flags.writeable
        or equality_dual.flags.writeable
    ):
        raise _CertificateError("candidate identity or readonly contract changed")


def _parent_primal_anchor(
    lower: Any,
    receipt: Mapping[str, Any],
    prepared: _PreparedProblem,
) -> Tuple[float, str]:
    if type(lower) is not float or not math.isfinite(lower):
        raise _CertificateError("parent primal lower is not finite binary64")
    if (
        receipt.get("schema")
        != "act.hybridz_preformed_split_primal_lower.v1"
        or receipt.get("status") != "verified_numeric_frame_primal_lower"
        or receipt.get("proof_authority") is not True
        or receipt.get("numeric_frame_authority") is not True
        or receipt.get("lower_certificate_authority") is not True
        or receipt.get("parent_binding_authority") is not False
        or receipt.get("verdict_authority") is not False
        or receipt.get("lower") != lower
        or receipt.get("numeric_frame_sha256_pre") != prepared.frame_sha256
        or receipt.get("numeric_frame_sha256_post") != prepared.frame_sha256
        or receipt.get("numeric_frame_unchanged") is not True
        or receipt.get("objective_binding_sha256")
        != prepared.expected_objective_binding_sha256
        or receipt.get("stable_ids_sha256")
        != prepared.sealed_fields["_stable_ids_sha256"]
        or receipt.get("authority_input_identity_rechecked") is not True
        or receipt.get("authority_input_readonly_rechecked") is not True
    ):
        raise _CertificateError("parent primal checker receipt is not authoritative")
    payload = {
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
        "lower_hex": lower.hex(),
        "frame_sha256": prepared.frame_sha256,
        "objective_binding_sha256": receipt.get("objective_binding_sha256"),
        "objective_envelope_sha256": receipt.get("objective_envelope_sha256"),
        "stable_ids_sha256": receipt.get("stable_ids_sha256"),
        "candidate_snapshot_sha256": receipt.get("candidate_snapshot_sha256"),
        "numeric_frame_authority": True,
        "parent_binding_authority": False,
        "verdict_authority": False,
    }
    for name in (
        "objective_envelope_sha256",
        "candidate_snapshot_sha256",
    ):
        _sha256(payload[name], name=f"parent_primal_{name}")
    return lower, _canonical_sha256(payload)


def _fresh_dual_anchor(
    upper_longdouble: Any,
    receipt: Mapping[str, Any],
    prepared: _PreparedProblem,
) -> Tuple[float, str]:
    upper = receipt.get("upper")
    if type(upper) is not float or not math.isfinite(upper):
        raise _CertificateError("fresh dual upper is not finite binary64")
    try:
        upper_ld = np.longdouble(upper_longdouble)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _CertificateError("fresh long-double upper is invalid") from exc
    if not np.isfinite(upper_ld) or np.longdouble(upper) < upper_ld:
        raise _CertificateError("fresh binary64 upper is not outward")
    if (
        receipt.get("schema")
        != "hz_lp_lagrangian_preformed_objective_split_blocks_longdouble_v1"
        or receipt.get("status") != "verified_upper"
        or receipt.get("proof_authority") is not True
        or receipt.get("verdict_authority") is not False
        or receipt.get("pcoh_authorization") is not False
        or receipt.get("uses_sparse_hstack") is not False
        or receipt.get("uses_sparse_vstack") is not False
        or receipt.get("assembled_sparse_nnz") != 0
        or receipt.get("parent_semantic_digest")
        != prepared.expected_parent_semantic_digest
        or receipt.get("exact_objective_sha256")
        != prepared.expected_exact_objective_sha256
        or receipt.get("objective_binding_sha256")
        != prepared.expected_objective_binding_sha256
        or receipt.get("objective_envelope_sha256")
        != prepared.sealed_fields["_envelope_sha256"]
        or receipt.get("stable_ids_sha256")
        != prepared.sealed_fields["_stable_ids_sha256"]
    ):
        raise _CertificateError("fresh dual checker receipt is not authoritative")
    payload = {
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
        "upper_hex": upper.hex(),
        "frame_sha256": prepared.frame_sha256,
        "parent_semantic_digest": receipt.get("parent_semantic_digest"),
        "exact_objective_sha256": receipt.get("exact_objective_sha256"),
        "objective_binding_sha256": receipt.get("objective_binding_sha256"),
        "objective_envelope_sha256": receipt.get("objective_envelope_sha256"),
        "stable_ids_sha256": receipt.get("stable_ids_sha256"),
        "proof_authority": True,
        "verdict_authority": False,
        "pcoh_authorization": False,
    }
    return upper, _canonical_sha256(payload)


def certify_strict_preformed_split_lp_improvement(
    *,
    expected_objective_id,
    parent_problem,
    fresh_problem,
    parent_candidate,
    fresh_candidate,
    deadline,
    parent_primal_caps: Optional[PreformedSplitPrimalCertificateCaps] = None,
):
    """Certify strict optimum ordering for two supplied numeric LP frames.

    Returns ``(descriptor, receipt)`` only when the exact comparison of the
    independently authorized binary64 bounds proves ``U_fresh < L_parent``.
    A valid but non-strict comparison returns ``(None, receipt)`` with every
    strict-improvement flag false.  This is not a live-parent tightening proof.
    """

    receipt = {
        "schema": _SCHEMA,
        "status": "not_started",
        "strict_relaxed_lp_improvement_certified": False,
        "strict_relaxed_lp_improvement_scope": (
            "two_supplied_numeric_frames_only"
        ),
        "strict_supplied_frame_optimum_ordering_certified": False,
        "numeric_frame_authority": False,
        "proof_authority": False,
        "verdict_authority": False,
        "ground_truth_authority": False,
        "reference_result_authority": False,
        "pcoh_authority": False,
        "parent_binding_authority": False,
        "constraint_frame_sound_extension_authority": False,
        "sound_tightening_improvement_authority": False,
        "candidate_provenance_authority": False,
        "solver_status_numeric_authority": False,
        "solver_objective_numeric_authority": False,
        "upper_vs_upper_comparison_used": False,
        "comparison": "exact_Fraction(U_fresh_binary64)<Fraction(L_parent_binary64)",
        "objective_extension_equivalent": False,
        "distinct_objective_envelope_identity": False,
        "distinct_stable_id_storage": False,
        "parent_lower": None,
        "fresh_upper": None,
        "exact_gap": None,
        "parent_frame_sha256": None,
        "fresh_frame_sha256": None,
        "parent_candidate_receipt_sha256": None,
        "fresh_candidate_receipt_sha256": None,
        "parent_primal_selected_receipt_sha256": None,
        "fresh_dual_selected_receipt_sha256": None,
        "full_checker_receipts_canonical_hashed": False,
        "nonfinite_diagnostics_copied_into_authority_chain": False,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "used_merged_sparse_frame": False,
        "candidate_native_models_closed_before_orchestration": False,
        "numeric_authority_scope": "two_supplied_readonly_numeric_frames_only",
        "hostile_concurrent_aba_resistance": False,
        "trusted_no_concurrent_mutation_required": True,
        "one_use_live_owner_required_for_sound_tightening": True,
    }
    try:
        deadline = _validate_deadline(deadline)
        _deadline(deadline, "entry")
        if (
            type(expected_objective_id) is not str
            or not expected_objective_id
        ):
            raise _CertificateError("expected_objective_id is invalid")
        parent = _prepare_problem(
            parent_problem,
            expected_objective_id=expected_objective_id,
            label="parent",
            deadline=deadline,
        )
        fresh = _prepare_problem(
            fresh_problem,
            expected_objective_id=expected_objective_id,
            label="fresh",
            deadline=deadline,
        )
        if parent.objective_envelope is fresh.objective_envelope:
            raise _CertificateError(
                "parent and fresh objective envelopes must be distinct"
            )
        for name, parent_ids, fresh_ids in (
            (
                "continuous",
                parent.continuous_ids,
                fresh.continuous_ids,
            ),
            ("binary", parent.binary_ids, fresh.binary_ids),
        ):
            if parent_ids is fresh_ids or (
                parent_ids.size
                and fresh_ids.size
                and np.shares_memory(parent_ids, fresh_ids)
            ):
                raise _CertificateError(
                    f"parent and fresh {name} stable-id storage must be distinct"
                )
        receipt["distinct_objective_envelope_identity"] = True
        receipt["distinct_stable_id_storage"] = True
        extra_continuous, extra_binary = _verify_objective_extension(
            parent, fresh, deadline=deadline
        )
        receipt.update(
            {
                "objective_extension_equivalent": True,
                "fresh_only_zero_objective_continuous_columns": extra_continuous,
                "fresh_only_zero_objective_binary_columns": extra_binary,
                "parent_frame_sha256": parent.frame_sha256,
                "fresh_frame_sha256": fresh.frame_sha256,
                "parent_objective_envelope_sha256": parent.sealed_fields[
                    "_envelope_sha256"
                ],
                "fresh_objective_envelope_sha256": fresh.sealed_fields[
                    "_envelope_sha256"
                ],
            }
        )

        parent_candidate_sha, parent_candidate_identity = (
            _validate_candidate_receipt(
                parent_candidate,
                parent,
                label="parent",
                deadline=deadline,
            )
        )
        fresh_candidate_sha, fresh_candidate_identity = (
            _validate_candidate_receipt(
                fresh_candidate,
                fresh,
                label="fresh",
                deadline=deadline,
            )
        )
        receipt["parent_candidate_receipt_sha256"] = parent_candidate_sha
        receipt["fresh_candidate_receipt_sha256"] = fresh_candidate_sha
        receipt["candidate_native_models_closed_before_orchestration"] = True

        parent_factor = parent_candidate.factor_primal
        n_parent_continuous = int(parent.continuous_ids.size)
        lower, parent_primal_receipt = certify_preformed_split_primal_lower(
            objective_envelope=parent.objective_envelope,
            expected_parent_semantic_digest=(
                parent.expected_parent_semantic_digest
            ),
            expected_objective_id=expected_objective_id,
            expected_objective_binding_sha256=(
                parent.expected_objective_binding_sha256
            ),
            continuous_col_ids=parent.continuous_ids,
            binary_col_ids=parent.binary_ids,
            Auc=parent.matrices[0][1],
            Aub=parent.matrices[1][1],
            Ac=parent.matrices[2][1],
            Ab=parent.matrices[3][1],
            ub=parent.upper_rhs,
            b=parent.equality_rhs,
            continuous_lb=parent.continuous_lower,
            continuous_ub=parent.continuous_upper,
            binary_lb=parent.binary_lower,
            binary_ub=parent.binary_upper,
            continuous_candidate=parent_factor[:n_parent_continuous],
            binary_candidate=parent_factor[n_parent_continuous:],
            deadline=deadline,
            caps=parent_primal_caps,
        )
        lower, parent_anchor = _parent_primal_anchor(
            lower, parent_primal_receipt, parent
        )
        receipt["parent_primal_selected_receipt_sha256"] = parent_anchor

        upper_longdouble, fresh_dual_receipt = (
            _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope(
                objective_envelope=fresh.objective_envelope,
                expected_parent_semantic_digest=(
                    fresh.expected_parent_semantic_digest
                ),
                expected_exact_objective_sha256=(
                    fresh.expected_exact_objective_sha256
                ),
                expected_objective_binding_sha256=(
                    fresh.expected_objective_binding_sha256
                ),
                Auc=fresh.matrices[0][1],
                Aub=fresh.matrices[1][1],
                Ac=fresh.matrices[2][1],
                Ab=fresh.matrices[3][1],
                ub=fresh.upper_rhs,
                b=fresh.equality_rhs,
                continuous_lb=fresh.continuous_lower,
                continuous_ub=fresh.continuous_upper,
                binary_lb=fresh.binary_lower,
                binary_ub=fresh.binary_upper,
                upper_row_dual=fresh_candidate.upper_row_dual,
                equality_row_dual=fresh_candidate.equality_row_dual,
                deadline=deadline,
            )
        )
        upper, fresh_anchor = _fresh_dual_anchor(
            upper_longdouble, fresh_dual_receipt, fresh
        )
        receipt["fresh_dual_selected_receipt_sha256"] = fresh_anchor

        for prepared in (parent, fresh):
            post_sha = _frame_sha256(
                matrices=prepared.matrices,
                arrays=prepared.dense_frame,
                deadline=deadline,
                chunk_bytes=8 * _CHUNK,
            )
            if post_sha != prepared.frame_sha256:
                raise _CertificateError("numeric frame changed during orchestration")
            _recheck_authority_input_identity_and_readonly(
                prepared.identity_records
            )
        _recheck_candidate_identity(parent_candidate_identity)
        _recheck_candidate_identity(fresh_candidate_identity)
        parent_candidate_sha_post, _ = _validate_candidate_receipt(
            parent_candidate,
            parent,
            label="parent_post",
            deadline=deadline,
        )
        fresh_candidate_sha_post, _ = _validate_candidate_receipt(
            fresh_candidate,
            fresh,
            label="fresh_post",
            deadline=deadline,
        )
        if (
            parent_candidate_sha_post != parent_candidate_sha
            or fresh_candidate_sha_post != fresh_candidate_sha
        ):
            raise _CertificateError("candidate receipt changed during orchestration")
        _deadline(deadline, "before_exact_comparison")

        lower_fraction = Fraction.from_float(lower)
        upper_fraction = Fraction.from_float(upper)
        gap = lower_fraction - upper_fraction
        receipt.update(
            {
                "parent_lower": lower,
                "parent_lower_hex": lower.hex(),
                "fresh_upper": upper,
                "fresh_upper_hex": upper.hex(),
                "exact_gap": _fraction_text(gap),
                "exact_comparison_completed": True,
                "numeric_frame_authority": True,
            }
        )
        if not upper_fraction < lower_fraction:
            receipt["status"] = "valid_bounds_without_strict_ordering"
            return None, receipt

        receipt.update(
            {
                "status": "strict_supplied_frame_optimum_ordering_certified",
                "strict_relaxed_lp_improvement_certified": True,
                "strict_supplied_frame_optimum_ordering_certified": True,
                "numeric_frame_authority": True,
                "proof_authority": False,
                "verdict_authority": False,
                "parent_binding_authority": False,
                "constraint_frame_sound_extension_authority": False,
                "sound_tightening_improvement_authority": False,
            }
        )
        descriptor_receipt = MappingProxyType(dict(receipt))
        descriptor = StrictSuppliedSplitLPOrderingCertificate(
            schema=_DESCRIPTOR_SCHEMA,
            parent_lower=lower,
            fresh_upper=upper,
            exact_gap=gap,
            parent_frame_sha256=parent.frame_sha256,
            fresh_frame_sha256=fresh.frame_sha256,
            parent_objective_envelope_sha256=parent.sealed_fields[
                "_envelope_sha256"
            ],
            fresh_objective_envelope_sha256=fresh.sealed_fields[
                "_envelope_sha256"
            ],
            strict_relaxed_lp_improvement_certified=True,
            strict_relaxed_lp_improvement_scope=(
                "two_supplied_numeric_frames_only"
            ),
            strict_supplied_frame_optimum_ordering_certified=True,
            numeric_frame_authority=True,
            proof_authority=False,
            verdict_authority=False,
            parent_binding_authority=False,
            sound_tightening_improvement_authority=False,
            hostile_concurrent_aba_resistance=False,
            trusted_no_concurrent_mutation_required=True,
            one_use_live_owner_required_for_sound_tightening=True,
            receipt=descriptor_receipt,
        )
        return descriptor, receipt
    except _Deadline as exc:
        receipt["status"] = f"deadline_exhausted:{str(exc)[:120]}"
        return None, receipt
    except Exception as exc:
        receipt["status"] = f"invalid:{type(exc).__name__}:{str(exc)[:120]}"
        return None, receipt


__all__ = [
    "PreformedSplitLPProblem",
    "SplitRelaxedLPFrame",
    "StrictSuppliedSplitLPOrderingCertificate",
    "certify_strict_preformed_split_lp_improvement",
]
