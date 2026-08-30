"""Toy-only live receipts for adaptive phase-forest LP upper bounds.

This module is deliberately disconnected from the verifier and BaB.  It
closes one narrower experimental hole: a phase-forest callback must not be
able to solve rival A and then attach the resulting number to rival B.

Each node/rival LP result is checked by the solver layer's independent
long-double Lagrangian upper-bound checker.  A process-local, object-identity
capability then binds that checked number to the live ``SparseHZono``, stable
rival identity, raw ASSERT digest, invocation configuration, and deadline.
Capabilities are single-use.  Serialized receipts remain diagnostics and
always carry ``proof_authority=False``.

The implementation is intentionally small-model only.  Any unsupported
shape, solver status, checker failure, deadline, tamper, omission, duplicate,
or reorder fails the whole wave closed.

As with the production capability boundary, trusted in-process Python is the
threat model.  Arbitrary reflection, registry mutation, or monkeypatching of
the issuer/checker is outside that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    PhaseBoundWaveRequest,
    PhaseForestNode,
    PhaseNodeBound,
    RivalSpec,
    RivalUpperBound,
    ordered_property_digest,
    sparse_hz_semantic_digest,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _hz_independent_lp_lagrangian_upper,
)


_SCHEMA = "hybridz_toy_phase_solver_live_receipt_v1"
_PROCESS_KEY = secrets.token_bytes(32)
_CAPABILITY_SENTINEL = object()


class ToyPhaseSolverReceiptError(RuntimeError):
    """Fail-closed error raised by the toy bound-wave adapter."""


@dataclass(frozen=True)
class ToyPhaseSolverConfig:
    """Complete bounded invocation configuration for the toy LP adapter."""

    primary_method: str = "highs-ds"
    max_variables: int = 256
    max_constraint_rows: int = 2048
    absolute_replay_tolerance: float = 2.0e-8
    relative_replay_tolerance: float = 2.0e-8
    minimum_solver_seconds: float = 1.0e-4


@dataclass(frozen=True)
class ToyPhaseSolverResult:
    """Live result object to which one capability is issued."""

    node_id: int
    node_depth: int
    lineage: Tuple[Tuple[int, int], ...]
    rival_id: int
    status: str
    upper: Optional[float]
    primary_optimum: Optional[float]
    deadline_respected: bool
    proof_authority: bool = False


@dataclass(frozen=True)
class ToyPhaseSolverEvidence:
    """One live result, diagnostic seal, and opaque single-use capability."""

    result: ToyPhaseSolverResult
    receipt: Mapping[str, Any]
    capability: object
    _replay_packet: object
    proof_authority: bool = False


@dataclass(frozen=True)
class ToyPhaseEvidenceValidation:
    """Validation outcome; it never grants production proof authority."""

    valid: bool
    upper: Optional[float]
    errors: Tuple[str, ...]
    proof_authority: bool = False


class _LiveCapability:
    __slots__ = ("_identity",)

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise TypeError("phase solver capabilities are issuer-only")
        self._identity = secrets.token_hex(32)


@dataclass(frozen=True)
class _ReplayPacket:
    row_dual_bytes: bytes
    row_dual_size: int
    checker_status: str
    checker_upper_hex: str


_LIVE_CAPABILITIES: dict[
    int,
    tuple[
        _LiveCapability,
        ToyPhaseSolverResult,
        dict[str, Any],
        _ReplayPacket,
        str,
        str,
    ],
] = {}
_ISSUED_SLOTS: set[tuple[str, int, int, int]] = set()
_ISSUED_SLOT_LOCK = threading.Lock()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(value: object, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError("expected strict integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError("integer below minimum")
    return normalized


def _validate_config(
    config: ToyPhaseSolverConfig,
) -> ToyPhaseSolverConfig:
    if type(config) is not ToyPhaseSolverConfig:
        raise ValueError("config must be exact ToyPhaseSolverConfig")
    if config.primary_method not in {"highs-ds", "highs-ipm"}:
        raise ValueError("unsupported primary LP method")
    _strict_int(config.max_variables, minimum=1)
    _strict_int(config.max_constraint_rows, minimum=0)
    for value in (
        config.absolute_replay_tolerance,
        config.relative_replay_tolerance,
        config.minimum_solver_seconds,
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (float, np.floating))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("invalid numeric solver configuration")
    if config.minimum_solver_seconds <= 0.0:
        raise ValueError("minimum_solver_seconds must be positive")
    return config


def _config_mapping(config: ToyPhaseSolverConfig) -> dict[str, Any]:
    normalized = _validate_config(config)
    return {
        "primary_method": normalized.primary_method,
        "max_variables": int(normalized.max_variables),
        "max_constraint_rows": int(
            normalized.max_constraint_rows
        ),
        "absolute_replay_tolerance_hex": float(
            normalized.absolute_replay_tolerance
        ).hex(),
        "relative_replay_tolerance_hex": float(
            normalized.relative_replay_tolerance
        ).hex(),
        "minimum_solver_seconds_hex": float(
            normalized.minimum_solver_seconds
        ).hex(),
    }


def toy_phase_solver_config_sha256(
    config: ToyPhaseSolverConfig,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(_config_mapping(config))
    ).hexdigest()


def new_toy_phase_solver_invocation_id() -> str:
    """Return an unpredictable process-local invocation nonce."""

    return secrets.token_hex(32)


def _revoke_invocation(invocation_id: object) -> None:
    """Revoke every still-live item from one atomic wave invocation."""

    if not _is_sha256(invocation_id):
        return
    doomed = [
        key
        for key, issued in _LIVE_CAPABILITIES.items()
        if issued[5] == invocation_id
    ]
    for key in doomed:
        _LIVE_CAPABILITIES.pop(key, None)


def _revoke_evidence_invocations(evidence: object) -> None:
    """Revoke invocation groups discoverable from supplied live objects."""

    for invocation_id in _live_evidence_invocation_ids(evidence):
        _revoke_invocation(invocation_id)


def _live_evidence_invocation_ids(
    evidence: object,
) -> Tuple[str, ...]:
    """Snapshot live invocation groups before any capability is consumed."""

    if not isinstance(evidence, (tuple, list)):
        return ()
    invocation_ids = set()
    for item in evidence:
        if not isinstance(item, ToyPhaseSolverEvidence):
            continue
        issued = _LIVE_CAPABILITIES.get(id(item.capability))
        if issued is not None and issued[0] is item.capability:
            invocation_ids.add(issued[5])
    return tuple(sorted(invocation_ids))


def _objective_bytes(rival: RivalSpec) -> bytes:
    objective = np.ascontiguousarray(
        np.asarray(rival.objective, dtype=np.float64)
    )
    if (
        objective.ndim != 1
        or objective.size < 1
        or not np.all(np.isfinite(objective))
    ):
        raise ValueError("malformed rival objective")
    return objective.tobytes(order="C")


def _normalize_deadline(deadline: object) -> float:
    if isinstance(deadline, (bool, np.bool_)):
        raise ValueError("deadline cannot be bool")
    normalized = float(deadline)
    if not math.isfinite(normalized):
        raise ValueError("deadline must be finite")
    return normalized


def _normalize_invocation_id(invocation_id: object) -> str:
    if not _is_sha256(invocation_id):
        raise ValueError("invocation id must be 32-byte lowercase hex")
    return str(invocation_id)


def _normalize_wave_index(wave_index: object) -> int:
    return _strict_int(wave_index, minimum=0)


def _normalize_position(position: object) -> int:
    return _strict_int(position, minimum=0)


def _claim_invocation_slot(
    invocation_id: str,
    wave_index: int,
    node_position: int,
    rival_position: int,
) -> None:
    """Burn one exact invocation slot, including after later failure."""

    key = (
        invocation_id,
        int(wave_index),
        int(node_position),
        int(rival_position),
    )
    with _ISSUED_SLOT_LOCK:
        if key in _ISSUED_SLOTS:
            raise ToyPhaseSolverReceiptError(
                "duplicate_or_stale_invocation_slot"
            )
        _ISSUED_SLOTS.add(key)


def _live_lp_frame(
    hz: SparseHZono,
    rival: RivalSpec,
) -> tuple[Any, ...]:
    """Build the exact live continuous relaxation without solver caches."""

    if not isinstance(hz, SparseHZono):
        raise ValueError("node HZ must be SparseHZono")
    objective = np.asarray(rival.objective, dtype=np.float64)
    if objective.shape != (hz.n_out,):
        raise ValueError("rival objective/output width mismatch")

    combined_generator = sp.hstack(
        [hz.Gc, hz.Gb], format="csr"
    )
    combined_generator.sum_duplicates()
    combined_generator.sort_indices()
    q = np.asarray(
        (
            sp.csr_matrix(objective.reshape(1, -1))
            @ combined_generator
        ).toarray(),
        dtype=np.float64,
    ).reshape(-1)
    constant = np.asarray(
        [float(objective @ hz.c)], dtype=np.float64
    )

    equality = sp.hstack([hz.Ac, hz.Ab], format="csr")
    if hz.Auc is None or hz.Aub is None or hz.ub is None:
        upper = sp.csr_matrix(
            (0, hz.n_cont + hz.n_bin), dtype=np.float64
        )
        upper_rhs = np.zeros(0, dtype=np.float64)
    else:
        upper = sp.hstack([hz.Auc, hz.Aub], format="csr")
        upper_rhs = np.asarray(hz.ub, dtype=np.float64).reshape(-1)
    equality.sum_duplicates()
    equality.sort_indices()
    upper.sum_duplicates()
    upper.sort_indices()
    equality_rhs = np.asarray(hz.b, dtype=np.float64).reshape(-1)
    full = sp.vstack([equality, upper], format="csr")
    lower_rows = np.concatenate(
        [
            equality_rhs,
            np.full(upper.shape[0], -np.inf, dtype=np.float64),
        ]
    )
    upper_rows = np.concatenate([equality_rhs, upper_rhs])
    lower_variables = np.full(
        hz.n_cont + hz.n_bin, -1.0, dtype=np.float64
    )
    upper_variables = np.full(
        hz.n_cont + hz.n_bin, 1.0, dtype=np.float64
    )
    arrays = (
        q,
        constant,
        equality_rhs,
        upper_rhs,
        lower_variables,
        upper_variables,
    )
    if any(
        not np.all(np.isfinite(array))
        for array in arrays
    ):
        raise ValueError("non-finite live LP frame")
    for matrix in (combined_generator, equality, upper, full):
        if (
            not matrix.has_canonical_format
            or not matrix.has_sorted_indices
            or (
                matrix.nnz
                and not np.all(np.isfinite(matrix.data))
            )
        ):
            raise ValueError("malformed live LP matrix")
    return (
        q,
        combined_generator,
        constant,
        equality,
        equality_rhs,
        upper,
        upper_rhs,
        lower_rows,
        upper_rows,
        lower_variables,
        upper_variables,
        full,
    )


def _checked_upper(
    *,
    hz: SparseHZono,
    rival: RivalSpec,
    frame,
    row_dual: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    (
        _q,
        combined_generator,
        _constant,
        _equality,
        _equality_rhs,
        _upper,
        _upper_rhs,
        lower_rows,
        upper_rows,
        lower_variables,
        upper_variables,
        full,
    ) = frame
    checked, checker_receipt = _hz_independent_lp_lagrangian_upper(
        c=hz.c,
        Gc=combined_generator,
        C_row=np.asarray(rival.objective, dtype=np.float64),
        threshold=0.0,
        A=full,
        rl=lower_rows,
        ru=upper_rows,
        lb=lower_variables,
        ub=upper_variables,
        row_dual=row_dual,
        center_error=None,
    )
    if (
        checked is None
        or checker_receipt.get("status") != "verified_upper"
        or not np.isfinite(checked)
    ):
        raise ToyPhaseSolverReceiptError(
            "independent_lagrangian_checker_incomplete"
        )
    rounded = np.float64(checked)
    if not np.isfinite(rounded):
        raise ToyPhaseSolverReceiptError(
            "checked_upper_float64_overflow"
        )
    # The forest stores float64 telemetry.  Round outward when narrowing the
    # checker's authoritative long-double result.
    upper = float(np.nextafter(rounded, np.inf))
    if not math.isfinite(upper):
        raise ToyPhaseSolverReceiptError(
            "checked_upper_outward_float64_overflow"
        )
    return upper, dict(checker_receipt)


def _receipt_body(
    *,
    node: PhaseForestNode,
    rival: RivalSpec,
    config: ToyPhaseSolverConfig,
    deadline: float,
    invocation_id: str,
    wave_index: int,
    node_position: int,
    rival_position: int,
    property_digest: str,
    solver_time_limit: float,
    result: ToyPhaseSolverResult,
    replay: _ReplayPacket,
    primary_status: int,
    primary_backend: str,
    primary_execution_method: str,
) -> dict[str, Any]:
    objective = _objective_bytes(rival)
    return {
        "schema": _SCHEMA,
        "proof_authority": False,
        "toy_only": True,
        "verifier_connected": False,
        "bab_connected": False,
        "live_process_only": True,
        "portable_authority": False,
        "node": {
            "node_id": int(node.node_id),
            "depth": int(node.depth),
            "lineage": [
                [int(column), int(sign)]
                for column, sign in node.lineage
            ],
            "live_sparse_hz_semantic_sha256": (
                sparse_hz_semantic_digest(node.hz)
            ),
        },
        "rival": {
            "rival_id": int(rival.rival_id),
            "binding_sha256": rival.binding_digest,
            "objective_shape": [len(rival.objective)],
            "objective_f64_sha256": hashlib.sha256(
                objective
            ).hexdigest(),
            "threshold_hex": float(rival.threshold).hex(),
            "assert_sha256": rival.assert_digest,
        },
        "invocation": {
            "invocation_id": invocation_id,
            "wave_index": int(wave_index),
            "node_position": int(node_position),
            "rival_position": int(rival_position),
            "ordered_property_sha256": property_digest,
            "requested_deadline_hex": float(deadline).hex(),
            "solver_time_limit_hex": float(solver_time_limit).hex(),
            "config": _config_mapping(config),
            "config_sha256": toy_phase_solver_config_sha256(config),
            "backend": primary_backend,
            "primary_method": primary_execution_method,
            "objective_direction": "maximize",
            "remaining_binary_domain": (
                "continuous_relaxation_minus_one_plus_one"
            ),
            "independent_checker": (
                "solver_hz_longdouble_lagrangian_upper"
            ),
        },
        "result": {
            "status": result.status,
            "upper_hex": (
                None
                if result.upper is None
                else float(result.upper).hex()
            ),
            "primary_optimum_hex": (
                None
                if result.primary_optimum is None
                else float(result.primary_optimum).hex()
            ),
            "primary_status": int(primary_status),
            "deadline_respected": bool(
                result.deadline_respected
            ),
            "checker_status": replay.checker_status,
            "checker_upper_hex": replay.checker_upper_hex,
            "row_dual_size": int(replay.row_dual_size),
            "row_dual_sha256": hashlib.sha256(
                replay.row_dual_bytes
            ).hexdigest(),
        },
    }


def solve_toy_phase_rival_evidence(
    node: PhaseForestNode,
    rival: RivalSpec,
    *,
    config: ToyPhaseSolverConfig,
    deadline: float,
    invocation_id: str,
    wave_index: int,
    node_position: int,
    rival_position: int,
    property_digest: str,
) -> ToyPhaseSolverEvidence:
    """Solve and issue one non-authoritative live node/rival capability."""

    config = _validate_config(config)
    deadline = _normalize_deadline(deadline)
    invocation_id = _normalize_invocation_id(invocation_id)
    wave_index = _normalize_wave_index(wave_index)
    node_position = _normalize_position(node_position)
    rival_position = _normalize_position(rival_position)
    if not isinstance(node, PhaseForestNode):
        raise ValueError("node must be PhaseForestNode")
    if not isinstance(rival, RivalSpec):
        raise ValueError("rival must be RivalSpec")
    if not _is_sha256(property_digest):
        raise ValueError("invalid ordered property digest")
    _claim_invocation_slot(
        invocation_id,
        wave_index,
        node_position,
        rival_position,
    )
    if time.monotonic() >= deadline:
        raise ToyPhaseSolverReceiptError("deadline_before_lp")
    semantic_before = sparse_hz_semantic_digest(node.hz)
    frame = _live_lp_frame(node.hz, rival)
    (
        q,
        _combined_generator,
        constant,
        equality,
        equality_rhs,
        upper,
        upper_rhs,
        _lower_rows,
        _upper_rows,
        lower_variables,
        upper_variables,
        _full,
    ) = frame
    variable_count = int(q.size)
    row_count = int(equality.shape[0] + upper.shape[0])
    if (
        variable_count > config.max_variables
        or row_count > config.max_constraint_rows
    ):
        raise ToyPhaseSolverReceiptError("toy_lp_size_cap")
    remaining = float(deadline - time.monotonic())
    if remaining < config.minimum_solver_seconds:
        raise ToyPhaseSolverReceiptError("deadline_before_primary_solve")
    if variable_count == 0:
        if (
            np.any(equality_rhs != 0.0)
            or np.any(upper_rhs < 0.0)
        ):
            raise ToyPhaseSolverReceiptError(
                "exact_point_base_infeasible"
            )
        primary_status = 0
        primary_backend = "exact_stored_float_point"
        primary_execution_method = "exact_point"
        primary = float(constant[0])
        equality_dual = np.zeros(
            equality.shape[0], dtype=np.float64
        )
        upper_dual = np.zeros(
            upper.shape[0], dtype=np.float64
        )
    else:
        solved = linprog(
            -q,
            A_ub=upper if upper.shape[0] else None,
            b_ub=upper_rhs if upper.shape[0] else None,
            A_eq=equality if equality.shape[0] else None,
            b_eq=equality_rhs if equality.shape[0] else None,
            bounds=list(zip(lower_variables, upper_variables)),
            method=config.primary_method,
            options={"time_limit": remaining},
        )
        if time.monotonic() >= deadline:
            raise ToyPhaseSolverReceiptError(
                "deadline_after_primary_solve"
            )
        if not solved.success or int(solved.status) != 0:
            raise ToyPhaseSolverReceiptError(
                f"primary_lp_not_optimal:{int(solved.status)}"
            )
        primary_status = int(solved.status)
        primary_backend = "scipy.optimize.linprog"
        primary_execution_method = config.primary_method
        primary = float(constant[0] - float(solved.fun))
        equality_dual = (
            np.asarray(
                solved.eqlin.marginals, dtype=np.float64
            )
            if equality.shape[0]
            else np.zeros(0, dtype=np.float64)
        )
        upper_dual = (
            np.asarray(
                solved.ineqlin.marginals, dtype=np.float64
            )
            if upper.shape[0]
            else np.zeros(0, dtype=np.float64)
        )
    row_dual = np.ascontiguousarray(
        np.concatenate([equality_dual, upper_dual]),
        dtype=np.float64,
    )
    checked_upper, checker_receipt = _checked_upper(
        hz=node.hz,
        rival=rival,
        frame=frame,
        row_dual=row_dual,
    )
    scale = max(1.0, abs(primary), abs(checked_upper))
    replay_tolerance = (
        config.absolute_replay_tolerance
        + config.relative_replay_tolerance * scale
    )
    if (
        not math.isfinite(primary)
        or checked_upper + replay_tolerance < primary
        or checked_upper - primary > replay_tolerance
    ):
        raise ToyPhaseSolverReceiptError(
            "primary_independent_upper_gap"
        )
    if sparse_hz_semantic_digest(node.hz) != semantic_before:
        raise ToyPhaseSolverReceiptError(
            "node_mutated_during_solver_invocation"
        )
    now = time.monotonic()
    if now >= deadline:
        raise ToyPhaseSolverReceiptError(
            "deadline_before_capability_issue"
        )
    result = ToyPhaseSolverResult(
        node_id=int(node.node_id),
        node_depth=int(node.depth),
        lineage=tuple(node.lineage),
        rival_id=int(rival.rival_id),
        status="VERIFIED_UPPER",
        upper=checked_upper,
        primary_optimum=primary,
        deadline_respected=True,
        proof_authority=False,
    )
    replay = _ReplayPacket(
        row_dual_bytes=row_dual.tobytes(order="C"),
        row_dual_size=int(row_dual.size),
        checker_status=str(checker_receipt["status"]),
        checker_upper_hex=float(checked_upper).hex(),
    )
    body = _receipt_body(
        node=node,
        rival=rival,
        config=config,
        deadline=deadline,
        invocation_id=invocation_id,
        wave_index=wave_index,
        node_position=node_position,
        rival_position=rival_position,
        property_digest=property_digest,
        solver_time_limit=remaining,
        result=result,
        replay=replay,
        primary_status=primary_status,
        primary_backend=primary_backend,
        primary_execution_method=primary_execution_method,
    )
    mac = hmac.new(
        _PROCESS_KEY,
        _canonical_bytes(body),
        digestmod=hashlib.sha256,
    ).hexdigest()
    receipt = {**body, "process_mac_sha256": mac}
    capability = _LiveCapability(_CAPABILITY_SENTINEL)
    _LIVE_CAPABILITIES[id(capability)] = (
        capability,
        result,
        receipt,
        replay,
        mac,
        invocation_id,
    )
    return ToyPhaseSolverEvidence(
        result=result,
        receipt=receipt,
        capability=capability,
        _replay_packet=replay,
        proof_authority=False,
    )


def _consume_one(
    evidence: object,
    *,
    node: PhaseForestNode,
    rival: RivalSpec,
    config: ToyPhaseSolverConfig,
    deadline: float,
    invocation_id: str,
    wave_index: int,
    node_position: int,
    rival_position: int,
    property_digest: str,
) -> ToyPhaseEvidenceValidation:
    errors: list[str] = []
    if not isinstance(evidence, ToyPhaseSolverEvidence):
        return ToyPhaseEvidenceValidation(
            valid=False,
            upper=None,
            errors=("evidence_wrong_type",),
        )
    capability = evidence.capability
    issued = _LIVE_CAPABILITIES.pop(id(capability), None)
    if (
        issued is None
        or issued[0] is not capability
        or not isinstance(capability, _LiveCapability)
    ):
        errors.append("missing_stale_or_forged_capability")
    else:
        if issued[1] is not evidence.result:
            errors.append("live_result_identity_mismatch")
        if issued[2] is not evidence.receipt:
            errors.append("live_receipt_identity_mismatch")
        if issued[3] is not evidence._replay_packet:
            errors.append("live_replay_packet_identity_mismatch")
    if evidence.proof_authority is not False:
        errors.append("evidence_claimed_proof_authority")
    if type(evidence.result) is not ToyPhaseSolverResult:
        errors.append("result_wrong_type")
    if type(evidence.receipt) is not dict:
        errors.append("receipt_not_live_dict")
        receipt: dict[str, Any] = {}
    else:
        receipt = evidence.receipt
    if not isinstance(evidence._replay_packet, _ReplayPacket):
        errors.append("replay_packet_wrong_type")
        replay = None
    else:
        replay = evidence._replay_packet

    expected_deadline: Optional[float] = None
    try:
        config = _validate_config(config)
        expected_deadline = _normalize_deadline(deadline)
        deadline = expected_deadline
        invocation_id = _normalize_invocation_id(invocation_id)
        wave_index = _normalize_wave_index(wave_index)
        node_position = _normalize_position(node_position)
        rival_position = _normalize_position(rival_position)
        if not _is_sha256(property_digest):
            raise ValueError("bad property digest")
    except Exception:
        errors.append("expected_invocation_malformed")

    result = evidence.result
    if type(result) is ToyPhaseSolverResult:
        if result.proof_authority is not False:
            errors.append("result_claimed_proof_authority")
        if result.status != "VERIFIED_UPPER":
            errors.append("result_status_mismatch")
        if result.deadline_respected is not True:
            errors.append("result_deadline_not_respected")
        if (
            result.node_id != int(node.node_id)
            or result.node_depth != int(node.depth)
            or result.lineage != tuple(node.lineage)
        ):
            errors.append("result_node_binding_mismatch")
        if result.rival_id != int(rival.rival_id):
            errors.append("result_rival_binding_mismatch")
        if (
            result.upper is None
            or not math.isfinite(float(result.upper))
            or result.primary_optimum is None
            or not math.isfinite(float(result.primary_optimum))
        ):
            errors.append("result_numeric_malformed")

    if (
        expected_deadline is not None
        and time.monotonic() >= expected_deadline
    ):
        errors.append("deadline_expired_at_consume")

    expected_mac = None
    if receipt:
        raw_mac = receipt.get("process_mac_sha256")
        body = {
            key: value
            for key, value in receipt.items()
            if key != "process_mac_sha256"
        }
        try:
            expected_mac = hmac.new(
                _PROCESS_KEY,
                _canonical_bytes(body),
                digestmod=hashlib.sha256,
            ).hexdigest()
        except Exception:
            errors.append("receipt_not_canonical")
        if (
            not _is_sha256(raw_mac)
            or expected_mac is None
            or not hmac.compare_digest(str(raw_mac), expected_mac)
        ):
            errors.append("receipt_mac_mismatch")
        if issued is not None and raw_mac != issued[4]:
            errors.append("issued_receipt_mac_mismatch")
        expected_keys = {
            "schema",
            "proof_authority",
            "toy_only",
            "verifier_connected",
            "bab_connected",
            "live_process_only",
            "portable_authority",
            "node",
            "rival",
            "invocation",
            "result",
            "process_mac_sha256",
        }
        if set(receipt) != expected_keys:
            errors.append("receipt_schema_keys_mismatch")
        if (
            receipt.get("schema") != _SCHEMA
            or receipt.get("proof_authority") is not False
            or receipt.get("toy_only") is not True
            or receipt.get("verifier_connected") is not False
            or receipt.get("bab_connected") is not False
            or receipt.get("live_process_only") is not True
            or receipt.get("portable_authority") is not False
        ):
            errors.append("receipt_scope_mismatch")

    if replay is not None and type(result) is ToyPhaseSolverResult:
        try:
            row_dual = np.frombuffer(
                replay.row_dual_bytes, dtype=np.float64
            )
            if row_dual.size != replay.row_dual_size:
                raise ValueError("row dual size mismatch")
            frame = _live_lp_frame(node.hz, rival)
            checked_upper, checker_receipt = _checked_upper(
                hz=node.hz,
                rival=rival,
                frame=frame,
                row_dual=row_dual,
            )
            if (
                checker_receipt.get("status")
                != replay.checker_status
                or float(checked_upper).hex()
                != replay.checker_upper_hex
                or result.upper is None
                or float(result.upper).hex()
                != float(checked_upper).hex()
            ):
                errors.append("independent_numeric_replay_mismatch")
        except Exception:
            errors.append("independent_numeric_replay_failed")
        if (
            expected_deadline is not None
            and time.monotonic() >= expected_deadline
        ):
            errors.append("deadline_expired_after_numeric_replay")

    try:
        if replay is None or type(result) is not ToyPhaseSolverResult:
            raise ValueError("missing live result/replay")
        node_receipt = receipt.get("node")
        rival_receipt = receipt.get("rival")
        invocation_receipt = receipt.get("invocation")
        result_receipt = receipt.get("result")
        if (
            type(node_receipt) is not dict
            or node_receipt
            != {
                "node_id": int(node.node_id),
                "depth": int(node.depth),
                "lineage": [
                    [int(column), int(sign)]
                    for column, sign in node.lineage
                ],
                "live_sparse_hz_semantic_sha256": (
                    sparse_hz_semantic_digest(node.hz)
                ),
            }
        ):
            errors.append("sealed_node_binding_mismatch")
        expected_rival = {
            "rival_id": int(rival.rival_id),
            "binding_sha256": rival.binding_digest,
            "objective_shape": [len(rival.objective)],
            "objective_f64_sha256": hashlib.sha256(
                _objective_bytes(rival)
            ).hexdigest(),
            "threshold_hex": float(rival.threshold).hex(),
            "assert_sha256": rival.assert_digest,
        }
        if type(rival_receipt) is not dict or rival_receipt != expected_rival:
            errors.append("sealed_rival_binding_mismatch")
        if type(invocation_receipt) is not dict:
            errors.append("sealed_invocation_malformed")
        else:
            invariant_invocation = {
                "invocation_id": invocation_id,
                "wave_index": int(wave_index),
                "node_position": int(node_position),
                "rival_position": int(rival_position),
                "ordered_property_sha256": property_digest,
                "requested_deadline_hex": float(
                    expected_deadline
                ).hex(),
                "config": _config_mapping(config),
                "config_sha256": toy_phase_solver_config_sha256(
                    config
                ),
                "backend": (
                    "exact_stored_float_point"
                    if node.hz.n_cont + node.hz.n_bin == 0
                    else "scipy.optimize.linprog"
                ),
                "primary_method": (
                    "exact_point"
                    if node.hz.n_cont + node.hz.n_bin == 0
                    else config.primary_method
                ),
                "objective_direction": "maximize",
                "remaining_binary_domain": (
                    "continuous_relaxation_minus_one_plus_one"
                ),
                "independent_checker": (
                    "solver_hz_longdouble_lagrangian_upper"
                ),
            }
            for key, value in invariant_invocation.items():
                if invocation_receipt.get(key) != value:
                    errors.append("sealed_invocation_binding_mismatch")
                    break
            try:
                solver_limit = float.fromhex(
                    invocation_receipt["solver_time_limit_hex"]
                )
                if (
                    not math.isfinite(solver_limit)
                    or solver_limit <= 0.0
                ):
                    raise ValueError("bad solver limit")
            except Exception:
                errors.append("sealed_solver_time_limit_malformed")
        expected_result = {
            "status": result.status,
            "upper_hex": (
                None
                if result.upper is None
                else float(result.upper).hex()
            ),
            "primary_optimum_hex": (
                None
                if result.primary_optimum is None
                else float(result.primary_optimum).hex()
            ),
            "primary_status": 0,
            "deadline_respected": bool(
                result.deadline_respected
            ),
            "checker_status": replay.checker_status,
            "checker_upper_hex": replay.checker_upper_hex,
            "row_dual_size": int(replay.row_dual_size),
            "row_dual_sha256": hashlib.sha256(
                replay.row_dual_bytes
            ).hexdigest(),
        }
        if type(result_receipt) is not dict or result_receipt != expected_result:
            errors.append("sealed_result_binding_mismatch")
    except Exception:
        errors.append("sealed_binding_rederivation_failed")

    unique_errors = tuple(sorted(set(errors)))
    return ToyPhaseEvidenceValidation(
        valid=not unique_errors,
        upper=(
            float(result.upper)
            if not unique_errors
            and type(result) is ToyPhaseSolverResult
            and result.upper is not None
            else None
        ),
        errors=unique_errors,
        proof_authority=False,
    )


def consume_toy_phase_rival_evidence(
    evidence: object,
    *,
    node: PhaseForestNode,
    rival: RivalSpec,
    config: ToyPhaseSolverConfig,
    deadline: float,
    invocation_id: str,
    wave_index: int,
    node_position: int,
    rival_position: int,
    property_digest: str,
) -> ToyPhaseEvidenceValidation:
    """Consume exactly one live capability and independently replay its upper."""

    return _consume_one(
        evidence,
        node=node,
        rival=rival,
        config=config,
        deadline=deadline,
        invocation_id=invocation_id,
        wave_index=wave_index,
        node_position=node_position,
        rival_position=rival_position,
        property_digest=property_digest,
    )


def solve_toy_phase_solver_batch(
    request: PhaseBoundWaveRequest,
    *,
    config: ToyPhaseSolverConfig,
    invocation_id: str,
) -> Tuple[ToyPhaseSolverEvidence, ...]:
    """Issue exact node-major/rival-major evidence for one complete wave."""

    if not isinstance(request, PhaseBoundWaveRequest):
        raise ValueError("request must be PhaseBoundWaveRequest")
    if request.proof_authority is not False:
        raise ValueError("request cannot claim proof authority")
    if ordered_property_digest(request.rivals) != request.property_digest:
        raise ValueError("request property binding mismatch")
    output = []
    try:
        for node_position, node in enumerate(request.nodes):
            for rival_position, rival in enumerate(request.rivals):
                output.append(
                    solve_toy_phase_rival_evidence(
                        node,
                        rival,
                        config=config,
                        deadline=request.deadline,
                        invocation_id=invocation_id,
                        wave_index=request.wave_index,
                        node_position=node_position,
                        rival_position=rival_position,
                        property_digest=request.property_digest,
                    )
                )
    except Exception:
        for evidence in output:
            _LIVE_CAPABILITIES.pop(id(evidence.capability), None)
        _revoke_invocation(invocation_id)
        raise
    return tuple(output)


def _consume_toy_phase_solver_batch_inner(
    evidence: object,
    request: PhaseBoundWaveRequest,
    *,
    config: ToyPhaseSolverConfig,
    invocation_id: str,
) -> tuple[
    Optional[Tuple[Tuple[RivalUpperBound, ...], ...]],
    Tuple[str, ...],
]:
    """Consume a complete exact-order wave or reject it atomically."""

    if not isinstance(request, PhaseBoundWaveRequest):
        _revoke_evidence_invocations(evidence)
        _revoke_invocation(invocation_id)
        return None, ("request_wrong_type",)
    if not isinstance(evidence, tuple):
        _revoke_evidence_invocations(evidence)
        _revoke_invocation(invocation_id)
        return None, ("batch_evidence_not_tuple",)
    try:
        request_binding_ok = bool(
            request.proof_authority is False
            and ordered_property_digest(request.rivals)
            == request.property_digest
        )
    except Exception:
        request_binding_ok = False
    if not request_binding_ok:
        _revoke_evidence_invocations(evidence)
        _revoke_invocation(invocation_id)
        return None, ("request_property_binding_mismatch",)
    expected = len(request.nodes) * len(request.rivals)
    errors: list[str] = []
    if len(evidence) != expected:
        errors.append("batch_evidence_count_mismatch")
    live_capability_ids = [
        id(item.capability)
        for item in evidence
        if isinstance(item, ToyPhaseSolverEvidence)
    ]
    if len(set(live_capability_ids)) != len(live_capability_ids):
        errors.append("batch_duplicate_capability")
    cursor = 0
    for node_position, node in enumerate(request.nodes):
        for rival_position, rival in enumerate(request.rivals):
            if cursor >= len(evidence):
                break
            item = evidence[cursor]
            if not isinstance(item, ToyPhaseSolverEvidence):
                errors.append("batch_evidence_wrong_type")
            else:
                result = item.result
                if (
                    type(result) is not ToyPhaseSolverResult
                    or result.node_id != int(node.node_id)
                    or result.node_depth != int(node.depth)
                    or result.lineage != tuple(node.lineage)
                    or result.rival_id != int(rival.rival_id)
                ):
                    errors.append("batch_order_or_binding_mismatch")
            cursor += 1
    if errors:
        _revoke_evidence_invocations(evidence)
        _revoke_invocation(invocation_id)
        return None, tuple(sorted(set(errors)))

    validated: list[Tuple[RivalUpperBound, ...]] = []
    cursor = 0
    for node_position, node in enumerate(request.nodes):
        node_bounds = []
        for rival_position, rival in enumerate(request.rivals):
            item = evidence[cursor]
            outcome = _consume_one(
                item,
                node=node,
                rival=rival,
                config=config,
                deadline=request.deadline,
                invocation_id=invocation_id,
                wave_index=request.wave_index,
                node_position=node_position,
                rival_position=rival_position,
                property_digest=request.property_digest,
            )
            cursor += 1
            if not outcome.valid or outcome.upper is None:
                errors.extend(outcome.errors or ("batch_consume_failed",))
                break
            node_bounds.append(
                RivalUpperBound(
                    rival_id=int(rival.rival_id),
                    binding_digest=rival.binding_digest,
                    upper=float(outcome.upper),
                )
            )
        if errors:
            break
        validated.append(tuple(node_bounds))
    if errors:
        _revoke_evidence_invocations(evidence)
        _revoke_invocation(invocation_id)
        return None, tuple(sorted(set(errors)))
    _revoke_invocation(invocation_id)
    return tuple(validated), ()


def consume_toy_phase_solver_batch(
    evidence: object,
    request: PhaseBoundWaveRequest,
    *,
    config: ToyPhaseSolverConfig,
    invocation_id: str,
) -> tuple[
    Optional[Tuple[Tuple[RivalUpperBound, ...], ...]],
    Tuple[str, ...],
]:
    """Consume one exact wave and revoke supplied invocations atomically."""

    actual_invocation_ids = _live_evidence_invocation_ids(evidence)
    try:
        return _consume_toy_phase_solver_batch_inner(
            evidence,
            request,
            config=config,
            invocation_id=invocation_id,
        )
    finally:
        _revoke_invocation(invocation_id)
        for actual_invocation_id in actual_invocation_ids:
            _revoke_invocation(actual_invocation_id)


def _strictly_safe(
    bounds: Sequence[RivalUpperBound],
    rivals: Sequence[RivalSpec],
) -> bool:
    upper = np.asarray(
        [bound.upper for bound in bounds], dtype=np.float64
    )
    threshold = np.asarray(
        [rival.threshold for rival in rivals], dtype=np.float64
    )
    scale = np.maximum(
        1.0, np.maximum(np.abs(upper), np.abs(threshold))
    )
    tolerance = max(
        100.0 * np.finfo(np.float64).eps, 1.0e-11
    ) * scale
    return bool(np.all(upper < threshold - tolerance))


class ToyReceiptedPhaseBoundWave:
    """Callable C77 adapter that cannot detach numbers from rival bindings."""

    def __init__(
        self,
        config: ToyPhaseSolverConfig = ToyPhaseSolverConfig(),
    ) -> None:
        self.config = _validate_config(config)
        self._run_nonce = new_toy_phase_solver_invocation_id()
        self._seen_waves: set[tuple[int, str]] = set()
        self.audit_receipts: list[dict[str, Any]] = []
        self.proof_authority = False

    def __call__(
        self, request: PhaseBoundWaveRequest
    ) -> Tuple[PhaseNodeBound, ...]:
        if not isinstance(request, PhaseBoundWaveRequest):
            raise ToyPhaseSolverReceiptError("request_wrong_type")
        wave_key = (int(request.wave_index), request.property_digest)
        if wave_key in self._seen_waves:
            raise ToyPhaseSolverReceiptError(
                "duplicate_wave_invocation"
            )
        self._seen_waves.add(wave_key)
        digest = hashlib.sha256()
        digest.update(b"hybridz_toy_phase_solver_wave_v1")
        digest.update(self._run_nonce.encode("ascii"))
        digest.update(str(request.wave_index).encode("ascii"))
        digest.update(request.property_digest.encode("ascii"))
        invocation_id = digest.hexdigest()
        evidence = solve_toy_phase_solver_batch(
            request,
            config=self.config,
            invocation_id=invocation_id,
        )
        # Copies retained here are deliberately non-authoritative diagnostics;
        # the live receipt objects themselves are consumed immediately below.
        diagnostic_copies = [
            json.loads(_canonical_bytes(item.receipt))
            for item in evidence
        ]
        bounds, errors = consume_toy_phase_solver_batch(
            evidence,
            request,
            config=self.config,
            invocation_id=invocation_id,
        )
        self.audit_receipts.extend(diagnostic_copies)
        if time.monotonic() >= request.deadline:
            raise ToyPhaseSolverReceiptError(
                "deadline_after_batch_capability_validation"
            )
        if bounds is None or errors:
            raise ToyPhaseSolverReceiptError(
                "batch_capability_validation:"
                + ",".join(errors or ("unknown",))
            )
        output = []
        for node, node_bounds in zip(request.nodes, bounds):
            output.append(
                PhaseNodeBound(
                    node_id=int(node.node_id),
                    lineage=tuple(node.lineage),
                    remaining_bcol_ids=tuple(
                        int(value)
                        for value in node.hz.bcol_ids.tolist()
                    ),
                    rival_bounds=node_bounds,
                    property_digest=request.property_digest,
                    node_semantic_digest=sparse_hz_semantic_digest(
                        node.hz
                    ),
                    verdict=(
                        "SAFE"
                        if _strictly_safe(
                            node_bounds, request.rivals
                        )
                        else "UNKNOWN"
                    ),
                    deadline_respected=True,
                    proof_authority=False,
                )
            )
        return tuple(output)


__all__ = [
    "ToyPhaseEvidenceValidation",
    "ToyPhaseSolverConfig",
    "ToyPhaseSolverEvidence",
    "ToyPhaseSolverReceiptError",
    "ToyPhaseSolverResult",
    "ToyReceiptedPhaseBoundWave",
    "consume_toy_phase_rival_evidence",
    "consume_toy_phase_solver_batch",
    "new_toy_phase_solver_invocation_id",
    "solve_toy_phase_rival_evidence",
    "solve_toy_phase_solver_batch",
    "toy_phase_solver_config_sha256",
]
