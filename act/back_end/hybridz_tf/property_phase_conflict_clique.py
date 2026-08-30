#!/usr/bin/env python3
"""Candidate-only property-conditioned phase-conflict clique cuts.

PC-PCC is deliberately isolated from the verifier and BaB dispatch.  It asks
whether the signed binary literals that improve one ordered property batch
are pairwise incompatible in the *complete* parent SparseHZ relaxation.  A
clique cut is emitted only when every pair passes two different checks:

* SciPy/HiGHS proposes that fixing both literals makes the relaxation empty.
* An independent exact ``Fraction.from_float`` Farkas replay proves the same
  fact using a stored HZ row and exact factor bounds.

HiGHS therefore has no proof role.  Unsupported exact certificates, missing
edges, caps, deadlines, non-finite data, and receipt mismatches all fail
closed.  Results always have ``proof_authority=False`` and are intended only
for exact controlled toys until a separate production design exists.

The serialized receipt uses an unkeyed checksum and is diagnostic only.
``verify_pc_pcc_result`` additionally consumes an opaque, process-local
capability bound to the exact live result, parent, property batch, invocation
caps, deadline, telemetry, certificates, and cut.  The repeatable structural
validator proves cut closure but deliberately cannot prove run provenance.
Trusted in-process reflection or direct registry mutation remains outside the
threat boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
import secrets
import threading
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    ordered_property_digest,
    sparse_hz_semantic_digest,
)
from act.back_end.solver.solver_hz import SparseHZono


@dataclass(frozen=True)
class PhaseLiteral:
    """One property-worsening signed HybridZ binary literal."""

    stable_bcol_id: int
    phase: int
    binding_digest: str


@dataclass(frozen=True)
class FarkasTerm:
    """One nonnegative multiplier in an exact infeasibility certificate."""

    kind: str
    index: int
    numerator: int
    denominator: int

    @property
    def multiplier(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class ExactConflictCertificate:
    """Exact dyadic Farkas replay for one unordered literal pair."""

    literals: Tuple[PhaseLiteral, PhaseLiteral]
    parent_semantic_digest: str
    property_digest: str
    source_row_digests: Tuple[str, ...]
    terms: Tuple[FarkasTerm, ...]
    contradiction_numerator: int
    contradiction_denominator: int
    certificate_digest: str
    arithmetic: str = "Fraction.from_float_exact_dyadic"
    proof_authority: bool = False

    @property
    def contradiction(self) -> Fraction:
        return Fraction(
            self.contradiction_numerator,
            self.contradiction_denominator,
        )


@dataclass(frozen=True)
class PCPCCResult:
    """A non-authoritative cut candidate and its omission-firewall receipt."""

    status: str
    hz: Optional[SparseHZono]
    literals: Tuple[PhaseLiteral, ...]
    certificates: Tuple[ExactConflictCertificate, ...]
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    _live_capability: object = field(
        default=None,
        repr=False,
        compare=False,
    )


class PCPCCError(ValueError):
    """Malformed PC-PCC input; no cut may be consumed."""


_SOURCE_KINDS = {"upper", "equality_pos", "equality_neg"}
_BOUND_KINDS = {"bound_lower", "bound_upper"}
_PROCESS_KEY = secrets.token_bytes(32)
_CAPABILITY_SENTINEL = object()
_LIVE_CAPABILITY_LOCK = threading.Lock()


class _LiveCapability:
    """Opaque process-local handle for one exact live candidate result."""

    __slots__ = ("_identity",)

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise TypeError("PC-PCC capabilities are issuer-only")
        self._identity = secrets.token_hex(32)


@dataclass(frozen=True)
class _LiveResultRecord:
    capability: _LiveCapability
    result: PCPCCResult
    parent: SparseHZono
    rivals: Sequence[RivalSpec]
    receipt: Mapping[str, Any]
    hz: Optional[SparseHZono]
    literals: Tuple[PhaseLiteral, ...]
    certificates: Tuple[ExactConflictCertificate, ...]
    caps: Tuple[Tuple[str, int], ...]
    deadline: float
    issued_at: float
    process_id: int
    snapshot_hmac_sha256: str


_LIVE_RESULTS: dict[int, _LiveResultRecord] = {}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _fraction_payload(value: Fraction) -> Tuple[int, int]:
    return int(value.numerator), int(value.denominator)


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise PCPCCError(f"{name}_not_integer")
    return int(value)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _literal_binding_digest(
    *,
    parent_digest: str,
    property_digest: str,
    stable_bcol_id: int,
    phase: int,
) -> str:
    return _canonical_sha256(
        {
            "schema": "act.pc_pcc.literal.v1",
            "parent_semantic_digest": parent_digest,
            "property_digest": property_digest,
            "stable_bcol_id": int(stable_bcol_id),
            "phase": int(phase),
        }
    )


def _make_literal(
    *,
    parent_digest: str,
    property_digest: str,
    stable_bcol_id: int,
    phase: int,
) -> PhaseLiteral:
    if phase not in {-1, 1}:
        raise PCPCCError("literal_phase_not_pm1")
    return PhaseLiteral(
        stable_bcol_id=int(stable_bcol_id),
        phase=int(phase),
        binding_digest=_literal_binding_digest(
            parent_digest=parent_digest,
            property_digest=property_digest,
            stable_bcol_id=int(stable_bcol_id),
            phase=int(phase),
        ),
    )


def _ordered_pair(
    left: PhaseLiteral,
    right: PhaseLiteral,
) -> Tuple[PhaseLiteral, PhaseLiteral]:
    if left.stable_bcol_id == right.stable_bcol_id:
        raise PCPCCError("conflict_pair_repeats_literal")
    return (
        (left, right)
        if left.stable_bcol_id < right.stable_bcol_id
        else (right, left)
    )


def _pair_key(
    literals: Tuple[PhaseLiteral, PhaseLiteral],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    left, right = _ordered_pair(*literals)
    return (
        (left.stable_bcol_id, left.phase),
        (right.stable_bcol_id, right.phase),
    )


def _literal_payload(literal: PhaseLiteral) -> Mapping[str, Any]:
    return {
        "stable_bcol_id": int(literal.stable_bcol_id),
        "phase": int(literal.phase),
        "binding_digest": literal.binding_digest,
    }


def _term_payload(term: FarkasTerm) -> Mapping[str, Any]:
    return {
        "kind": term.kind,
        "index": int(term.index),
        "numerator": int(term.numerator),
        "denominator": int(term.denominator),
    }


def _certificate_payload(
    certificate: ExactConflictCertificate,
    *,
    include_digest: bool,
) -> Mapping[str, Any]:
    payload = {
        "schema": "act.pc_pcc.exact_conflict_certificate.v1",
        "literals": [
            _literal_payload(literal)
            for literal in certificate.literals
        ],
        "parent_semantic_digest": certificate.parent_semantic_digest,
        "property_digest": certificate.property_digest,
        "source_row_digests": list(certificate.source_row_digests),
        "terms": [
            _term_payload(term) for term in certificate.terms
        ],
        "contradiction_numerator": int(
            certificate.contradiction_numerator
        ),
        "contradiction_denominator": int(
            certificate.contradiction_denominator
        ),
        "arithmetic": certificate.arithmetic,
        "proof_authority": certificate.proof_authority,
    }
    if include_digest:
        payload["certificate_digest"] = certificate.certificate_digest
    return payload


def _certificate_digest(
    certificate: ExactConflictCertificate,
) -> str:
    return _canonical_sha256(
        _certificate_payload(certificate, include_digest=False)
    )


def _csr_row_fractions(
    continuous: sp.csr_matrix,
    binary: sp.csr_matrix,
    row: int,
    *,
    sign: int = 1,
) -> Tuple[Fraction, ...]:
    n_variables = int(continuous.shape[1] + binary.shape[1])
    values = [Fraction(0) for _ in range(n_variables)]
    for matrix, offset in (
        (continuous, 0),
        (binary, int(continuous.shape[1])),
    ):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        for position in range(start, stop):
            column = offset + int(matrix.indices[position])
            value = float(matrix.data[position])
            if not math.isfinite(value):
                raise PCPCCError("source_row_nonfinite")
            values[column] = (
                Fraction(sign)
                * Fraction.from_float(value)
            )
    return tuple(values)


def _source_row(
    hz: SparseHZono,
    kind: str,
    row: int,
) -> Tuple[Tuple[Fraction, ...], Fraction]:
    row = _strict_int(row, name="source_row")
    if kind == "upper":
        if row < 0 or row >= hz.n_ub:
            raise PCPCCError("upper_source_row_out_of_range")
        coefficients = _csr_row_fractions(
            hz.Auc, hz.Aub, row
        )
        rhs_value = float(hz.ub[row])
        if not math.isfinite(rhs_value):
            raise PCPCCError("upper_source_rhs_nonfinite")
        return coefficients, Fraction.from_float(rhs_value)
    if kind in {"equality_pos", "equality_neg"}:
        if row < 0 or row >= hz.n_eq:
            raise PCPCCError("equality_source_row_out_of_range")
        sign = 1 if kind == "equality_pos" else -1
        coefficients = _csr_row_fractions(
            hz.Ac, hz.Ab, row, sign=sign
        )
        rhs_value = float(hz.b[row])
        if not math.isfinite(rhs_value):
            raise PCPCCError("equality_source_rhs_nonfinite")
        return (
            coefficients,
            Fraction(sign) * Fraction.from_float(rhs_value),
        )
    raise PCPCCError("source_row_kind_invalid")


def _source_row_digest(
    kind: str,
    row: int,
    coefficients: Sequence[Fraction],
    rhs: Fraction,
) -> str:
    return _canonical_sha256(
        {
            "schema": "act.pc_pcc.source_row.v1",
            "kind": kind,
            "row": int(row),
            "coefficients": [
                list(_fraction_payload(value))
                for value in coefficients
            ],
            "rhs": list(_fraction_payload(rhs)),
        }
    )


def _stable_position_map(hz: SparseHZono) -> Mapping[int, int]:
    if hz.bcol_ids is None:
        raise PCPCCError("missing_stable_bcol_ids")
    raw = np.asarray(hz.bcol_ids)
    if (
        raw.dtype != np.dtype(np.int64)
        or raw.ndim != 1
        or int(raw.size) != hz.n_bin
        or (raw.size and np.any(raw < 0))
    ):
        raise PCPCCError("malformed_stable_bcol_ids")
    ids = tuple(int(value) for value in raw.tolist())
    if len(set(ids)) != len(ids):
        raise PCPCCError("duplicate_stable_bcol_ids")
    return {stable_id: position for position, stable_id in enumerate(ids)}


def _fixed_variable_phases(
    hz: SparseHZono,
    literals: Sequence[PhaseLiteral],
) -> Mapping[int, int]:
    positions = _stable_position_map(hz)
    fixed = {}
    for literal in literals:
        if not isinstance(literal, PhaseLiteral):
            raise PCPCCError("literal_wrong_type")
        stable_id = _strict_int(
            literal.stable_bcol_id,
            name="literal_stable_bcol_id",
        )
        phase = _strict_int(literal.phase, name="literal_phase")
        if phase not in {-1, 1}:
            raise PCPCCError("literal_phase_not_pm1")
        if stable_id not in positions:
            raise PCPCCError("literal_stable_bcol_id_missing")
        variable = hz.n_cont + positions[stable_id]
        if variable in fixed and fixed[variable] != phase:
            raise PCPCCError("literal_phase_contradiction")
        fixed[variable] = phase
    return fixed


def _variable_bounds(
    hz: SparseHZono,
    literals: Sequence[PhaseLiteral],
) -> Tuple[Tuple[Fraction, Fraction], ...]:
    fixed = _fixed_variable_phases(hz, literals)
    bounds = []
    for variable in range(hz.n_cont + hz.n_bin):
        if variable in fixed:
            phase = Fraction(fixed[variable])
            bounds.append((phase, phase))
        else:
            bounds.append((Fraction(-1), Fraction(1)))
    return tuple(bounds)


def _farkas_bound_row(
    *,
    kind: str,
    variable: int,
    bounds: Sequence[Tuple[Fraction, Fraction]],
) -> Tuple[Tuple[Fraction, ...], Fraction]:
    variable = _strict_int(variable, name="bound_variable")
    if variable < 0 or variable >= len(bounds):
        raise PCPCCError("bound_variable_out_of_range")
    coefficients = [Fraction(0) for _ in bounds]
    lower, upper = bounds[variable]
    if kind == "bound_lower":
        coefficients[variable] = Fraction(-1)
        return tuple(coefficients), -lower
    if kind == "bound_upper":
        coefficients[variable] = Fraction(1)
        return tuple(coefficients), upper
    raise PCPCCError("bound_kind_invalid")


def _certificate_from_sources(
    hz: SparseHZono,
    literals: Tuple[PhaseLiteral, PhaseLiteral],
    *,
    sources: Sequence[
        Tuple[str, int, Tuple[Fraction, ...], Fraction]
    ],
    parent_digest: str,
    property_digest: str,
    max_terms: int,
) -> Optional[ExactConflictCertificate]:
    bounds = _variable_bounds(hz, literals)
    if not sources:
        return None
    coefficients = [Fraction(0) for _ in bounds]
    rhs = Fraction(0)
    terms = []
    source_digests = []
    for kind, row, source_coefficients, source_rhs in sources:
        if len(source_coefficients) != len(coefficients):
            return None
        for index, coefficient in enumerate(source_coefficients):
            coefficients[index] += coefficient
        rhs += source_rhs
        terms.append(
            FarkasTerm(
                kind=kind,
                index=row,
                numerator=1,
                denominator=1,
            )
        )
        source_digests.append(
            _source_row_digest(
                kind, row, source_coefficients, source_rhs
            )
        )
    minimum = Fraction(0)
    for coefficient, (lower, upper) in zip(coefficients, bounds):
        minimum += coefficient * (
            lower if coefficient >= 0 else upper
        )
    if minimum <= rhs:
        return None
    for variable, coefficient in enumerate(coefficients):
        if coefficient > 0:
            numerator, denominator = _fraction_payload(coefficient)
            terms.append(
                FarkasTerm(
                    kind="bound_lower",
                    index=variable,
                    numerator=numerator,
                    denominator=denominator,
                )
            )
        elif coefficient < 0:
            numerator, denominator = _fraction_payload(-coefficient)
            terms.append(
                FarkasTerm(
                    kind="bound_upper",
                    index=variable,
                    numerator=numerator,
                    denominator=denominator,
                )
            )
    if len(terms) > max_terms:
        return None
    contradiction = rhs - minimum
    contradiction_num, contradiction_den = _fraction_payload(
        contradiction
    )
    placeholder = ExactConflictCertificate(
        literals=_ordered_pair(*literals),
        parent_semantic_digest=parent_digest,
        property_digest=property_digest,
        source_row_digests=tuple(source_digests),
        terms=tuple(terms),
        contradiction_numerator=contradiction_num,
        contradiction_denominator=contradiction_den,
        certificate_digest="",
    )
    return ExactConflictCertificate(
        **{
            **placeholder.__dict__,
            "certificate_digest": _certificate_digest(placeholder),
        }
    )


def _search_exact_bounded_farkas(
    hz: SparseHZono,
    literals: Tuple[PhaseLiteral, PhaseLiteral],
    *,
    parent_digest: str,
    property_digest: str,
    max_terms: int,
    max_source_pairs: int,
    deadline: float,
) -> Optional[ExactConflictCertificate]:
    """Find an exact one- or two-source-row Farkas contradiction.

    Each stored source row is converted from binary64 to its exact dyadic
    ``Fraction``.  Single rows are tried first.  Then bounded pairs with unit
    multipliers are added exactly; this captures conflicts whose continuous
    factors cancel only across multiple ReLU/HZ rows.
    """

    source_rows = [
        ("upper", row) for row in range(hz.n_ub)
    ]
    source_rows.extend(
        ("equality_pos", row) for row in range(hz.n_eq)
    )
    source_rows.extend(
        ("equality_neg", row) for row in range(hz.n_eq)
    )
    materialized = []
    for kind, row in source_rows:
        if time.monotonic() >= deadline:
            return None
        coefficients, rhs = _source_row(hz, kind, row)
        source = (kind, row, coefficients, rhs)
        materialized.append(source)
        certificate = _certificate_from_sources(
            hz,
            literals,
            sources=(source,),
            parent_digest=parent_digest,
            property_digest=property_digest,
            max_terms=max_terms,
        )
        if certificate is not None:
            return certificate

    pairs_seen = 0
    for left_index, left in enumerate(materialized):
        for right in materialized[left_index + 1 :]:
            if (
                pairs_seen >= max_source_pairs
                or time.monotonic() >= deadline
            ):
                return None
            pairs_seen += 1
            certificate = _certificate_from_sources(
                hz,
                literals,
                sources=(left, right),
                parent_digest=parent_digest,
                property_digest=property_digest,
                max_terms=max_terms,
            )
            if certificate is not None:
                return certificate
    return None


def verify_exact_conflict_certificate(
    hz: SparseHZono,
    certificate: ExactConflictCertificate,
    *,
    property_digest: str,
) -> bool:
    """Replay one Farkas ray entirely in exact ``Fraction`` arithmetic."""

    try:
        if (
            not isinstance(hz, SparseHZono)
            or not isinstance(certificate, ExactConflictCertificate)
            or certificate.proof_authority is not False
            or certificate.arithmetic
            != "Fraction.from_float_exact_dyadic"
            or certificate.parent_semantic_digest
            != sparse_hz_semantic_digest(hz)
            or certificate.property_digest != property_digest
            or not isinstance(certificate.source_row_digests, tuple)
            or not certificate.source_row_digests
            or len(certificate.source_row_digests) > 2
            or any(
                not _valid_sha256(digest)
                for digest in certificate.source_row_digests
            )
            or not _valid_sha256(certificate.certificate_digest)
            or _certificate_digest(certificate)
            != certificate.certificate_digest
        ):
            return False
        literals = _ordered_pair(*certificate.literals)
        if literals != certificate.literals:
            return False
        for literal in literals:
            if literal.binding_digest != _literal_binding_digest(
                parent_digest=certificate.parent_semantic_digest,
                property_digest=property_digest,
                stable_bcol_id=literal.stable_bcol_id,
                phase=literal.phase,
            ):
                return False
        bounds = _variable_bounds(hz, literals)
        n_variables = hz.n_cont + hz.n_bin
        accumulated = [Fraction(0) for _ in range(n_variables)]
        accumulated_rhs = Fraction(0)
        source_digests = []
        if not isinstance(certificate.terms, tuple):
            return False
        if not certificate.terms or len(certificate.terms) > 256:
            return False
        for term in certificate.terms:
            if not isinstance(term, FarkasTerm):
                return False
            if (
                isinstance(term.numerator, bool)
                or isinstance(term.denominator, bool)
                or not isinstance(term.numerator, int)
                or not isinstance(term.denominator, int)
                or term.denominator <= 0
                or term.numerator <= 0
            ):
                return False
            multiplier = term.multiplier
            if term.kind in _SOURCE_KINDS:
                coefficients, rhs = _source_row(
                    hz, term.kind, term.index
                )
                row_digest = _source_row_digest(
                    term.kind, term.index, coefficients, rhs
                )
                source_digests.append(row_digest)
            elif term.kind in _BOUND_KINDS:
                coefficients, rhs = _farkas_bound_row(
                    kind=term.kind,
                    variable=term.index,
                    bounds=bounds,
                )
            else:
                return False
            for index, coefficient in enumerate(coefficients):
                accumulated[index] += multiplier * coefficient
            accumulated_rhs += multiplier * rhs
        if (
            not source_digests
            or tuple(source_digests)
            != certificate.source_row_digests
            or any(value != 0 for value in accumulated)
            or accumulated_rhs >= 0
            or accumulated_rhs != certificate.contradiction
        ):
            return False
        return True
    except (
        PCPCCError,
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _derive_property_literals(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    *,
    parent_digest: str,
    property_digest: str,
) -> Tuple[PhaseLiteral, ...]:
    positions = _stable_position_map(hz)
    if not isinstance(rivals, Sequence) or not rivals:
        raise PCPCCError("rivals_empty")
    if len(rivals) > 16:
        raise PCPCCError("rival_cap_exceeded")
    objectives = []
    for rival in rivals:
        if not isinstance(rival, RivalSpec):
            raise PCPCCError("rival_wrong_type")
        objective = np.asarray(rival.objective, dtype=np.float64)
        if (
            objective.ndim != 1
            or int(objective.size) != hz.n_out
            or not np.all(np.isfinite(objective))
        ):
            raise PCPCCError("rival_objective_malformed")
        objectives.append(objective)
    literal_phases = []
    for position in range(hz.n_bin):
        generator = np.asarray(
            hz.Gb.getcol(position).toarray(),
            dtype=np.float64,
        ).reshape(-1)
        effects = [
            float(np.dot(objective, generator))
            for objective in objectives
        ]
        if any(not math.isfinite(value) for value in effects):
            raise PCPCCError("property_binary_effect_nonfinite")
        if all(value > 0.0 for value in effects):
            literal_phases.append(1)
        elif all(value < 0.0 for value in effects):
            literal_phases.append(-1)
        elif all(value == 0.0 for value in effects):
            continue
        else:
            raise PCPCCError(
                "ordered_rivals_disagree_on_literal_polarity"
            )
    stable_ids = tuple(positions)
    if len(literal_phases) != len(stable_ids):
        # Zero-effect factors are deliberately unsupported in this exact toy:
        # silently omitting one could turn an incomplete graph into a clique.
        raise PCPCCError("property_literal_does_not_cover_all_binaries")
    return tuple(
        _make_literal(
            parent_digest=parent_digest,
            property_digest=property_digest,
            stable_bcol_id=stable_id,
            phase=phase,
        )
        for stable_id, phase in zip(stable_ids, literal_phases)
    )


def _relaxed_matrices(
    hz: SparseHZono,
) -> Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, np.ndarray]:
    upper = sp.hstack([hz.Auc, hz.Aub], format="csr")
    equality = sp.hstack([hz.Ac, hz.Ab], format="csr")
    return upper, hz.ub, equality, hz.b


def _highs_pair_infeasible_candidate(
    hz: SparseHZono,
    literals: Tuple[PhaseLiteral, PhaseLiteral],
    *,
    deadline: float,
) -> bool:
    """Return only a floating candidate; exact replay is mandatory later."""

    upper, upper_rhs, equality, equality_rhs = _relaxed_matrices(hz)
    bounds = [
        (float(lower), float(upper_bound))
        for lower, upper_bound in _variable_bounds(hz, literals)
    ]
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0.0:
        return False
    result = linprog(
        np.zeros(hz.n_cont + hz.n_bin, dtype=np.float64),
        A_ub=upper,
        b_ub=upper_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={"time_limit": remaining},
    )
    return bool(not result.success and int(result.status) == 2)


def _highs_property_upper(
    hz: SparseHZono,
    rival: RivalSpec,
    *,
    deadline: float,
) -> Optional[float]:
    objective = np.asarray(rival.objective, dtype=np.float64)
    factor_objective = np.concatenate(
        [
            np.asarray(objective @ hz.Gc).reshape(-1),
            np.asarray(objective @ hz.Gb).reshape(-1),
        ]
    )
    constant = float(np.dot(objective, hz.c))
    upper, upper_rhs, equality, equality_rhs = _relaxed_matrices(hz)
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0.0:
        return None
    result = linprog(
        -factor_objective,
        A_ub=upper,
        b_ub=upper_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=[(-1.0, 1.0)] * (hz.n_cont + hz.n_bin),
        method="highs",
        options={"time_limit": remaining},
    )
    if not result.success:
        return None
    value = constant - float(result.fun)
    return value if math.isfinite(value) else None


def _copy_parent_with_clique_cut(
    hz: SparseHZono,
    literals: Sequence[PhaseLiteral],
) -> SparseHZono:
    positions = _stable_position_map(hz)
    cut_binary = np.zeros(hz.n_bin, dtype=np.float64)
    for literal in literals:
        cut_binary[positions[literal.stable_bcol_id]] = float(
            literal.phase
        )
    result = SparseHZono(
        c=np.array(hz.c, dtype=np.float64, copy=True),
        Gc=hz.Gc.copy(),
        Gb=hz.Gb.copy(),
        Ac=hz.Ac.copy(),
        Ab=hz.Ab.copy(),
        b=np.array(hz.b, dtype=np.float64, copy=True),
        Auc=sp.vstack(
            [
                hz.Auc,
                sp.csr_matrix((1, hz.n_cont), dtype=np.float64),
            ],
            format="csr",
        ),
        Aub=sp.vstack(
            [hz.Aub, sp.csr_matrix(cut_binary.reshape(1, -1))],
            format="csr",
        ),
        ub=np.concatenate(
            [
                np.array(hz.ub, dtype=np.float64, copy=True),
                np.asarray([2 - len(literals)], dtype=np.float64),
            ]
        ),
        col_ids=np.array(hz.col_ids, dtype=np.int64, copy=True),
        bcol_ids=np.array(hz.bcol_ids, dtype=np.int64, copy=True),
    )
    # Conditional child semantics are proof-relevant to the parent digest.
    # Preserve only that metadata; stale micro-RLT result receipts must not be
    # copied onto an HZ with an additional row.
    for name, value in vars(hz).items():
        if "conditional" in name.lower():
            setattr(result, name, value)
    return result


def _csr_equal(left: sp.csr_matrix, right: sp.csr_matrix) -> bool:
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _strict_safe_candidate(
    uppers: Sequence[Optional[float]],
    rivals: Sequence[RivalSpec],
) -> bool:
    if len(uppers) != len(rivals) or any(
        upper is None for upper in uppers
    ):
        return False
    for upper, rival in zip(uppers, rivals):
        threshold = float(rival.threshold)
        scale = max(1.0, abs(float(upper)), abs(threshold))
        tolerance = max(
            100.0 * np.finfo(np.float64).eps,
            1.0e-11,
        ) * scale
        if not float(upper) < threshold - tolerance:
            return False
    return True


def _seal_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def verify_pc_pcc_receipt(receipt: Mapping[str, Any]) -> bool:
    try:
        if not isinstance(receipt, Mapping):
            return False
        expected = receipt["receipt_sha256"]
        if not _valid_sha256(expected):
            return False
        payload = dict(receipt)
        del payload["receipt_sha256"]
        return _canonical_sha256(payload) == expected
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _build_receipt(
    *,
    status: str,
    reason: str,
    parent_digest: str,
    property_digest: str,
    literals: Sequence[PhaseLiteral],
    pair_records: Sequence[Mapping[str, Any]],
    certificates: Sequence[ExactConflictCertificate],
    max_literals: int,
    max_pairs: int,
    max_highs_calls: int,
    max_exact_terms: int,
    deadline_respected: bool,
    cut_hz: Optional[SparseHZono],
    parent_n_ub: int,
    pre_uppers: Sequence[Optional[float]],
    post_uppers: Sequence[Optional[float]],
) -> Mapping[str, Any]:
    expected_pairs = len(literals) * (len(literals) - 1) // 2
    highs_calls = len(pair_records)
    certified = sum(
        record.get("status") == "certified_conflict"
        for record in pair_records
    )
    feasible = sum(
        record.get("status") == "highs_feasible_or_unknown"
        for record in pair_records
    )
    exact_rejected = sum(
        record.get("status") == "exact_replay_rejected"
        for record in pair_records
    )
    unprocessed = expected_pairs - highs_calls
    complete_graph = (
        expected_pairs > 0
        and highs_calls == expected_pairs
        and certified == expected_pairs
        and len(certificates) == expected_pairs
    )
    cut_eligible = (
        complete_graph and status != "stopped_without_cut"
    )
    cut_rows_added = 0 if cut_hz is None else cut_hz.n_ub - parent_n_ub
    closure_complete = (
        expected_pairs == highs_calls + unprocessed
        and highs_calls == certified + feasible + exact_rejected
        and certified == len(certificates)
        and cut_rows_added == (1 if cut_eligible else 0)
    )
    receipt = {
        "schema": "act.pc_pcc_candidate.v1",
        "status": status,
        "reason": reason,
        "proof_authority": False,
        "intended_scope": "candidate_only_exact_controlled_toy",
        "highs_role": "candidate_generation_only",
        "exact_replay": "Fraction.from_float_exact_dyadic_farkas",
        "parent_semantic_digest": parent_digest,
        "property_digest": property_digest,
        "literals": [
            _literal_payload(literal) for literal in literals
        ],
        "literal_count": len(literals),
        "expected_pairs": expected_pairs,
        "pair_records": list(pair_records),
        "certificate_digests": [
            certificate.certificate_digest
            for certificate in certificates
        ],
        "highs_calls": highs_calls,
        "certified_conflict_edges": certified,
        "highs_feasible_or_unknown_pairs": feasible,
        "exact_replay_rejected_pairs": exact_rejected,
        "unprocessed_pairs": unprocessed,
        "complete_conflict_graph": complete_graph,
        "cut_eligible": cut_eligible,
        "parent_n_ub": int(parent_n_ub),
        "result_n_ub": (
            None if cut_hz is None else int(cut_hz.n_ub)
        ),
        "cut_rows_added": int(cut_rows_added),
        "phase_children_minted": 0,
        "branching_used": False,
        "cut_semantic_digest": (
            None
            if cut_hz is None
            else sparse_hz_semantic_digest(cut_hz)
        ),
        "pre_cut_highs_property_uppers": [
            None if value is None else float(value)
            for value in pre_uppers
        ],
        "post_cut_highs_property_uppers": [
            None if value is None else float(value)
            for value in post_uppers
        ],
        "caps": {
            "max_literals": int(max_literals),
            "max_pairs": int(max_pairs),
            "max_highs_calls": int(max_highs_calls),
            "max_exact_terms": int(max_exact_terms),
            "max_exact_source_pairs": min(
                int(max_pairs) * 64, 8192
            ),
        },
        "deadline_respected": bool(deadline_respected),
        "closure": {
            "expected_equals_processed_plus_unprocessed": (
                expected_pairs == highs_calls + unprocessed
            ),
            "processed_status_partition_complete": (
                highs_calls == certified + feasible + exact_rejected
            ),
            "edge_certificate_bijection": (
                certified == len(certificates)
            ),
            "cut_row_conservation": (
                cut_rows_added == (1 if cut_eligible else 0)
            ),
            "complete": closure_complete,
        },
    }
    return _seal_receipt(receipt)


def _live_snapshot_payload(
    *,
    result: PCPCCResult,
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    caps: Mapping[str, int],
    deadline: float,
    issued_at: float,
) -> Mapping[str, Any]:
    """Bind every live object and invocation fact behind the audit receipt."""

    cut_digest = (
        None
        if result.hz is None
        else sparse_hz_semantic_digest(result.hz)
    )
    return {
        "schema": "act.pc_pcc.live_candidate.v1",
        "process_id": os.getpid(),
        "result_object_id": id(result),
        "receipt_object_id": id(result.receipt),
        "cut_object_id": (
            None if result.hz is None else id(result.hz)
        ),
        "literal_tuple_object_id": id(result.literals),
        "certificate_tuple_object_id": id(result.certificates),
        "parent_object_id": id(parent),
        "rivals_object_id": id(rivals),
        "parent_semantic_digest": sparse_hz_semantic_digest(parent),
        "property_digest": ordered_property_digest(rivals),
        "caps": {
            key: int(value) for key, value in sorted(caps.items())
        },
        "deadline_hex": float(deadline).hex(),
        "issued_at_hex": float(issued_at).hex(),
        "status": result.status,
        "proof_authority": result.proof_authority,
        "receipt_live_sha256": _canonical_sha256(result.receipt),
        "literal_bindings": [
            _literal_payload(literal) for literal in result.literals
        ],
        "certificate_digests": [
            certificate.certificate_digest
            for certificate in result.certificates
        ],
        "cut_semantic_digest": cut_digest,
        "cut_receipt_object_id": (
            None
            if result.hz is None
            else id(
                getattr(
                    result.hz,
                    "_pc_pcc_candidate_receipt",
                    None,
                )
            )
        ),
    }


def _issue_live_result(
    *,
    result: PCPCCResult,
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    caps: Mapping[str, int],
    deadline: float,
) -> PCPCCResult:
    """Issue one process-local, single-use candidate audit capability."""

    capability = _LiveCapability(_CAPABILITY_SENTINEL)
    live_result = PCPCCResult(
        status=result.status,
        hz=result.hz,
        literals=result.literals,
        certificates=result.certificates,
        receipt=result.receipt,
        proof_authority=False,
        _live_capability=capability,
    )
    if not verify_pc_pcc_structural_result(
        parent, rivals, live_result
    ):
        raise PCPCCError("live_issue_structural_audit_failed")
    issued_at = time.monotonic()
    if (
        live_result.status != "stopped_without_cut"
        and issued_at >= float(deadline)
    ):
        raise PCPCCError("deadline_expired_before_live_issue")
    snapshot = _live_snapshot_payload(
        result=live_result,
        parent=parent,
        rivals=rivals,
        caps=caps,
        deadline=deadline,
        issued_at=issued_at,
    )
    snapshot_mac = hmac.new(
        _PROCESS_KEY,
        _canonical_bytes(snapshot),
        digestmod=hashlib.sha256,
    ).hexdigest()
    record = _LiveResultRecord(
        capability=capability,
        result=live_result,
        parent=parent,
        rivals=rivals,
        receipt=live_result.receipt,
        hz=live_result.hz,
        literals=live_result.literals,
        certificates=live_result.certificates,
        caps=tuple(
            (key, int(value))
            for key, value in sorted(caps.items())
        ),
        deadline=float(deadline),
        issued_at=issued_at,
        process_id=os.getpid(),
        snapshot_hmac_sha256=snapshot_mac,
    )
    with _LIVE_CAPABILITY_LOCK:
        _LIVE_RESULTS[id(capability)] = record
    return live_result


def _consume_live_result(
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PCPCCResult,
) -> bool:
    """Consume and validate the exact live result; saved copies always fail."""

    if not isinstance(result, PCPCCResult):
        return False
    capability = result._live_capability
    with _LIVE_CAPABILITY_LOCK:
        record = _LIVE_RESULTS.pop(id(capability), None)
    if (
        record is None
        or not isinstance(capability, _LiveCapability)
        or record.capability is not capability
        or record.result is not result
        or record.parent is not parent
        or record.rivals is not rivals
        or record.receipt is not result.receipt
        or record.hz is not result.hz
        or record.literals is not result.literals
        or record.certificates is not result.certificates
        or record.process_id != os.getpid()
    ):
        return False
    try:
        caps = dict(record.caps)
        snapshot = _live_snapshot_payload(
            result=result,
            parent=parent,
            rivals=rivals,
            caps=caps,
            deadline=record.deadline,
            issued_at=record.issued_at,
        )
        live_mac = hmac.new(
            _PROCESS_KEY,
            _canonical_bytes(snapshot),
            digestmod=hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(
            live_mac, record.snapshot_hmac_sha256
        ):
            return False
        return verify_pc_pcc_structural_result(
            parent, rivals, result
        )
    except (
        PCPCCError,
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _stopped_result(
    *,
    reason: str,
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    parent_digest: str,
    property_digest: str,
    literals: Sequence[PhaseLiteral],
    pair_records: Sequence[Mapping[str, Any]],
    certificates: Sequence[ExactConflictCertificate],
    caps: Mapping[str, int],
    deadline: float,
    pre_uppers: Sequence[Optional[float]],
    deadline_respected: bool = False,
) -> PCPCCResult:
    receipt = _build_receipt(
        status="stopped_without_cut",
        reason=reason,
        parent_digest=parent_digest,
        property_digest=property_digest,
        literals=literals,
        pair_records=pair_records,
        certificates=certificates,
        max_literals=caps["max_literals"],
        max_pairs=caps["max_pairs"],
        max_highs_calls=caps["max_highs_calls"],
        max_exact_terms=caps["max_exact_terms"],
        deadline_respected=bool(deadline_respected),
        cut_hz=None,
        parent_n_ub=parent.n_ub,
        pre_uppers=pre_uppers,
        post_uppers=(),
    )
    raw_result = PCPCCResult(
        status="stopped_without_cut",
        hz=None,
        literals=tuple(literals),
        certificates=tuple(certificates),
        receipt=receipt,
    )
    return _issue_live_result(
        result=raw_result,
        parent=parent,
        rivals=rivals,
        caps=caps,
        deadline=deadline,
    )


def run_pc_pcc_candidate(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    *,
    deadline: float,
    max_literals: int = 16,
    max_pairs: int = 120,
    max_highs_calls: int = 120,
    max_exact_terms: int = 64,
) -> PCPCCResult:
    """Run isolated PC-PCC edge discovery and, only for a full clique, cut."""

    try:
        if not isinstance(hz, SparseHZono):
            raise PCPCCError("parent_not_sparse_hz")
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise PCPCCError("deadline_nonfinite")
        caps = {
            "max_literals": _strict_int(
                max_literals, name="max_literals"
            ),
            "max_pairs": _strict_int(max_pairs, name="max_pairs"),
            "max_highs_calls": _strict_int(
                max_highs_calls, name="max_highs_calls"
            ),
            "max_exact_terms": _strict_int(
                max_exact_terms, name="max_exact_terms"
            ),
        }
        if any(value < 1 for value in caps.values()):
            raise PCPCCError("cap_must_be_positive")
        if (
            caps["max_literals"] > 64
            or caps["max_pairs"] > 2016
            or caps["max_highs_calls"] > 2016
            or caps["max_exact_terms"] > 256
        ):
            raise PCPCCError("cap_exceeds_candidate_hard_limit")
        parent_digest = sparse_hz_semantic_digest(hz)
        property_digest = ordered_property_digest(rivals)
        literals = _derive_property_literals(
            hz,
            rivals,
            parent_digest=parent_digest,
            property_digest=property_digest,
        )
        expected_pairs = len(literals) * (len(literals) - 1) // 2
        pre_uppers: Tuple[Optional[float], ...] = ()
        pair_records = []
        certificates = []

        early_reason = None
        if len(literals) < 2:
            early_reason = "fewer_than_two_property_literals"
        elif len(literals) > caps["max_literals"]:
            early_reason = "literal_cap_exceeded"
        elif expected_pairs > caps["max_pairs"]:
            early_reason = "pair_cap_exceeded"
        elif expected_pairs > caps["max_highs_calls"]:
            early_reason = "highs_call_cap_exceeded"
        elif time.monotonic() >= deadline:
            early_reason = "deadline_expired_before_telemetry"
        if early_reason is not None:
            return _stopped_result(
                reason=early_reason,
                parent=hz,
                rivals=rivals,
                parent_digest=parent_digest,
                property_digest=property_digest,
                literals=literals,
                pair_records=pair_records,
                certificates=certificates,
                caps=caps,
                deadline=deadline,
                pre_uppers=pre_uppers,
                deadline_respected=not early_reason.startswith(
                    "deadline_"
                ),
            )

        telemetry = []
        for rival in rivals:
            telemetry.append(
                _highs_property_upper(
                    hz, rival, deadline=deadline
                )
            )
            if time.monotonic() >= deadline:
                return _stopped_result(
                    reason="deadline_expired_during_pre_cut_telemetry",
                    parent=hz,
                    rivals=rivals,
                    parent_digest=parent_digest,
                    property_digest=property_digest,
                    literals=literals,
                    pair_records=pair_records,
                    certificates=certificates,
                    caps=caps,
                    deadline=deadline,
                    pre_uppers=tuple(telemetry),
                )
        pre_uppers = tuple(telemetry)

        for left_index, left in enumerate(literals):
            for right in literals[left_index + 1 :]:
                if time.monotonic() >= deadline:
                    return _stopped_result(
                        reason="deadline_expired_during_pairs",
                        parent=hz,
                        rivals=rivals,
                        parent_digest=parent_digest,
                        property_digest=property_digest,
                        literals=literals,
                        pair_records=pair_records,
                        certificates=certificates,
                        caps=caps,
                        deadline=deadline,
                        pre_uppers=pre_uppers,
                    )
                pair = _ordered_pair(left, right)
                record = {
                    "literals": [
                        _literal_payload(literal)
                        for literal in pair
                    ],
                    "status": "highs_feasible_or_unknown",
                    "certificate_digest": None,
                }
                highs_infeasible = _highs_pair_infeasible_candidate(
                    hz, pair, deadline=deadline
                )
                if time.monotonic() >= deadline:
                    pair_records.append(record)
                    return _stopped_result(
                        reason="deadline_expired_in_pair_highs",
                        parent=hz,
                        rivals=rivals,
                        parent_digest=parent_digest,
                        property_digest=property_digest,
                        literals=literals,
                        pair_records=pair_records,
                        certificates=certificates,
                        caps=caps,
                        deadline=deadline,
                        pre_uppers=pre_uppers,
                    )
                if highs_infeasible:
                    certificate = _search_exact_bounded_farkas(
                        hz,
                        pair,
                        parent_digest=parent_digest,
                        property_digest=property_digest,
                        max_terms=caps["max_exact_terms"],
                        max_source_pairs=min(
                            caps["max_pairs"] * 64, 8192
                        ),
                        deadline=deadline,
                    )
                    if time.monotonic() >= deadline:
                        pair_records.append(record)
                        return _stopped_result(
                            reason="deadline_expired_in_exact_replay",
                            parent=hz,
                            rivals=rivals,
                            parent_digest=parent_digest,
                            property_digest=property_digest,
                            literals=literals,
                            pair_records=pair_records,
                            certificates=certificates,
                            caps=caps,
                            deadline=deadline,
                            pre_uppers=pre_uppers,
                        )
                    if (
                        certificate is not None
                        and verify_exact_conflict_certificate(
                            hz,
                            certificate,
                            property_digest=property_digest,
                        )
                    ):
                        record["status"] = "certified_conflict"
                        record["certificate_digest"] = (
                            certificate.certificate_digest
                        )
                        certificates.append(certificate)
                    else:
                        record["status"] = "exact_replay_rejected"
                pair_records.append(record)

        if sparse_hz_semantic_digest(hz) != parent_digest:
            raise PCPCCError("parent_mutated_during_candidate")
        complete_graph = len(certificates) == expected_pairs
        cut_hz = (
            _copy_parent_with_clique_cut(hz, literals)
            if complete_graph
            else None
        )
        post_telemetry = []
        if cut_hz is not None:
            for rival in rivals:
                post_telemetry.append(
                    _highs_property_upper(
                        cut_hz, rival, deadline=deadline
                    )
                )
                if time.monotonic() >= deadline:
                    return _stopped_result(
                        reason=(
                            "deadline_expired_during_post_cut_telemetry"
                        ),
                        parent=hz,
                        rivals=rivals,
                        parent_digest=parent_digest,
                        property_digest=property_digest,
                        literals=literals,
                        pair_records=pair_records,
                        certificates=certificates,
                        caps=caps,
                        deadline=deadline,
                        pre_uppers=pre_uppers,
                    )
        post_uppers = tuple(post_telemetry)
        pre_unknown = not _strict_safe_candidate(
            pre_uppers, rivals
        )
        post_safe = (
            cut_hz is not None
            and _strict_safe_candidate(post_uppers, rivals)
        )
        if cut_hz is None:
            status = "incomplete_conflict_graph"
            reason = "not_every_literal_pair_has_exact_conflict"
        elif pre_unknown and post_safe:
            status = "unknown_to_safe_candidate"
            reason = "full_exact_conflict_clique_cut_tightens_highs"
        elif post_safe:
            status = "safe_candidate"
            reason = "full_exact_conflict_clique_cut_highs_safe"
        else:
            status = "cut_candidate"
            reason = "full_exact_conflict_clique_cut_not_highs_safe"
        receipt = _build_receipt(
            status=status,
            reason=reason,
            parent_digest=parent_digest,
            property_digest=property_digest,
            literals=literals,
            pair_records=pair_records,
            certificates=certificates,
            max_exact_terms=caps["max_exact_terms"],
            deadline_respected=True,
            cut_hz=cut_hz,
            parent_n_ub=hz.n_ub,
            pre_uppers=pre_uppers,
            post_uppers=post_uppers,
            **{
                key: value
                for key, value in caps.items()
                if key != "max_exact_terms"
            },
        )
        if cut_hz is not None:
            setattr(cut_hz, "_pc_pcc_candidate_receipt", receipt)
        result = PCPCCResult(
            status=status,
            hz=cut_hz,
            literals=literals,
            certificates=tuple(certificates),
            receipt=receipt,
        )
        if not verify_pc_pcc_structural_result(
            hz, rivals, result
        ):
            raise PCPCCError("self_audit_failed")
        if time.monotonic() >= deadline:
            return _stopped_result(
                reason="deadline_expired_during_final_self_audit",
                parent=hz,
                rivals=rivals,
                parent_digest=parent_digest,
                property_digest=property_digest,
                literals=literals,
                pair_records=pair_records,
                certificates=certificates,
                caps=caps,
                deadline=deadline,
                pre_uppers=pre_uppers,
            )
        return _issue_live_result(
            result=result,
            parent=hz,
            rivals=rivals,
            caps=caps,
            deadline=deadline,
        )
    except PCPCCError:
        raise
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise PCPCCError(f"candidate_failed:{type(exc).__name__}") from exc


_RECEIPT_KEYS = {
    "schema",
    "status",
    "reason",
    "proof_authority",
    "intended_scope",
    "highs_role",
    "exact_replay",
    "parent_semantic_digest",
    "property_digest",
    "literals",
    "literal_count",
    "expected_pairs",
    "pair_records",
    "certificate_digests",
    "highs_calls",
    "certified_conflict_edges",
    "highs_feasible_or_unknown_pairs",
    "exact_replay_rejected_pairs",
    "unprocessed_pairs",
    "complete_conflict_graph",
    "cut_eligible",
    "parent_n_ub",
    "result_n_ub",
    "cut_rows_added",
    "phase_children_minted",
    "branching_used",
    "cut_semantic_digest",
    "pre_cut_highs_property_uppers",
    "post_cut_highs_property_uppers",
    "caps",
    "deadline_respected",
    "closure",
    "receipt_sha256",
}
_CLOSURE_KEYS = {
    "expected_equals_processed_plus_unprocessed",
    "processed_status_partition_complete",
    "edge_certificate_bijection",
    "cut_row_conservation",
    "complete",
}


def _validated_telemetry(
    receipt: Mapping[str, Any],
    key: str,
) -> Tuple[Optional[float], ...]:
    values = receipt.get(key)
    if not isinstance(values, list):
        raise PCPCCError(f"{key}_not_list")
    normalized = []
    for value in values:
        if value is None:
            normalized.append(None)
        elif type(value) is float and math.isfinite(value):
            normalized.append(value)
        else:
            raise PCPCCError(f"{key}_value_malformed")
    return tuple(normalized)


def _stopped_reason_matches(
    *,
    reason: str,
    literal_count: int,
    expected_pairs: int,
    record_count: int,
    certificate_count: int,
    pre_count: int,
    rival_count: int,
    complete_graph: bool,
    caps: Mapping[str, int],
) -> bool:
    """Validate all facts observable without trusting wall-clock telemetry."""

    no_work = (
        record_count == 0
        and certificate_count == 0
        and pre_count == 0
    )
    within_caps = (
        literal_count <= caps["max_literals"]
        and expected_pairs <= caps["max_pairs"]
        and expected_pairs <= caps["max_highs_calls"]
    )
    if reason == "fewer_than_two_property_literals":
        return literal_count < 2 and no_work
    if reason == "literal_cap_exceeded":
        return literal_count >= 2 and (
            literal_count > caps["max_literals"] and no_work
        )
    if reason == "pair_cap_exceeded":
        return (
            literal_count <= caps["max_literals"]
            and expected_pairs > caps["max_pairs"]
            and no_work
        )
    if reason == "highs_call_cap_exceeded":
        return (
            literal_count <= caps["max_literals"]
            and expected_pairs <= caps["max_pairs"]
            and expected_pairs > caps["max_highs_calls"]
            and no_work
        )
    if not reason.startswith("deadline_") or not within_caps:
        return False
    if reason == "deadline_expired_before_telemetry":
        return no_work
    if reason == "deadline_expired_during_pre_cut_telemetry":
        return (
            record_count == 0
            and certificate_count == 0
            and 1 <= pre_count <= rival_count
        )
    if reason == "deadline_expired_during_pairs":
        return (
            pre_count == rival_count
            and record_count < expected_pairs
        )
    if reason in {
        "deadline_expired_in_pair_highs",
        "deadline_expired_in_exact_replay",
    }:
        return (
            pre_count == rival_count
            and 1 <= record_count <= expected_pairs
        )
    if reason in {
        "deadline_expired_during_post_cut_telemetry",
        "deadline_expired_during_final_self_audit",
    }:
        return (
            pre_count == rival_count
            and record_count == expected_pairs
            and complete_graph
        )
    return False


def verify_pc_pcc_structural_result(
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PCPCCResult,
) -> bool:
    """Repeatably validate cut soundness and serialized structural closure.

    This function proves no run provenance: a receipt checksum is not a
    signature, and floating HiGHS telemetry is diagnostic.  Call
    :func:`verify_pc_pcc_result` to consume the original live result and its
    process-local invocation binding.
    """

    try:
        if (
            not isinstance(parent, SparseHZono)
            or not isinstance(result, PCPCCResult)
            or type(result.receipt) is not dict
            or type(result.literals) is not tuple
            or type(result.certificates) is not tuple
            or result.proof_authority is not False
            or not verify_pc_pcc_receipt(result.receipt)
        ):
            return False
        receipt = result.receipt
        if (
            set(receipt) != _RECEIPT_KEYS
            or receipt.get("schema") != "act.pc_pcc_candidate.v1"
            or receipt.get("proof_authority") is not False
            or receipt.get("intended_scope")
            != "candidate_only_exact_controlled_toy"
            or receipt.get("highs_role")
            != "candidate_generation_only"
            or receipt.get("exact_replay")
            != "Fraction.from_float_exact_dyadic_farkas"
            or receipt.get("status") != result.status
            or type(receipt.get("reason")) is not str
            or result.status
            not in {
                "unknown_to_safe_candidate",
                "safe_candidate",
                "cut_candidate",
                "incomplete_conflict_graph",
                "stopped_without_cut",
            }
            or receipt.get("parent_semantic_digest")
            != sparse_hz_semantic_digest(parent)
            or receipt.get("parent_n_ub") != parent.n_ub
        ):
            return False
        caps = receipt.get("caps")
        cap_limits = {
            "max_literals": 64,
            "max_pairs": 2016,
            "max_highs_calls": 2016,
            "max_exact_terms": 256,
        }
        if (
            type(caps) is not dict
            or set(caps)
            != {*cap_limits, "max_exact_source_pairs"}
        ):
            return False
        for key, hard_limit in cap_limits.items():
            value = caps.get(key)
            if (
                type(value) is not int
                or value < 1
                or value > hard_limit
            ):
                return False
        if caps.get("max_exact_source_pairs") != min(
            caps["max_pairs"] * 64, 8192
        ):
            return False
        pre_uppers = _validated_telemetry(
            receipt, "pre_cut_highs_property_uppers"
        )
        post_uppers = _validated_telemetry(
            receipt, "post_cut_highs_property_uppers"
        )
        property_digest = ordered_property_digest(rivals)
        if receipt.get("property_digest") != property_digest:
            return False
        expected_literals = _derive_property_literals(
            parent,
            rivals,
            parent_digest=receipt["parent_semantic_digest"],
            property_digest=property_digest,
        )
        if (
            result.literals != expected_literals
            or receipt.get("literals")
            != [
                _literal_payload(literal)
                for literal in expected_literals
            ]
            or receipt.get("literal_count")
            != len(expected_literals)
        ):
            return False
        expected_pair_literals = {
            _pair_key((left, right)): _ordered_pair(left, right)
            for index, left in enumerate(expected_literals)
            for right in expected_literals[index + 1 :]
        }
        expected_pair_sequence = tuple(expected_pair_literals)
        expected_pairs = set(expected_pair_sequence)
        if receipt.get("expected_pairs") != len(expected_pairs):
            return False
        certificate_pairs = []
        certificate_digests = []
        for certificate in result.certificates:
            if (
                not isinstance(
                    certificate, ExactConflictCertificate
                )
                or len(certificate.terms)
                > caps["max_exact_terms"]
                or len(certificate.source_row_digests) > 2
            ):
                return False
            certificate_pair = _pair_key(certificate.literals)
            if (
                certificate_pair not in expected_pair_literals
                or certificate.literals
                != expected_pair_literals[certificate_pair]
                or not verify_exact_conflict_certificate(
                    parent,
                    certificate,
                    property_digest=property_digest,
                )
            ):
                return False
            certificate_pairs.append(certificate_pair)
            certificate_digests.append(
                certificate.certificate_digest
            )
        if (
            len(set(certificate_pairs)) != len(certificate_pairs)
            or receipt.get("certificate_digests")
            != certificate_digests
        ):
            return False
        records = receipt.get("pair_records")
        if not isinstance(records, list):
            return False
        record_pairs = []
        record_certificates = []
        certified_record_pairs = []
        valid_statuses = {
            "certified_conflict",
            "highs_feasible_or_unknown",
            "exact_replay_rejected",
        }
        by_id = {
            literal.stable_bcol_id: literal
            for literal in expected_literals
        }
        for record in records:
            if (
                type(record) is not dict
                or set(record)
                != {"literals", "status", "certificate_digest"}
                or record.get("status") not in valid_statuses
                or not isinstance(record.get("literals"), list)
                or len(record["literals"]) != 2
            ):
                return False
            literal_payloads = record["literals"]
            try:
                pair = _ordered_pair(
                    *tuple(
                        by_id[
                            _strict_int(
                                payload["stable_bcol_id"],
                                name="record_literal_id",
                            )
                        ]
                        for payload in literal_payloads
                    )
                )
            except (KeyError, PCPCCError):
                return False
            if literal_payloads != [
                _literal_payload(literal) for literal in pair
            ]:
                return False
            key = _pair_key(pair)
            record_pairs.append(key)
            if record["status"] == "certified_conflict":
                digest = record.get("certificate_digest")
                if not _valid_sha256(digest):
                    return False
                certified_record_pairs.append(key)
                record_certificates.append(digest)
            elif record.get("certificate_digest") is not None:
                return False
        if (
            tuple(record_pairs)
            != expected_pair_sequence[: len(record_pairs)]
            or tuple(certified_record_pairs)
            != tuple(certificate_pairs)
            or record_certificates != certificate_digests
        ):
            return False
        certified = len(certificate_pairs)
        feasible_count = sum(
            record["status"] == "highs_feasible_or_unknown"
            for record in records
        )
        exact_rejected_count = sum(
            record["status"] == "exact_replay_rejected"
            for record in records
        )
        complete_graph = bool(expected_pair_sequence) and (
            tuple(certificate_pairs) == expected_pair_sequence
            and len(records) == len(expected_pair_sequence)
        )
        closure = receipt.get("closure")
        if (
            type(closure) is not dict
            or set(closure) != _CLOSURE_KEYS
            or closure.get("complete") is not True
            or closure.get(
                "expected_equals_processed_plus_unprocessed"
            )
            is not True
            or closure.get("processed_status_partition_complete")
            is not True
            or closure.get("edge_certificate_bijection") is not True
            or closure.get("cut_row_conservation") is not True
            or receipt.get("highs_calls") != len(records)
            or receipt.get("certified_conflict_edges") != certified
            or receipt.get("highs_feasible_or_unknown_pairs")
            != feasible_count
            or receipt.get("exact_replay_rejected_pairs")
            != exact_rejected_count
            or receipt.get("unprocessed_pairs")
            != len(expected_pairs) - len(records)
            or receipt.get("complete_conflict_graph")
            is not complete_graph
            or receipt.get("phase_children_minted") != 0
            or receipt.get("branching_used") is not False
            or len(records) > caps["max_pairs"]
            or len(records) > caps["max_highs_calls"]
        ):
            return False
        cut_expected = (
            complete_graph and result.status != "stopped_without_cut"
        )
        reason = receipt["reason"]
        if result.status == "stopped_without_cut":
            if (
                post_uppers
                or receipt.get("deadline_respected")
                is not (not reason.startswith("deadline_"))
                or not _stopped_reason_matches(
                    reason=reason,
                    literal_count=len(expected_literals),
                    expected_pairs=len(expected_pairs),
                    record_count=len(records),
                    certificate_count=len(result.certificates),
                    pre_count=len(pre_uppers),
                    rival_count=len(rivals),
                    complete_graph=complete_graph,
                    caps=caps,
                )
            ):
                return False
        elif result.status == "incomplete_conflict_graph":
            if (
                complete_graph
                or len(expected_literals) > caps["max_literals"]
                or len(expected_pairs) > caps["max_pairs"]
                or len(expected_pairs) > caps["max_highs_calls"]
                or len(pre_uppers) != len(rivals)
                or post_uppers
                or reason
                != "not_every_literal_pair_has_exact_conflict"
                or receipt.get("deadline_respected") is not True
            ):
                return False
        else:
            if (
                not cut_expected
                or len(expected_literals) > caps["max_literals"]
                or len(expected_pairs) > caps["max_pairs"]
                or len(expected_pairs) > caps["max_highs_calls"]
                or len(pre_uppers) != len(rivals)
                or len(post_uppers) != len(rivals)
                or receipt.get("deadline_respected") is not True
            ):
                return False
            pre_unknown = not _strict_safe_candidate(
                pre_uppers, rivals
            )
            post_safe = _strict_safe_candidate(
                post_uppers, rivals
            )
            if pre_unknown and post_safe:
                expected_status = "unknown_to_safe_candidate"
                expected_reason = (
                    "full_exact_conflict_clique_cut_tightens_highs"
                )
            elif post_safe:
                expected_status = "safe_candidate"
                expected_reason = (
                    "full_exact_conflict_clique_cut_highs_safe"
                )
            else:
                expected_status = "cut_candidate"
                expected_reason = (
                    "full_exact_conflict_clique_cut_not_highs_safe"
                )
            if (
                result.status != expected_status
                or reason != expected_reason
            ):
                return False
        if receipt.get("cut_eligible") is not cut_expected:
            return False
        if not cut_expected:
            return (
                result.hz is None
                and receipt.get("cut_rows_added") == 0
                and receipt.get("result_n_ub") is None
                and receipt.get("cut_semantic_digest") is None
            )
        cut_hz = result.hz
        expected_cut_hz = _copy_parent_with_clique_cut(
            parent, expected_literals
        )
        expected_cut_digest = sparse_hz_semantic_digest(
            expected_cut_hz
        )
        if (
            not isinstance(cut_hz, SparseHZono)
            or cut_hz.n_ub != parent.n_ub + 1
            or receipt.get("result_n_ub") != cut_hz.n_ub
            or receipt.get("cut_rows_added") != 1
            or receipt.get("cut_semantic_digest")
            != sparse_hz_semantic_digest(cut_hz)
            or sparse_hz_semantic_digest(cut_hz)
            != expected_cut_digest
            or getattr(
                cut_hz, "_pc_pcc_candidate_receipt", None
            )
            != receipt
        ):
            return False
        if (
            not np.array_equal(cut_hz.c, parent.c)
            or not np.array_equal(cut_hz.b, parent.b)
            or not np.array_equal(cut_hz.col_ids, parent.col_ids)
            or not np.array_equal(cut_hz.bcol_ids, parent.bcol_ids)
            or not _csr_equal(cut_hz.Gc, parent.Gc)
            or not _csr_equal(cut_hz.Gb, parent.Gb)
            or not _csr_equal(cut_hz.Ac, parent.Ac)
            or not _csr_equal(cut_hz.Ab, parent.Ab)
            or not _csr_equal(
                cut_hz.Auc[: parent.n_ub], parent.Auc
            )
            or not _csr_equal(
                cut_hz.Aub[: parent.n_ub], parent.Aub
            )
            or not np.array_equal(
                cut_hz.ub[: parent.n_ub], parent.ub
            )
        ):
            return False
        positions = _stable_position_map(parent)
        expected_row = np.zeros(parent.n_bin, dtype=np.float64)
        for literal in expected_literals:
            expected_row[positions[literal.stable_bcol_id]] = (
                literal.phase
            )
        live_row = np.asarray(
            cut_hz.Aub.getrow(parent.n_ub).toarray()
        ).reshape(-1)
        live_continuous = cut_hz.Auc.getrow(parent.n_ub)
        if (
            live_continuous.nnz != 0
            or not np.array_equal(live_row, expected_row)
            or float(cut_hz.ub[parent.n_ub])
            != float(2 - len(expected_literals))
        ):
            return False
        return True
    except (
        PCPCCError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_pc_pcc_result(
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PCPCCResult,
) -> bool:
    """Consume the original live candidate and validate all live bindings."""

    return _consume_live_result(parent, rivals, result)


__all__ = [
    "ExactConflictCertificate",
    "FarkasTerm",
    "PCPCCError",
    "PCPCCResult",
    "PhaseLiteral",
    "run_pc_pcc_candidate",
    "verify_exact_conflict_certificate",
    "verify_pc_pcc_receipt",
    "verify_pc_pcc_result",
    "verify_pc_pcc_structural_result",
]
