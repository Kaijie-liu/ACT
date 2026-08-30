#!/usr/bin/env python3
# ===- property_pairhull.py - exact two-ReLU property hull -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Independent exact core for a property-bundled two-ReLU upper plane.

This module is intentionally not connected to the production HybridZ builder.
It is a soundness/tightness gate for a possible future PairHull operator.

For two stored affine preactivations

``z = center + generators @ xi + eta``,
``xi in [-1, 1]`` and ``eta_i in [-error_i, error_i]``,

the caller supplies a finite set of directed projection templates ``d``.
Each row

``d . z <= h_d``

is formed in two steps:

1. compute the support of the *stored coefficients* exactly with
   :class:`fractions.Fraction`;
2. store that support as binary64 rounded toward positive infinity.

The exact optimizer deliberately uses the second, outward-stored bound.  It
therefore proves a statement about the polygon that would actually be handed
to downstream code, rather than a slightly tighter ideal polygon that could
be lost during float storage.

Given stored property coefficients ``q`` and a candidate slope ``a``, the
core computes

``beta = max_{z in P} q . ReLU(z) - a . z``.

It enumerates all four ReLU phases.  Within one phase the objective is affine,
so every maximum is attained at an intersection of two polygon/phase
boundaries.  All intersections, feasibility decisions, and objective values
are exact Fractions.  No floating-point LP status and no candidate heuristic
has proof authority.  Only the final exact beta is rounded toward positive
infinity for storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


FractionPair = Tuple[Fraction, Fraction]

DEFAULT_PAIRHULL_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
)

_ZERO = Fraction(0)
_ONE = Fraction(1)


class PropertyPairHullError(ValueError):
    """The requested exact PairHull problem is invalid or unbounded."""


@dataclass(frozen=True)
class PairHullProjection:
    """A bounded, sound two-dimensional projection polygon.

    ``supports`` are the exact rational values of ``stored_supports`` and
    therefore define the polygon optimized by :func:`exact_pairhull_beta`.
    ``required_supports`` retain the tighter exact supports before outward
    storage so callers can audit every rounding step.
    """

    center: FractionPair
    generators: Tuple[Tuple[Fraction, ...], Tuple[Fraction, ...]]
    error: FractionPair
    directions: Tuple[FractionPair, ...]
    required_supports: Tuple[Fraction, ...]
    stored_supports: Tuple[float, ...]
    supports: Tuple[Fraction, ...]
    source_affine_sha256: str
    constraints_sha256: str

    @property
    def constraints(self) -> Tuple[Tuple[Fraction, Fraction, Fraction], ...]:
        """Return exact rows ``(d0, d1, h)`` defining this polygon."""

        return tuple(
            (direction[0], direction[1], support)
            for direction, support in zip(self.directions, self.supports)
        )


@dataclass(frozen=True)
class PairHullResult:
    """Exact optimum, outward-stored beta, witness, and audit receipt."""

    beta_exact: Fraction
    beta_stored: float
    witness: FractionPair
    phase: Tuple[bool, bool]
    receipt: Dict[str, Any]


def _fraction(value: Any, *, name: str) -> Fraction:
    """Convert a rational or a finite stored real to an exact Fraction."""

    if isinstance(value, bool):
        raise PropertyPairHullError(f"{name}: bool is not a coefficient")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Real):
        stored = float(value)
        if not math.isfinite(stored):
            raise PropertyPairHullError(f"{name}: coefficient must be finite")
        return Fraction.from_float(stored)
    raise PropertyPairHullError(
        f"{name}: expected Fraction, integer, or finite stored real"
    )


def _pair(values: Sequence[Any], *, name: str) -> FractionPair:
    if len(values) != 2:
        raise PropertyPairHullError(f"{name}: expected exactly two values")
    return (
        _fraction(values[0], name=f"{name}[0]"),
        _fraction(values[1], name=f"{name}[1]"),
    )


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _rational_pair_payload(value: FractionPair) -> Tuple[str, str]:
    return (_rational_text(value[0]), _rational_text(value[1]))


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _outward_float(value: Fraction) -> float:
    """Store a finite Fraction as binary64 rounded toward ``+inf``."""

    try:
        stored = float(value)
    except OverflowError:
        stored = math.inf if value > 0 else -math.inf

    if stored == math.inf:
        return stored
    if stored == -math.inf:
        # A negative overflow can still be stored soundly as the largest
        # finite negative binary64, which is greater than the exact value.
        return -float.fromhex("0x1.fffffffffffffp+1023")
    if Fraction.from_float(stored) < value:
        stored = math.nextafter(stored, math.inf)
    if Fraction.from_float(stored) < value:
        raise PropertyPairHullError("outward binary64 conversion failed")
    return stored


def _cross(left: FractionPair, right: FractionPair) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


def _dot(left: FractionPair, right: FractionPair) -> Fraction:
    return left[0] * right[0] + left[1] * right[1]


def _target_in_normal_cone(
    target: FractionPair,
    normals: Sequence[FractionPair],
) -> bool:
    """Whether ``target`` is a nonnegative combination of polygon normals."""

    for normal in normals:
        cross = _cross(normal, target)
        if cross == 0 and _dot(normal, target) > 0:
            return True
    for left, right in itertools.combinations(normals, 2):
        determinant = _cross(left, right)
        if determinant == 0:
            continue
        left_weight = _cross(target, right) / determinant
        right_weight = _cross(left, target) / determinant
        if left_weight >= 0 and right_weight >= 0:
            return True
    return False


def _require_bounded(normals: Sequence[FractionPair]) -> None:
    # A two-dimensional recession cone is {0} exactly when the conic hull of
    # the row normals is all of R^2.  Checking the four signed basis vectors
    # gives a small, entirely rational certificate of that fact.
    basis = (
        (_ONE, _ZERO),
        (-_ONE, _ZERO),
        (_ZERO, _ONE),
        (_ZERO, -_ONE),
    )
    if not all(_target_in_normal_cone(target, normals) for target in basis):
        raise PropertyPairHullError(
            "directed templates do not define a bounded projection polygon"
        )


def build_pairhull_projection(
    *,
    center: Sequence[Any],
    generators: Sequence[Sequence[Any]],
    error: Sequence[Any] = (0, 0),
    directions: Sequence[Sequence[Any]],
) -> PairHullProjection:
    """Build a sound polygon from a stored two-row affine form.

    Fractions and integers are accepted for exact toys.  Any other real input
    is first converted to binary64 and then interpreted as the exact stored
    coefficient via ``Fraction.from_float``.
    """

    exact_center = _pair(center, name="center")
    if len(generators) != 2:
        raise PropertyPairHullError("generators: expected exactly two rows")
    if len(generators[0]) != len(generators[1]):
        raise PropertyPairHullError("generator rows have different widths")
    exact_generators = (
        tuple(
            _fraction(value, name=f"generators[0][{column}]")
            for column, value in enumerate(generators[0])
        ),
        tuple(
            _fraction(value, name=f"generators[1][{column}]")
            for column, value in enumerate(generators[1])
        ),
    )
    exact_error = _pair(error, name="error")
    if any(value < 0 for value in exact_error):
        raise PropertyPairHullError("rowwise error radii must be nonnegative")

    if len(directions) == 0:
        raise PropertyPairHullError("at least one directed template is required")
    exact_directions = tuple(
        _pair(direction, name=f"directions[{row}]")
        for row, direction in enumerate(directions)
    )
    if any(direction == (_ZERO, _ZERO) for direction in exact_directions):
        raise PropertyPairHullError("zero projection direction is invalid")
    _require_bounded(exact_directions)

    required_supports = []
    stored_supports = []
    supports = []
    for direction in exact_directions:
        required = _dot(direction, exact_center)
        for column in range(len(exact_generators[0])):
            projected = (
                direction[0] * exact_generators[0][column]
                + direction[1] * exact_generators[1][column]
            )
            required += abs(projected)
        # The two rowwise errors are independent intervals.  They must not be
        # projected as one shared generator: opposite signs in d cannot
        # cancel independent uncertainty.
        required += (
            abs(direction[0]) * exact_error[0]
            + abs(direction[1]) * exact_error[1]
        )
        stored = _outward_float(required)
        if not math.isfinite(stored):
            raise PropertyPairHullError(
                "projection support overflowed binary64; polygon row is vacuous"
            )
        support = Fraction.from_float(stored)
        if support < required:
            raise PropertyPairHullError("projection support was not outward")
        required_supports.append(required)
        stored_supports.append(stored)
        supports.append(support)

    constraints = tuple(
        (direction[0], direction[1], support)
        for direction, support in zip(exact_directions, supports)
    )
    if any(
        direction[0] * exact_center[0]
        + direction[1] * exact_center[1]
        > support
        for direction, support in zip(exact_directions, supports)
    ):
        raise PropertyPairHullError("source center is not in its projection")

    source_payload = {
        "center": _rational_pair_payload(exact_center),
        "generators": [
            [_rational_text(value) for value in row]
            for row in exact_generators
        ],
        "error": _rational_pair_payload(exact_error),
    }
    constraint_payload = [
        {
            "direction": _rational_pair_payload(
                (direction[0], direction[1])
            ),
            "required_support": _rational_text(required),
            "stored_support_hex": stored.hex(),
            "support": _rational_text(support),
        }
        for direction, required, stored, support in zip(
            exact_directions,
            required_supports,
            stored_supports,
            supports,
        )
    ]
    return PairHullProjection(
        center=exact_center,
        generators=exact_generators,
        error=exact_error,
        directions=exact_directions,
        required_supports=tuple(required_supports),
        stored_supports=tuple(stored_supports),
        supports=tuple(supports),
        source_affine_sha256=_canonical_sha256(source_payload),
        constraints_sha256=_canonical_sha256(constraint_payload),
    )


def _intersection(
    left: Tuple[Fraction, Fraction, Fraction],
    right: Tuple[Fraction, Fraction, Fraction],
) -> Optional[FractionPair]:
    determinant = left[0] * right[1] - left[1] * right[0]
    if determinant == 0:
        return None
    return (
        (left[2] * right[1] - left[1] * right[2]) / determinant,
        (left[0] * right[2] - left[2] * right[0]) / determinant,
    )


def _is_feasible(
    point: FractionPair,
    constraints: Sequence[Tuple[Fraction, Fraction, Fraction]],
) -> bool:
    return all(
        row[0] * point[0] + row[1] * point[1] <= row[2]
        for row in constraints
    )


def _phase_constraints(
    phase: Tuple[bool, bool],
) -> Tuple[Tuple[Fraction, Fraction, Fraction], ...]:
    # active: z >= 0 -> -z <= 0; inactive: z <= 0.
    return (
        ((-_ONE if phase[0] else _ONE), _ZERO, _ZERO),
        (_ZERO, (-_ONE if phase[1] else _ONE), _ZERO),
    )


def _phase_objective(
    q: FractionPair,
    slope: FractionPair,
    phase: Tuple[bool, bool],
) -> FractionPair:
    return (
        (q[0] if phase[0] else _ZERO) - slope[0],
        (q[1] if phase[1] else _ZERO) - slope[1],
    )


def _relu_objective(
    point: FractionPair,
    q: FractionPair,
    slope: FractionPair,
) -> Fraction:
    return (
        q[0] * max(point[0], _ZERO)
        + q[1] * max(point[1], _ZERO)
        - slope[0] * point[0]
        - slope[1] * point[1]
    )


def verify_pairhull_receipt(receipt: Mapping[str, Any]) -> bool:
    """Check the deterministic checksum of a JSON-friendly receipt."""

    try:
        expected = receipt["receipt_sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            return False
        payload = dict(receipt)
        del payload["receipt_sha256"]
        return _canonical_sha256(payload) == expected
    except (KeyError, TypeError, ValueError):
        return False


def exact_pairhull_beta(
    projection: PairHullProjection,
    *,
    q: Sequence[Any],
    candidate_slope: Sequence[Any],
) -> PairHullResult:
    """Compute the exact PairHull intercept over ``projection``.

    The candidate slope may have been proposed by arbitrary float/GPU code.
    It has no proof authority: its stored coefficients are converted to exact
    Fractions here, and the intercept is recomputed by exhaustive rational
    phase/vertex enumeration.
    """

    exact_q = _pair(q, name="q")
    exact_slope = _pair(candidate_slope, name="candidate_slope")
    polygon_constraints = projection.constraints
    _require_bounded(projection.directions)

    phase_records = []
    optima = []
    intersections_checked = 0
    unique_vertices_checked = 0
    for phase in itertools.product((False, True), repeat=2):
        phase_tuple = (bool(phase[0]), bool(phase[1]))
        constraints = polygon_constraints + _phase_constraints(phase_tuple)
        vertices = set()
        for left, right in itertools.combinations(constraints, 2):
            intersections_checked += 1
            point = _intersection(left, right)
            if point is not None and _is_feasible(point, constraints):
                vertices.add(point)
        unique_vertices_checked += len(vertices)
        if not vertices:
            phase_records.append(
                {
                    "phase": "".join("1" if value else "0" for value in phase),
                    "feasible": False,
                    "vertices": 0,
                    "maximum": None,
                    "witness": None,
                }
            )
            continue

        objective = _phase_objective(exact_q, exact_slope, phase_tuple)
        values = {
            point: objective[0] * point[0] + objective[1] * point[1]
            for point in vertices
        }
        maximum = max(values.values())
        witness = min(
            point for point, value in values.items() if value == maximum
        )
        direct = _relu_objective(witness, exact_q, exact_slope)
        if direct != maximum:
            raise PropertyPairHullError(
                "phase objective disagrees with direct ReLU evaluation"
            )
        phase_text = "".join("1" if value else "0" for value in phase)
        phase_records.append(
            {
                "phase": phase_text,
                "feasible": True,
                "vertices": len(vertices),
                "maximum": _rational_text(maximum),
                "witness": list(_rational_pair_payload(witness)),
            }
        )
        optima.append((maximum, phase_text, witness, phase_tuple))

    if not optima:
        raise PropertyPairHullError("projection has no feasible ReLU phase")
    beta_exact = max(entry[0] for entry in optima)
    selected = min(
        (
            (phase_text, witness, phase_tuple)
            for maximum, phase_text, witness, phase_tuple in optima
            if maximum == beta_exact
        ),
        key=lambda entry: (entry[0], entry[1]),
    )
    selected_phase_text, witness, selected_phase = selected
    beta_stored = _outward_float(beta_exact)
    if math.isfinite(beta_stored):
        if Fraction.from_float(beta_stored) < beta_exact:
            raise PropertyPairHullError("stored beta is not outward")
    elif beta_stored != math.inf:
        raise PropertyPairHullError("invalid infinite stored beta")

    q_payload = list(_rational_pair_payload(exact_q))
    slope_payload = list(_rational_pair_payload(exact_slope))
    receipt: Dict[str, Any] = {
        "schema": "act.property_pairhull.exact.v1",
        "algorithm": "four_phase_fraction_boundary_intersections",
        "proof_authority": "exact_fraction_phase_vertex_enumeration",
        "candidate_slope_proof_authority": False,
        "float_lp_proof_authority": False,
        "projection_uses_outward_stored_supports": True,
        "source_affine_sha256": projection.source_affine_sha256,
        "constraints_sha256": projection.constraints_sha256,
        "q": q_payload,
        "q_sha256": _canonical_sha256(q_payload),
        "candidate_slope": slope_payload,
        "candidate_slope_sha256": _canonical_sha256(slope_payload),
        "directions": len(projection.directions),
        "phases_total": 4,
        "phases_feasible": sum(
            bool(record["feasible"]) for record in phase_records
        ),
        "intersections_checked": intersections_checked,
        "unique_vertices_checked": unique_vertices_checked,
        "phase_records": phase_records,
        "phase_records_sha256": _canonical_sha256(phase_records),
        "beta_exact": _rational_text(beta_exact),
        "beta_stored_hex": beta_stored.hex(),
        "selected_phase": selected_phase_text,
        "selected_witness": list(_rational_pair_payload(witness)),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return PairHullResult(
        beta_exact=beta_exact,
        beta_stored=beta_stored,
        witness=witness,
        phase=selected_phase,
        receipt=receipt,
    )


__all__ = [
    "DEFAULT_PAIRHULL_DIRECTIONS",
    "PairHullProjection",
    "PairHullResult",
    "PropertyPairHullError",
    "build_pairhull_projection",
    "exact_pairhull_beta",
    "verify_pairhull_receipt",
]
