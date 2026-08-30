#!/usr/bin/env python3
# ===- property_cross_layer_residual_facet.py - joint residual toy ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Toy-only exact core for cross-layer residual facets.

The controlled residual DAG is

``x -> y=ReLU(x)`` and ``(y, -x-theta) -> z -> v=ReLU(z)``.

For the ordinary secant slopes ``alpha=1/2`` and ``gamma=1-theta``, define

``rho_y = y-alpha*x`` and ``rho_v = v-gamma*z``.

Independent ReLU triangles box these residuals separately.  This module
enumerates all four *joint* phase assignments of the original graph, projects
their exact Fraction endpoints into ``(rho_y, rho_v)``, and constructs the
convex-hull facets.  For ``theta=1/2`` one separating facet is

``rho_y - rho_v <= 1/4``.

It removes a triangle-feasible cross-layer fake point while retaining every
true graph point.  The candidate is deliberately disconnected from
``operator_hz``, solver/verifier dispatch, and benchmark configuration.  Its
receipts permanently carry ``proof_authority=False`` and
``verdict_authority=False``.

Mathematical scope
------------------
This is a compact projection of a two-ReLU phase-disjunctive hull, not a
claim to dominate an ideal complete PCOH/phase hull.  Its possible production
advantage is that one sparse, network-valid residual row can be reused by all
rivals without materializing phase binaries or one objective hull per rival.
That cost hypothesis is intentionally not asserted by this toy module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
from typing import Any, Mapping, Sequence, Tuple


Q = Fraction
_ZERO = Q(0)
_ONE = Q(1)
_HALF = Q(1, 2)
MAX_PAIR_BUDGET = 4


def _q(value: int | Fraction) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("bool is not an exact scalar")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise ValueError("toy inputs must be int or Fraction")


def _qtext(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class ResidualJoinToy:
    """One-width residual DAG with an exactly dyadic join offset."""

    theta: Fraction = Q(1, 2)
    input_lower: Fraction = Q(-1)
    input_upper: Fraction = Q(1)

    def __post_init__(self) -> None:
        theta = _q(self.theta)
        lower = _q(self.input_lower)
        upper = _q(self.input_upper)
        if lower != -1 or upper != 1:
            raise ValueError("the audited family fixes x in [-1,1]")
        if not 0 < theta < 1:
            raise ValueError("theta must lie strictly between zero and one")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "input_lower", lower)
        object.__setattr__(self, "input_upper", upper)

    @property
    def upstream_slope(self) -> Fraction:
        return _HALF

    @property
    def downstream_lower(self) -> Fraction:
        return -self.theta

    @property
    def downstream_upper(self) -> Fraction:
        return _ONE - self.theta

    @property
    def downstream_slope(self) -> Fraction:
        # u/(u-l), where l=-theta and u=1-theta.
        return _ONE - self.theta

    @property
    def downstream_intercept(self) -> Fraction:
        return self.theta * (_ONE - self.theta)

    @property
    def layer_widths(self) -> Tuple[Tuple[str, int], ...]:
        return (
            ("input:x", 1),
            ("relu:y", 1),
            ("skip:-x-theta", 1),
            ("add:z", 1),
            ("relu:v", 1),
            ("property:q", 1),
        )

    @property
    def semantic_digest(self) -> str:
        return _digest(
            {
                "schema": "act.cross_layer_residual_toy.v1",
                "theta": _qtext(self.theta),
                "input": ["-1", "1"],
                "dag": [
                    "y=relu(x)",
                    "skip=-x-theta",
                    "z=y+skip",
                    "v=relu(z)",
                    "q=-2*x+3*y-3*v",
                ],
            }
        )


@dataclass(frozen=True)
class GraphPoint:
    x: Fraction
    y: Fraction
    skip: Fraction
    z: Fraction
    v: Fraction
    q: Fraction
    rho_upstream: Fraction
    rho_downstream: Fraction


def evaluate_graph(toy: ResidualJoinToy, x: int | Fraction) -> GraphPoint:
    """Evaluate the stored rational graph at one exact input point."""

    value = _q(x)
    if not toy.input_lower <= value <= toy.input_upper:
        raise ValueError("point lies outside the input interval")
    y = max(_ZERO, value)
    skip = -value - toy.theta
    z = y + skip
    v = max(_ZERO, z)
    return GraphPoint(
        x=value,
        y=y,
        skip=skip,
        z=z,
        v=v,
        q=-2 * value + 3 * y - 3 * v,
        rho_upstream=y - toy.upstream_slope * value,
        rho_downstream=v - toy.downstream_slope * z,
    )


def exact_layer_jacobian(
    toy: ResidualJoinToy, x: int | Fraction
) -> Tuple[Tuple[str, Fraction], ...]:
    """Return the exact scalar Jacobian away from the two ReLU kinks."""

    value = _q(x)
    if value in (-toy.theta, _ZERO):
        raise ValueError("Jacobian is set-valued at a ReLU kink")
    if not toy.input_lower < value < toy.input_upper:
        raise ValueError("Jacobian probe must be in the interval interior")
    dy = _ZERO if value < 0 else _ONE
    dskip = -_ONE
    dz = dy + dskip
    point = evaluate_graph(toy, value)
    dv = dz if point.z > 0 else _ZERO
    dq = -2 + 3 * dy - 3 * dv
    return (
        ("input:x", _ONE),
        ("relu:y", dy),
        ("skip:-x-theta", dskip),
        ("add:z", dz),
        ("relu:v", dv),
        ("property:q", dq),
    )


@dataclass(frozen=True)
class PhaseProjection:
    upstream_active: bool
    downstream_active: bool
    feasible: bool
    lower: Fraction | None
    upper: Fraction | None
    endpoints: Tuple[GraphPoint, ...]


def _restrict_affine_sign(
    lower: Fraction,
    upper: Fraction,
    *,
    slope: Fraction,
    intercept: Fraction,
    nonnegative: bool,
) -> Tuple[Fraction, Fraction] | None:
    """Intersect an interval with ``slope*x+intercept >= 0`` or ``<=0``."""

    if slope == 0:
        holds = intercept >= 0 if nonnegative else intercept <= 0
        return (lower, upper) if holds else None
    root = -intercept / slope
    if (slope > 0) == nonnegative:
        lower = max(lower, root)
    else:
        upper = min(upper, root)
    return None if lower > upper else (lower, upper)


def _phase_point(
    toy: ResidualJoinToy,
    x: Fraction,
    *,
    upstream_active: bool,
    downstream_active: bool,
) -> GraphPoint:
    y = x if upstream_active else _ZERO
    skip = -x - toy.theta
    z = y + skip
    v = z if downstream_active else _ZERO
    return GraphPoint(
        x=x,
        y=y,
        skip=skip,
        z=z,
        v=v,
        q=-2 * x + 3 * y - 3 * v,
        rho_upstream=y - toy.upstream_slope * x,
        rho_downstream=v - toy.downstream_slope * z,
    )


def enumerate_joint_phase_projection(
    toy: ResidualJoinToy,
) -> Tuple[PhaseProjection, ...]:
    """Enumerate all four original-graph phase assignments exactly.

    No status label, ground truth, relaxation witness, or property threshold
    enters this construction.  Every feasible phase is a scalar interval and
    therefore projects to the segment between its two exact endpoints.
    """

    result = []
    for upstream_active, downstream_active in itertools.product(
        (False, True), repeat=2
    ):
        interval = _restrict_affine_sign(
            toy.input_lower,
            toy.input_upper,
            slope=_ONE,
            intercept=_ZERO,
            nonnegative=upstream_active,
        )
        assert interval is not None
        y_slope = _ONE if upstream_active else _ZERO
        z_slope = y_slope - _ONE
        z_intercept = -toy.theta
        interval = _restrict_affine_sign(
            interval[0],
            interval[1],
            slope=z_slope,
            intercept=z_intercept,
            nonnegative=downstream_active,
        )
        if interval is None:
            result.append(
                PhaseProjection(
                    upstream_active,
                    downstream_active,
                    False,
                    None,
                    None,
                    (),
                )
            )
            continue
        lo, hi = interval
        endpoints = tuple(
            _phase_point(
                toy,
                point,
                upstream_active=upstream_active,
                downstream_active=downstream_active,
            )
            for point in ((lo,) if lo == hi else (lo, hi))
        )
        result.append(
            PhaseProjection(
                upstream_active,
                downstream_active,
                True,
                lo,
                hi,
                endpoints,
            )
        )
    return tuple(result)


ResidualPoint = Tuple[Fraction, Fraction]


def phase_projection_vertices(
    projections: Sequence[PhaseProjection],
) -> Tuple[ResidualPoint, ...]:
    return tuple(
        sorted(
            {
                (point.rho_upstream, point.rho_downstream)
                for phase in projections
                for point in phase.endpoints
            }
        )
    )


def _cross(
    origin: ResidualPoint, left: ResidualPoint, right: ResidualPoint
) -> Fraction:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (
        left[1] - origin[1]
    ) * (right[0] - origin[0])


def exact_convex_hull_2d(
    points: Sequence[ResidualPoint],
) -> Tuple[ResidualPoint, ...]:
    """Return the counter-clockwise exact hull without collinear interiors."""

    ordered = sorted(set(points))
    if len(ordered) < 3:
        raise ValueError("a two-dimensional residual hull needs three points")
    lower: list[ResidualPoint] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[ResidualPoint] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("residual projection is collinear")
    return hull


def _primitive_positive_scale(
    a: Fraction, b: Fraction, rhs: Fraction
) -> Tuple[Fraction, Fraction, Fraction]:
    scale = next((abs(value) for value in (a, b) if value), None)
    if scale is None:
        raise ValueError("facet normal is zero")
    return a / scale, b / scale, rhs / scale


@dataclass(frozen=True)
class ResidualFacet:
    upstream_coefficient: Fraction
    downstream_coefficient: Fraction
    rhs: Fraction

    def __post_init__(self) -> None:
        raw = (
            self.upstream_coefficient,
            self.downstream_coefficient,
            self.rhs,
        )
        if any(type(value) not in {int, Fraction} for value in raw):
            raise ValueError("residual facet coefficients must be exact scalars")
        upstream, downstream, rhs = (_q(value) for value in raw)
        if upstream == 0 and downstream == 0:
            raise ValueError("residual facet normal must be nonzero")
        object.__setattr__(self, "upstream_coefficient", upstream)
        object.__setattr__(self, "downstream_coefficient", downstream)
        object.__setattr__(self, "rhs", rhs)

    def value(self, point: ResidualPoint) -> Fraction:
        return (
            self.upstream_coefficient * point[0]
            + self.downstream_coefficient * point[1]
        )

    def contains(self, point: ResidualPoint) -> bool:
        return self.value(point) <= self.rhs


def exact_hull_facets(
    hull: Sequence[ResidualPoint],
) -> Tuple[ResidualFacet, ...]:
    """Convert a counter-clockwise hull to canonical outward inequalities."""

    facets = []
    for index, first in enumerate(hull):
        second = hull[(index + 1) % len(hull)]
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        # Right/outward normal for a counter-clockwise polygon.
        a, b = dy, -dx
        rhs = a * first[0] + b * first[1]
        a, b, rhs = _primitive_positive_scale(a, b, rhs)
        facets.append(ResidualFacet(a, b, rhs))
    return tuple(facets)


def validate_facet_against_original_phases(
    toy: ResidualJoinToy, facet: ResidualFacet
) -> bool:
    """Independently check every endpoint of every feasible original phase."""

    try:
        snapshot = _snapshot_residual_facet(facet)
    except ValueError:
        return False
    return all(
        snapshot.contains((point.rho_upstream, point.rho_downstream))
        for phase in enumerate_joint_phase_projection(toy)
        for point in phase.endpoints
    )


def _residual_facet_state_is_exact(facet: Any) -> bool:
    return bool(
        type(facet) is ResidualFacet
        and type(facet.upstream_coefficient) is Fraction
        and type(facet.downstream_coefficient) is Fraction
        and type(facet.rhs) is Fraction
        and (
            facet.upstream_coefficient != 0
            or facet.downstream_coefficient != 0
        )
    )


def _snapshot_residual_facet(facet: Any) -> ResidualFacet:
    """Read caller-owned facet fields once into a private exact object."""

    if type(facet) is not ResidualFacet:
        raise ValueError("residual facet state is malformed")
    upstream = facet.upstream_coefficient
    downstream = facet.downstream_coefficient
    rhs = facet.rhs
    if not (
        type(upstream) is Fraction
        and type(downstream) is Fraction
        and type(rhs) is Fraction
        and (upstream != 0 or downstream != 0)
    ):
        raise ValueError("residual facet state is malformed")
    return ResidualFacet(upstream, downstream, rhs)


@dataclass(frozen=True)
class ExactLPResult:
    upper: Fraction
    witness: Tuple[Fraction, ...]
    vertices_checked: int


@dataclass(frozen=True)
class _LinearRow:
    coefficients: Tuple[Fraction, ...]
    rhs: Fraction
    tag: str


def _solve_square(
    rows: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> Tuple[Fraction, ...] | None:
    size = len(rows)
    if len(rhs) != size or any(len(row) != size for row in rows):
        raise ValueError("exact square system has inconsistent dimensions")
    matrix = [
        [_q(value) for value in row] + [_q(bound)]
        for row, bound in zip(rows, rhs)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if matrix[row][column]),
            None,
        )
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(size):
            if row == column or not matrix[row][column]:
                continue
            scale = matrix[row][column]
            matrix[row] = [
                left - scale * right
                for left, right in zip(matrix[row], matrix[column])
            ]
    return tuple(matrix[row][-1] for row in range(size))


def _exact_lp_max(
    rows: Sequence[_LinearRow],
    objective: Sequence[Fraction],
    constant: Fraction = _ZERO,
) -> ExactLPResult:
    dimension = len(objective)
    best: Fraction | None = None
    witness: Tuple[Fraction, ...] = ()
    checked = 0
    for selected in itertools.combinations(rows, dimension):
        point = _solve_square(
            [row.coefficients for row in selected],
            [row.rhs for row in selected],
        )
        if point is None or any(
            sum((coefficient * value for coefficient, value in zip(row.coefficients, point)), _ZERO)
            > row.rhs
            for row in rows
        ):
            continue
        checked += 1
        value = constant + sum(
            (coefficient * coordinate for coefficient, coordinate in zip(objective, point)),
            _ZERO,
        )
        if best is None or value > best:
            best, witness = value, point
    if best is None:
        raise ValueError("exact LP has no enumerated feasible vertex")
    return ExactLPResult(best, witness, checked)


def _upstream_triangle_rows() -> list[_LinearRow]:
    # Variables are (x,y,v) or (x,y,t); the third coordinate is ignored.
    return [
        _LinearRow((1, 0, 0), Q(1), "x<=1"),
        _LinearRow((-1, 0, 0), Q(1), "x>=-1"),
        _LinearRow((0, -1, 0), Q(0), "y>=0"),
        _LinearRow((0, 1, 0), Q(1), "y<=1"),
        _LinearRow((1, -1, 0), Q(0), "y>=x"),
        _LinearRow((Q(-1, 2), 1, 0), Q(1, 2), "y<=secant(x)"),
    ]


def _triangle_rows(toy: ResidualJoinToy) -> list[_LinearRow]:
    gamma = toy.downstream_slope
    rows = _upstream_triangle_rows()
    # z=y-x-theta, variables (x,y,v).
    rows.extend(
        [
            _LinearRow((0, 0, -1), Q(0), "v>=0"),
            _LinearRow((0, 0, 1), toy.downstream_upper, "v<=u"),
            _LinearRow((-1, 1, -1), toy.theta, "v>=z"),
            _LinearRow((gamma, -gamma, 1), Q(0), "v<=secant(z)"),
        ]
    )
    return rows


def _facet_row(toy: ResidualJoinToy, facet: ResidualFacet) -> _LinearRow:
    if not _residual_facet_state_is_exact(facet):
        raise ValueError("residual facet state is malformed")
    a = facet.upstream_coefficient
    b = facet.downstream_coefficient
    gamma = toy.downstream_slope
    # rho_y=y-x/2; rho_v=v-gamma*(y-x-theta).
    coefficients = (
        -a * _HALF + b * gamma,
        a - b * gamma,
        b,
    )
    rhs = facet.rhs - b * gamma * toy.theta
    return _LinearRow(coefficients, rhs, "cross_layer_residual_facet")


def exact_triangle_upper(
    toy: ResidualJoinToy,
    facets: Sequence[ResidualFacet] = (),
) -> ExactLPResult:
    rows = _triangle_rows(toy)
    snapshots = tuple(_snapshot_residual_facet(facet) for facet in facets)
    for snapshot in snapshots:
        if not validate_facet_against_original_phases(toy, snapshot):
            raise ValueError("residual facet is not valid for every original phase")
        rows.append(_facet_row(toy, snapshot))
    return _exact_lp_max(rows, (Q(-2), Q(3), Q(-3)))


def exact_downstream_rcmph_upper(toy: ResidualJoinToy) -> ExactLPResult:
    """Exact downstream two-plane hypograph over the upstream triangle.

    The two independently valid suffix planes use ``v>=0`` and ``v>=z``:

    ``f0=-2*x+3*y`` and ``f1=x+3*theta``.
    """

    rows = _upstream_triangle_rows()
    # Variables (x,y,t), maximize t with t<=f0 and t<=f1.
    rows.extend(
        [
            _LinearRow((2, -3, 1), Q(0), "t<=f0"),
            _LinearRow((-1, 0, 1), 3 * toy.theta, "t<=f1"),
        ]
    )
    return _exact_lp_max(rows, (Q(0), Q(0), Q(1)))


def exact_graph_range(toy: ResidualJoinToy) -> Tuple[Fraction, Fraction]:
    values = [
        point.q
        for phase in enumerate_joint_phase_projection(toy)
        for point in phase.endpoints
    ]
    return min(values), max(values)


@dataclass(frozen=True)
class ResidualFacetBinding:
    upstream_preactivation_stable_id: int
    upstream_output_stable_id: int
    downstream_preactivation_stable_id: int
    downstream_output_stable_id: int
    upstream_row_tag: str
    join_row_tag: str
    downstream_row_tag: str

    def __post_init__(self) -> None:
        ids = (
            self.upstream_preactivation_stable_id,
            self.upstream_output_stable_id,
            self.downstream_preactivation_stable_id,
            self.downstream_output_stable_id,
        )
        tags = (
            self.upstream_row_tag,
            self.join_row_tag,
            self.downstream_row_tag,
        )
        if (
            any(type(value) is not int or value < 0 for value in ids)
            or len(set(ids)) != len(ids)
            or any(type(tag) is not str or not tag for tag in tags)
        ):
            raise ValueError("binding needs unique stable ids and nonempty row tags")

    @property
    def semantic_digest(self) -> str:
        return _digest(
            {
                "schema": "act.cross_layer_residual_binding.v1",
                "stable_ids": [
                    self.upstream_preactivation_stable_id,
                    self.upstream_output_stable_id,
                    self.downstream_preactivation_stable_id,
                    self.downstream_output_stable_id,
                ],
                "row_tags": [
                    self.upstream_row_tag,
                    self.join_row_tag,
                    self.downstream_row_tag,
                ],
            }
        )


_CANONICAL_BINDING_IDS = (101, 102, 201, 202)
_CANONICAL_BINDING_TAGS = (
    "relu:y:pre_to_output",
    "add:z:y_plus_same_x_skip",
    "relu:v:pre_to_output",
)


def _binding_state_is_canonical(binding: Any) -> bool:
    if type(binding) is not ResidualFacetBinding:
        return False
    ids = (
        binding.upstream_preactivation_stable_id,
        binding.upstream_output_stable_id,
        binding.downstream_preactivation_stable_id,
        binding.downstream_output_stable_id,
    )
    tags = (
        binding.upstream_row_tag,
        binding.join_row_tag,
        binding.downstream_row_tag,
    )
    return bool(
        all(type(value) is int and value >= 0 for value in ids)
        and len(set(ids)) == len(ids)
        and all(type(tag) is str and bool(tag) for tag in tags)
        and ids == _CANONICAL_BINDING_IDS
        and tags == _CANONICAL_BINDING_TAGS
    )


def default_facet_binding() -> ResidualFacetBinding:
    return ResidualFacetBinding(
        upstream_preactivation_stable_id=_CANONICAL_BINDING_IDS[0],
        upstream_output_stable_id=_CANONICAL_BINDING_IDS[1],
        downstream_preactivation_stable_id=_CANONICAL_BINDING_IDS[2],
        downstream_output_stable_id=_CANONICAL_BINDING_IDS[3],
        upstream_row_tag=_CANONICAL_BINDING_TAGS[0],
        join_row_tag=_CANONICAL_BINDING_TAGS[1],
        downstream_row_tag=_CANONICAL_BINDING_TAGS[2],
    )


def _phase_payload(phases: Sequence[PhaseProjection]) -> list[Mapping[str, Any]]:
    return [
        {
            "upstream_active": phase.upstream_active,
            "downstream_active": phase.downstream_active,
            "feasible": phase.feasible,
            "interval": (
                None
                if not phase.feasible
                else [_qtext(phase.lower), _qtext(phase.upper)]  # type: ignore[arg-type]
            ),
            "residual_endpoints": [
                [_qtext(point.rho_upstream), _qtext(point.rho_downstream)]
                for point in phase.endpoints
            ],
        }
        for phase in phases
    ]


def _facet_payload(facet: ResidualFacet) -> list[str]:
    return [
        _qtext(facet.upstream_coefficient),
        _qtext(facet.downstream_coefficient),
        _qtext(facet.rhs),
    ]


@dataclass(frozen=True)
class CrossLayerResidualFacetCertificate:
    toy_digest: str
    binding_digest: str
    phases: Tuple[PhaseProjection, ...]
    hull_vertices: Tuple[ResidualPoint, ...]
    hull_facets: Tuple[ResidualFacet, ...]
    selected_facet: ResidualFacet
    relaxed_witness_residual: ResidualPoint
    selected_violation: Fraction
    receipt_sha256: str
    proof_authority: bool = False
    verdict_authority: bool = False


_GRAPH_POINT_FIELDS = (
    "x",
    "y",
    "skip",
    "z",
    "v",
    "q",
    "rho_upstream",
    "rho_downstream",
)


def _graph_point_state_is_exact(point: Any) -> bool:
    return bool(
        type(point) is GraphPoint
        and all(type(getattr(point, field)) is Fraction for field in _GRAPH_POINT_FIELDS)
    )


def _phase_projection_state_is_exact(phase: Any) -> bool:
    if not (
        type(phase) is PhaseProjection
        and type(phase.upstream_active) is bool
        and type(phase.downstream_active) is bool
        and type(phase.feasible) is bool
        and type(phase.endpoints) is tuple
        and all(_graph_point_state_is_exact(point) for point in phase.endpoints)
    ):
        return False
    if phase.feasible:
        if not (
            type(phase.lower) is Fraction
            and type(phase.upper) is Fraction
            and phase.lower <= phase.upper
        ):
            return False
        expected_x = (
            (phase.lower,)
            if phase.lower == phase.upper
            else (phase.lower, phase.upper)
        )
        return tuple(point.x for point in phase.endpoints) == expected_x
    return bool(
        phase.lower is None
        and phase.upper is None
        and phase.endpoints == ()
    )


def _residual_point_state_is_exact(point: Any) -> bool:
    return bool(
        type(point) is tuple
        and len(point) == 2
        and all(type(value) is Fraction for value in point)
    )


def _sha256_text_is_exact(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _certificate_state_is_exact(certificate: Any) -> bool:
    return bool(
        type(certificate) is CrossLayerResidualFacetCertificate
        and _sha256_text_is_exact(certificate.toy_digest)
        and _sha256_text_is_exact(certificate.binding_digest)
        and _sha256_text_is_exact(certificate.receipt_sha256)
        and type(certificate.phases) is tuple
        and len(certificate.phases) == 4
        and all(
            _phase_projection_state_is_exact(phase)
            for phase in certificate.phases
        )
        and type(certificate.hull_vertices) is tuple
        and len(certificate.hull_vertices) >= 3
        and all(
            _residual_point_state_is_exact(point)
            for point in certificate.hull_vertices
        )
        and type(certificate.hull_facets) is tuple
        and len(certificate.hull_facets) >= 3
        and all(
            _residual_facet_state_is_exact(facet)
            for facet in certificate.hull_facets
        )
        and _residual_facet_state_is_exact(certificate.selected_facet)
        and _residual_point_state_is_exact(
            certificate.relaxed_witness_residual
        )
        and type(certificate.selected_violation) is Fraction
        and type(certificate.proof_authority) is bool
        and certificate.proof_authority is False
        and type(certificate.verdict_authority) is bool
        and certificate.verdict_authority is False
    )


def _snapshot_graph_point(point: Any) -> GraphPoint:
    """Capture caller-owned graph state without retaining mutable aliases."""

    if type(point) is not GraphPoint:
        raise ValueError("graph point state is malformed")
    values = tuple(getattr(point, field) for field in _GRAPH_POINT_FIELDS)
    if any(type(value) is not Fraction for value in values):
        raise ValueError("graph point state is malformed")
    return GraphPoint(*values)


def _snapshot_phase_projection(phase: Any) -> PhaseProjection:
    """Capture one phase and every nested endpoint into private exact state."""

    if type(phase) is not PhaseProjection:
        raise ValueError("phase projection state is malformed")
    upstream_active = phase.upstream_active
    downstream_active = phase.downstream_active
    feasible = phase.feasible
    lower = phase.lower
    upper = phase.upper
    endpoints = phase.endpoints
    if (
        type(upstream_active) is not bool
        or type(downstream_active) is not bool
        or type(feasible) is not bool
        or type(endpoints) is not tuple
    ):
        raise ValueError("phase projection state is malformed")
    snapshot = PhaseProjection(
        upstream_active=upstream_active,
        downstream_active=downstream_active,
        feasible=feasible,
        lower=lower,
        upper=upper,
        endpoints=tuple(_snapshot_graph_point(point) for point in endpoints),
    )
    if not _phase_projection_state_is_exact(snapshot):
        raise ValueError("phase projection state is malformed")
    return snapshot


def _snapshot_residual_point(point: Any) -> ResidualPoint:
    """Capture one residual pair into a fresh exact tuple."""

    if type(point) is not tuple or len(point) != 2:
        raise ValueError("residual point state is malformed")
    first, second = point
    if type(first) is not Fraction or type(second) is not Fraction:
        raise ValueError("residual point state is malformed")
    return (first, second)


def _snapshot_certificate(
    certificate: Any,
) -> CrossLayerResidualFacetCertificate:
    """Deep-snapshot a certificate once at the public trust boundary."""

    if type(certificate) is not CrossLayerResidualFacetCertificate:
        raise ValueError("certificate state is malformed")

    # Read every top-level caller-owned field once.  Validation, receipt replay,
    # and equality below consume only the reconstructed private object.
    toy_digest = certificate.toy_digest
    binding_digest = certificate.binding_digest
    phases = certificate.phases
    hull_vertices = certificate.hull_vertices
    hull_facets = certificate.hull_facets
    selected_facet = certificate.selected_facet
    relaxed_witness_residual = certificate.relaxed_witness_residual
    selected_violation = certificate.selected_violation
    receipt_sha256 = certificate.receipt_sha256
    proof_authority = certificate.proof_authority
    verdict_authority = certificate.verdict_authority

    if (
        type(phases) is not tuple
        or type(hull_vertices) is not tuple
        or type(hull_facets) is not tuple
    ):
        raise ValueError("certificate state is malformed")
    snapshot = CrossLayerResidualFacetCertificate(
        toy_digest=toy_digest,
        binding_digest=binding_digest,
        phases=tuple(_snapshot_phase_projection(phase) for phase in phases),
        hull_vertices=tuple(
            _snapshot_residual_point(point) for point in hull_vertices
        ),
        hull_facets=tuple(
            _snapshot_residual_facet(facet) for facet in hull_facets
        ),
        selected_facet=_snapshot_residual_facet(selected_facet),
        relaxed_witness_residual=_snapshot_residual_point(
            relaxed_witness_residual
        ),
        selected_violation=selected_violation,
        receipt_sha256=receipt_sha256,
        proof_authority=proof_authority,
        verdict_authority=verdict_authority,
    )
    if not _certificate_state_is_exact(snapshot):
        raise ValueError("certificate state is malformed")
    return snapshot


def _certificate_payload(
    toy: ResidualJoinToy,
    binding: ResidualFacetBinding,
    phases: Sequence[PhaseProjection],
    hull: Sequence[ResidualPoint],
    facets: Sequence[ResidualFacet],
    selected: ResidualFacet,
    witness: ResidualPoint,
    violation: Fraction,
) -> Mapping[str, Any]:
    return {
        "schema": "act.cross_layer_residual_facet_certificate.v1",
        "toy_digest": toy.semantic_digest,
        "binding_digest": binding.semantic_digest,
        "complete_joint_phase_count": 4,
        "phases": _phase_payload(phases),
        "hull_vertices": [
            [_qtext(first), _qtext(second)] for first, second in hull
        ],
        "hull_facets": [_facet_payload(facet) for facet in facets],
        "selected_facet": _facet_payload(selected),
        "relaxed_witness_residual": [
            _qtext(witness[0]),
            _qtext(witness[1]),
        ],
        "selected_violation": _qtext(violation),
        "selection_has_proof_authority": False,
        "proof_authority": False,
        "verdict_authority": False,
    }


def derive_cross_layer_residual_facet(
    toy: ResidualJoinToy,
    binding: ResidualFacetBinding,
) -> CrossLayerResidualFacetCertificate:
    """Derive the most violated exact phase-hull facet at the triangle witness."""

    if not _binding_state_is_canonical(binding):
        raise ValueError("toy facet binding is not the canonical causal anchor")
    phases = enumerate_joint_phase_projection(toy)
    if len(phases) != 4:
        raise AssertionError("joint phase coverage is incomplete")
    hull = exact_convex_hull_2d(phase_projection_vertices(phases))
    facets = exact_hull_facets(hull)
    if not facets or not all(
        validate_facet_against_original_phases(toy, facet) for facet in facets
    ):
        raise AssertionError("derived hull facet excluded an original phase")
    relaxed = exact_triangle_upper(toy)
    x, y, v = relaxed.witness
    z = y - x - toy.theta
    witness = (
        y - toy.upstream_slope * x,
        v - toy.downstream_slope * z,
    )
    violations = tuple(facet.value(witness) - facet.rhs for facet in facets)
    selected_index = max(range(len(facets)), key=lambda index: (violations[index], -index))
    selected = facets[selected_index]
    violation = violations[selected_index]
    if violation <= 0:
        raise ValueError("phase projection does not separate the triangle witness")
    payload = _certificate_payload(
        toy,
        binding,
        phases,
        hull,
        facets,
        selected,
        witness,
        violation,
    )
    return CrossLayerResidualFacetCertificate(
        toy_digest=toy.semantic_digest,
        binding_digest=binding.semantic_digest,
        phases=phases,
        hull_vertices=hull,
        hull_facets=facets,
        selected_facet=selected,
        relaxed_witness_residual=witness,
        selected_violation=violation,
        receipt_sha256=_digest(payload),
    )


def validate_cross_layer_residual_facet(
    toy: ResidualJoinToy,
    binding: ResidualFacetBinding,
    certificate: CrossLayerResidualFacetCertificate,
) -> bool:
    """Re-derive phases, binding, facet coefficients, and receipt exactly."""

    if not _binding_state_is_canonical(binding):
        return False
    try:
        snapshot = _snapshot_certificate(certificate)
        if (
            snapshot.toy_digest != toy.semantic_digest
            or snapshot.binding_digest != binding.semantic_digest
        ):
            return False
        public_payload = _certificate_payload(
            toy,
            binding,
            snapshot.phases,
            snapshot.hull_vertices,
            snapshot.hull_facets,
            snapshot.selected_facet,
            snapshot.relaxed_witness_residual,
            snapshot.selected_violation,
        )
        if snapshot.receipt_sha256 != _digest(public_payload):
            return False
        expected = derive_cross_layer_residual_facet(toy, binding)
    except (AssertionError, AttributeError, TypeError, ValueError):
        return False
    return snapshot == expected


@dataclass(frozen=True)
class PairProposal:
    pair_key: Tuple[int, int]
    score: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.pair_key) is not tuple
            or len(self.pair_key) != 2
            or any(type(value) is not int or value < 0 for value in self.pair_key)
            or self.pair_key[0] >= self.pair_key[1]
            or type(self.score) not in {int, Fraction}
            or _q(self.score) < 0
        ):
            raise ValueError("pair proposal is malformed")
        object.__setattr__(self, "score", _q(self.score))


def _snapshot_pair_proposal(proposal: Any) -> PairProposal:
    """Capture one caller-owned proposal into exact immutable private state."""

    if type(proposal) is not PairProposal:
        raise ValueError("pair proposal state is malformed")
    pair_key = proposal.pair_key
    score = proposal.score
    if not (
        type(pair_key) is tuple
        and len(pair_key) == 2
        and all(type(value) is int and value >= 0 for value in pair_key)
        and pair_key[0] < pair_key[1]
        and type(score) is Fraction
        and score >= 0
    ):
        raise ValueError("pair proposal state is malformed")
    return PairProposal((pair_key[0], pair_key[1]), Fraction(score))


def bounded_pair_prefix(
    proposals: Sequence[PairProposal], budget: int
) -> Tuple[PairProposal, ...]:
    """Return a deterministic nested prefix under the hard four-pair cap."""

    if type(budget) is not int or budget < 0 or budget > MAX_PAIR_BUDGET:
        raise ValueError(f"pair budget must be in [0,{MAX_PAIR_BUDGET}]")
    snapshot = tuple(_snapshot_pair_proposal(proposal) for proposal in proposals)
    keys = [proposal.pair_key for proposal in snapshot]
    if len(set(keys)) != len(keys):
        raise ValueError("pair proposals repeat a semantic pair")
    ordered = sorted(
        snapshot,
        key=lambda proposal: (-proposal.score, proposal.pair_key),
    )
    return tuple(ordered[:budget])


def exact_budget_frontier(
    toy: ResidualJoinToy,
    binding: ResidualFacetBinding,
    certificate: CrossLayerResidualFacetCertificate,
) -> Tuple[ExactLPResult, ExactLPResult]:
    """The only authorized toy frontier: zero pairs then one selected pair."""

    if not validate_cross_layer_residual_facet(toy, binding, certificate):
        raise ValueError("facet certificate failed exact replay")
    trusted = derive_cross_layer_residual_facet(toy, binding)
    baseline = exact_triangle_upper(toy)
    tightened = exact_triangle_upper(toy, (trusted.selected_facet,))
    if tightened.upper > baseline.upper:
        raise AssertionError("adding a valid facet increased the exact LP upper")
    return baseline, tightened


def uncorrelated_join_residuals(
    toy: ResidualJoinToy,
    upstream_x: int | Fraction,
    skip_x: int | Fraction,
) -> ResidualPoint:
    """Negative-control graph after deleting the causal same-input join."""

    p, q = _q(upstream_x), _q(skip_x)
    if not -1 <= p <= 1 or not -1 <= q <= 1:
        raise ValueError("uncorrelated inputs must remain in [-1,1]")
    y = max(_ZERO, p)
    z = y - q - toy.theta
    v = max(_ZERO, z)
    return (
        y - toy.upstream_slope * p,
        v - toy.downstream_slope * z,
    )


def raw_vnnlib_margin(
    point: GraphPoint, threshold: Fraction = Q(5, 4)
) -> Fraction:
    """Return the raw unsafe assertion scalar ``q-threshold``."""

    return point.q - _q(threshold)


__all__ = [
    "CrossLayerResidualFacetCertificate",
    "ExactLPResult",
    "GraphPoint",
    "MAX_PAIR_BUDGET",
    "PairProposal",
    "PhaseProjection",
    "ResidualFacet",
    "ResidualFacetBinding",
    "ResidualJoinToy",
    "bounded_pair_prefix",
    "default_facet_binding",
    "derive_cross_layer_residual_facet",
    "enumerate_joint_phase_projection",
    "evaluate_graph",
    "exact_budget_frontier",
    "exact_convex_hull_2d",
    "exact_downstream_rcmph_upper",
    "exact_graph_range",
    "exact_hull_facets",
    "exact_layer_jacobian",
    "exact_triangle_upper",
    "phase_projection_vertices",
    "raw_vnnlib_margin",
    "uncorrelated_join_residuals",
    "validate_cross_layer_residual_facet",
    "validate_facet_against_original_phases",
]
