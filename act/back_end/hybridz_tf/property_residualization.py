"""Proof-oriented prototype for property-aware ReLU residualization.

This module is intentionally not wired into the official HybridZ path yet.
It isolates the small mathematical transformation that should be audited
before it is allowed to affect a benchmark:

``y = relu(x)`` with a proven ``l < 0 < u`` is rewritten as

``y = slope * x + rho``, ``rho in [0, residual_upper]``.

For any ``slope in [0, 1]`` the residual interval is exact:

``residual_upper = max(-slope*l, (1-slope)*u)``.

The usual secant slope minimizes that interval.  One residual variable is
allocated *per ReLU node*, rather than per use, so every fanout observes the
same ``rho``.  The secant upper predicate is then implicit.  Either, both, or
neither of the two lower predicates (``y >= 0`` and ``y >= x``) may be kept.
Dropping a valid predicate only enlarges the abstraction, so all four modes
remain sound; they trade coupling rows for tightness.

The rest of this file is a deliberately tiny, exact-arithmetic audit harness:

* a scalar residual DAG with one rational input interval;
* exhaustive ReLU-phase enumeration using :class:`fractions.Fraction`;
* an exact vertex-enumeration LP oracle for the continuous relaxation;
* signed property influence and fanout-cancellation scores;
* nested budget frontiers and a strict, first-regression stop-loss gate.

It is suitable for controlled toys and design audits.  Production integration
must additionally use the outward binary64 assembly rules in ``operator_hz``
and pass the Phase-0 and fixed-sentinel gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
from typing import Dict, Mapping, Optional, Sequence, Tuple


def _q(value: int | float | Fraction | str) -> Fraction:
    """Convert a scalar to the exact value represented by the caller."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction.from_float(value)
    return Fraction(value)


@dataclass(frozen=True)
class LinearForm:
    """An exact affine form over named continuous variables."""

    constant: Fraction = Fraction(0)
    terms: Tuple[Tuple[str, Fraction], ...] = ()

    @staticmethod
    def make(
        constant: int | float | Fraction | str = 0,
        terms: Optional[Mapping[str, int | float | Fraction | str]] = None,
    ) -> "LinearForm":
        combined: Dict[str, Fraction] = {}
        for name, coefficient in (terms or {}).items():
            value = _q(coefficient)
            if value:
                combined[str(name)] = combined.get(str(name), Fraction(0)) + value
        return LinearForm(
            _q(constant),
            tuple(sorted((name, value) for name, value in combined.items() if value)),
        )

    @staticmethod
    def variable(name: str) -> "LinearForm":
        return LinearForm.make(terms={str(name): Fraction(1)})

    def as_dict(self) -> Dict[str, Fraction]:
        return dict(self.terms)

    def coefficient(self, name: str) -> Fraction:
        return self.as_dict().get(str(name), Fraction(0))

    def scaled(self, scale: int | float | Fraction | str) -> "LinearForm":
        value = _q(scale)
        return LinearForm.make(
            self.constant * value,
            {name: coefficient * value for name, coefficient in self.terms},
        )

    def plus(self, other: "LinearForm") -> "LinearForm":
        terms = self.as_dict()
        for name, coefficient in other.terms:
            terms[name] = terms.get(name, Fraction(0)) + coefficient
        return LinearForm.make(self.constant + other.constant, terms)

    def minus(self, other: "LinearForm") -> "LinearForm":
        return self.plus(other.scaled(-1))

    def evaluate(self, values: Mapping[str, Fraction]) -> Fraction:
        result = self.constant
        for name, coefficient in self.terms:
            result += coefficient * values[name]
        return result


@dataclass(frozen=True)
class InputAffine:
    """An exact scalar affine expression ``slope*x + intercept``."""

    slope: Fraction
    intercept: Fraction

    def scaled(self, scale: Fraction) -> "InputAffine":
        return InputAffine(self.slope * scale, self.intercept * scale)

    def plus(self, other: "InputAffine") -> "InputAffine":
        return InputAffine(
            self.slope + other.slope,
            self.intercept + other.intercept,
        )

    def evaluate(self, x: Fraction) -> Fraction:
        return self.slope * x + self.intercept

    def bounds(self, lower: Fraction, upper: Fraction) -> Tuple[Fraction, Fraction]:
        left = self.evaluate(lower)
        right = self.evaluate(upper)
        return (min(left, right), max(left, right))


@dataclass(frozen=True)
class ScalarNode:
    """One topologically ordered scalar DAG node."""

    name: str
    kind: str
    terms: Tuple[Tuple[str, Fraction], ...] = ()
    bias: Fraction = Fraction(0)

    @staticmethod
    def input(name: str = "x") -> "ScalarNode":
        return ScalarNode(str(name), "input")

    @staticmethod
    def affine(
        name: str,
        terms: Mapping[str, int | float | Fraction | str],
        bias: int | float | Fraction | str = 0,
    ) -> "ScalarNode":
        canonical = LinearForm.make(0, terms).terms
        return ScalarNode(str(name), "affine", canonical, _q(bias))

    @staticmethod
    def relu(name: str, source: str) -> "ScalarNode":
        return ScalarNode(str(name), "relu", ((str(source), Fraction(1)),))

    @property
    def source(self) -> str:
        if self.kind != "relu" or len(self.terms) != 1:
            raise ValueError(f"{self.name} is not a canonical ReLU node")
        return self.terms[0][0]


@dataclass(frozen=True)
class FractionScalarDAG:
    """A controlled one-input scalar DAG used by the proof harness."""

    nodes: Tuple[ScalarNode, ...]
    input_lower: Fraction
    input_upper: Fraction
    output: str

    @staticmethod
    def make(
        nodes: Sequence[ScalarNode],
        *,
        input_lower: int | float | Fraction | str,
        input_upper: int | float | Fraction | str,
        output: str,
    ) -> "FractionScalarDAG":
        dag = FractionScalarDAG(
            tuple(nodes),
            _q(input_lower),
            _q(input_upper),
            str(output),
        )
        dag.validate()
        return dag

    def validate(self) -> None:
        if self.input_lower > self.input_upper:
            raise ValueError("input_lower must not exceed input_upper")
        seen: set[str] = set()
        inputs = 0
        for node in self.nodes:
            if not node.name or node.name in seen:
                raise ValueError(f"duplicate/empty node name {node.name!r}")
            if node.kind not in {"input", "affine", "relu"}:
                raise ValueError(f"unsupported scalar node kind {node.kind!r}")
            if node.kind == "input":
                inputs += 1
                if node.terms or node.bias:
                    raise ValueError("input node cannot have terms or a bias")
            else:
                missing = [name for name, _ in node.terms if name not in seen]
                if missing:
                    raise ValueError(
                        f"node {node.name} is not topological; missing {missing}"
                    )
                if node.kind == "relu" and (
                    len(node.terms) != 1 or node.terms[0][1] != 1
                ):
                    raise ValueError(f"ReLU node {node.name} must have one source")
            seen.add(node.name)
        if inputs != 1:
            raise ValueError(f"exactly one input node is required, got {inputs}")
        if self.output not in seen:
            raise ValueError(f"missing output node {self.output!r}")

    @property
    def input_name(self) -> str:
        return next(node.name for node in self.nodes if node.kind == "input")

    @property
    def relu_names(self) -> Tuple[str, ...]:
        return tuple(node.name for node in self.nodes if node.kind == "relu")

    def by_name(self) -> Dict[str, ScalarNode]:
        return {node.name: node for node in self.nodes}


@dataclass(frozen=True)
class PhaseRegion:
    phases: Tuple[Tuple[str, bool], ...]
    lower: Fraction
    upper: Fraction
    expressions: Mapping[str, InputAffine]


@dataclass(frozen=True)
class FractionPhaseOracle:
    true_lower: Fraction
    true_upper: Fraction
    lower_witness: Fraction
    upper_witness: Fraction
    feasible_phase_regions: int
    enumerated_phase_patterns: int
    node_bounds: Mapping[str, Tuple[Fraction, Fraction]]
    regions: Tuple[PhaseRegion, ...]


def _restrict_phase(
    lower: Fraction,
    upper: Fraction,
    expr: InputAffine,
    *,
    active: bool,
) -> Optional[Tuple[Fraction, Fraction]]:
    """Intersect a 1-D interval with ``expr>=0`` or ``expr<=0`` exactly."""

    slope, intercept = expr.slope, expr.intercept
    if not active:
        slope, intercept = -slope, -intercept
    # The normalized condition is slope*x + intercept >= 0.
    if slope == 0:
        return (lower, upper) if intercept >= 0 else None
    threshold = -intercept / slope
    if slope > 0:
        lower = max(lower, threshold)
    else:
        upper = min(upper, threshold)
    if lower > upper:
        return None
    return lower, upper


def enumerate_fraction_phases(dag: FractionScalarDAG) -> Tuple[PhaseRegion, ...]:
    """Enumerate all ReLU phase patterns and retain exact feasible regions."""

    dag.validate()
    relu_names = dag.relu_names
    regions = []
    for bits in itertools.product((False, True), repeat=len(relu_names)):
        phase_map = dict(zip(relu_names, bits))
        lower, upper = dag.input_lower, dag.input_upper
        expressions: Dict[str, InputAffine] = {}
        feasible = True
        for node in dag.nodes:
            if node.kind == "input":
                expressions[node.name] = InputAffine(Fraction(1), Fraction(0))
            elif node.kind == "affine":
                expr = InputAffine(Fraction(0), node.bias)
                for predecessor, coefficient in node.terms:
                    expr = expr.plus(expressions[predecessor].scaled(coefficient))
                expressions[node.name] = expr
            else:
                pre = expressions[node.source]
                restricted = _restrict_phase(
                    lower,
                    upper,
                    pre,
                    active=phase_map[node.name],
                )
                if restricted is None:
                    feasible = False
                    break
                lower, upper = restricted
                expressions[node.name] = (
                    pre
                    if phase_map[node.name]
                    else InputAffine(Fraction(0), Fraction(0))
                )
        if feasible:
            regions.append(
                PhaseRegion(
                    tuple((name, phase_map[name]) for name in relu_names),
                    lower,
                    upper,
                    dict(expressions),
                )
            )
    return tuple(regions)


def fraction_phase_oracle(dag: FractionScalarDAG) -> FractionPhaseOracle:
    """Return exact true ranges by exhaustive phase and endpoint evaluation."""

    regions = enumerate_fraction_phases(dag)
    if not regions:
        raise ValueError("the scalar DAG has no feasible ReLU phase")
    node_bounds: Dict[str, Tuple[Fraction, Fraction]] = {}
    true_lower: Optional[Fraction] = None
    true_upper: Optional[Fraction] = None
    lower_witness = upper_witness = dag.input_lower
    for region in regions:
        for name, expression in region.expressions.items():
            local_lower, local_upper = expression.bounds(region.lower, region.upper)
            if name in node_bounds:
                old_lower, old_upper = node_bounds[name]
                node_bounds[name] = (
                    min(old_lower, local_lower),
                    max(old_upper, local_upper),
                )
            else:
                node_bounds[name] = (local_lower, local_upper)
        output = region.expressions[dag.output]
        for x in (region.lower, region.upper):
            value = output.evaluate(x)
            if true_lower is None or value < true_lower:
                true_lower, lower_witness = value, x
            if true_upper is None or value > true_upper:
                true_upper, upper_witness = value, x
    assert true_lower is not None and true_upper is not None
    return FractionPhaseOracle(
        true_lower,
        true_upper,
        lower_witness,
        upper_witness,
        len(regions),
        2 ** len(dag.relu_names),
        node_bounds,
        regions,
    )


@dataclass(frozen=True)
class ResidualEnvelope:
    lower: Fraction
    upper: Fraction
    slope: Fraction
    residual_lower: Fraction
    residual_upper: Fraction

    @property
    def center(self) -> Fraction:
        return (self.residual_lower + self.residual_upper) / 2

    @property
    def radius(self) -> Fraction:
        return (self.residual_upper - self.residual_lower) / 2


def relu_residual_envelope(
    lower: int | float | Fraction | str,
    upper: int | float | Fraction | str,
    *,
    slope: Optional[int | float | Fraction | str] = None,
) -> ResidualEnvelope:
    """Construct the exact residual range for an unstable scalar ReLU."""

    l, u = _q(lower), _q(upper)
    if not l < 0 < u:
        raise ValueError(f"residualization requires l<0<u, got [{l}, {u}]")
    a = u / (u - l) if slope is None else _q(slope)
    if not Fraction(0) <= a <= Fraction(1):
        raise ValueError(f"residual slope must be in [0,1], got {a}")
    requirements = (Fraction(0), -a * l, (Fraction(1) - a) * u)
    return ResidualEnvelope(l, u, a, min(requirements), max(requirements))


@dataclass(frozen=True)
class ResidualCandidate:
    node: str
    source_lower: Fraction
    source_upper: Fraction
    slope: Fraction
    residual_upper: Fraction
    signed_property_sensitivity: Fraction
    absolute_fanout_sensitivity: Fraction
    fanout_count: int
    guard_mode: str
    saved_rows: int
    shared_radius_impact: Fraction
    duplicated_radius_impact: Fraction

    @property
    def cancellation_credit(self) -> Fraction:
        return self.duplicated_radius_impact - self.shared_radius_impact


_GUARD_ROWS = {"none": 0, "zero": 1, "identity": 1, "both": 2}


def property_residual_candidates(
    dag: FractionScalarDAG,
    oracle: Optional[FractionPhaseOracle] = None,
) -> Tuple[ResidualCandidate, ...]:
    """Rank unstable ReLUs using signed property influence.

    The influence is a secant-linearized reverse pass.  It is a ranking proxy,
    not a proof.  Soundness comes solely from the residual envelope and any
    retained predicates.  For an upper-bound objective, nonnegative influence
    receives ``guard_mode='none'`` because the residual factor already embeds
    the secant upper predicate.  Negative influence keeps both lower facets.
    """

    oracle = oracle or fraction_phase_oracle(dag)
    adjoint: Dict[str, Fraction] = {
        node.name: Fraction(0) for node in dag.nodes
    }
    adjoint[dag.output] = Fraction(1)
    edge_contributions: Dict[str, list[Fraction]] = {
        node.name: [] for node in dag.nodes
    }

    for node in reversed(dag.nodes):
        downstream = adjoint[node.name]
        if node.kind == "affine":
            for predecessor, coefficient in node.terms:
                contribution = downstream * coefficient
                adjoint[predecessor] += contribution
                edge_contributions[predecessor].append(contribution)
        elif node.kind == "relu":
            lower, upper = oracle.node_bounds[node.source]
            if upper <= 0:
                local_slope = Fraction(0)
            elif lower >= 0:
                local_slope = Fraction(1)
            else:
                local_slope = relu_residual_envelope(lower, upper).slope
            contribution = downstream * local_slope
            adjoint[node.source] += contribution
            edge_contributions[node.source].append(contribution)

    candidates = []
    for node in dag.nodes:
        if node.kind != "relu":
            continue
        lower, upper = oracle.node_bounds[node.source]
        if not lower < 0 < upper:
            continue
        envelope = relu_residual_envelope(lower, upper)
        sensitivity = adjoint[node.name]
        absolute_fanout = sum(
            (abs(value) for value in edge_contributions[node.name]),
            Fraction(0),
        )
        guard_mode = "none" if sensitivity >= 0 else "both"
        radius = envelope.radius
        candidates.append(
            ResidualCandidate(
                node=node.name,
                source_lower=lower,
                source_upper=upper,
                slope=envelope.slope,
                residual_upper=envelope.residual_upper,
                signed_property_sensitivity=sensitivity,
                absolute_fanout_sensitivity=absolute_fanout,
                fanout_count=len(edge_contributions[node.name]),
                guard_mode=guard_mode,
                saved_rows=2 - _GUARD_ROWS[guard_mode],
                shared_radius_impact=abs(sensitivity) * radius,
                duplicated_radius_impact=absolute_fanout * radius,
            )
        )

    def rank_key(candidate: ResidualCandidate):
        if candidate.saved_rows <= 0:
            return (1, Fraction(0), Fraction(0), candidate.node)
        return (
            0,
            candidate.shared_radius_impact / candidate.saved_rows,
            -candidate.cancellation_credit,
            candidate.node,
        )

    return tuple(sorted(candidates, key=rank_key))


@dataclass(frozen=True)
class ResidualPlan:
    selected: Tuple[str, ...]
    guards: Tuple[Tuple[str, str], ...]
    slopes: Tuple[Tuple[str, Fraction], ...]

    def guard_for(self, node: str) -> str:
        return dict(self.guards)[node]

    def slope_for(self, node: str) -> Fraction:
        return dict(self.slopes)[node]


def nested_residual_plan(
    candidates: Sequence[ResidualCandidate],
    budget: int,
) -> ResidualPlan:
    """Return a deterministic nested prefix containing only row-saving nodes."""

    if int(budget) < 0:
        raise ValueError("residualization budget must be nonnegative")
    chosen = tuple(
        candidate
        for candidate in candidates
        if candidate.saved_rows > 0
    )[: int(budget)]
    return ResidualPlan(
        tuple(candidate.node for candidate in chosen),
        tuple((candidate.node, candidate.guard_mode) for candidate in chosen),
        tuple((candidate.node, candidate.slope) for candidate in chosen),
    )


@dataclass(frozen=True)
class LinearConstraint:
    lhs: LinearForm
    rhs: Fraction
    tag: str


@dataclass(frozen=True)
class RelaxedRelu:
    node: str
    source: str
    selected: bool
    variable: str
    slope: Fraction
    residual_upper: Fraction
    guard_mode: str


@dataclass(frozen=True)
class FractionRelaxation:
    dag: FractionScalarDAG
    plan: ResidualPlan
    variable_bounds: Mapping[str, Tuple[Fraction, Fraction]]
    constraints: Tuple[LinearConstraint, ...]
    node_forms: Mapping[str, LinearForm]
    relus: Tuple[RelaxedRelu, ...]
    objective: LinearForm
    coupling_rows: int
    coupling_nnz: int
    residual_factor_count: int
    objective_residual_nnz: int


def encode_fraction_relaxation(
    dag: FractionScalarDAG,
    plan: ResidualPlan,
    *,
    oracle: Optional[FractionPhaseOracle] = None,
) -> FractionRelaxation:
    """Encode the controlled DAG as triangle or shared-residual constraints."""

    oracle = oracle or fraction_phase_oracle(dag)
    selected = set(plan.selected)
    if len(selected) != len(plan.selected):
        raise ValueError("residual plan repeats a selected node")
    unknown = selected - set(dag.relu_names)
    if unknown:
        raise ValueError(f"residual plan references unknown ReLUs {sorted(unknown)}")

    bounds: Dict[str, Tuple[Fraction, Fraction]] = {
        dag.input_name: (dag.input_lower, dag.input_upper)
    }
    forms: Dict[str, LinearForm] = {}
    constraints: list[LinearConstraint] = []
    relus: list[RelaxedRelu] = []

    def append(lhs: LinearForm, rhs: Fraction, tag: str) -> None:
        if not lhs.terms:
            if lhs.constant > rhs:
                raise ValueError(f"constant contradiction in {tag}")
            return
        constraints.append(LinearConstraint(lhs, rhs, tag))

    for node in dag.nodes:
        if node.kind == "input":
            forms[node.name] = LinearForm.variable(node.name)
        elif node.kind == "affine":
            value = LinearForm.make(node.bias)
            for predecessor, coefficient in node.terms:
                value = value.plus(forms[predecessor].scaled(coefficient))
            forms[node.name] = value
        else:
            pre = forms[node.source]
            lower, upper = oracle.node_bounds[node.source]
            if upper <= 0:
                forms[node.name] = LinearForm.make()
                continue
            if lower >= 0:
                forms[node.name] = pre
                continue

            if node.name in selected:
                slope = plan.slope_for(node.name)
                envelope = relu_residual_envelope(lower, upper, slope=slope)
                variable = f"rho:{node.name}"
                bounds[variable] = (
                    envelope.residual_lower,
                    envelope.residual_upper,
                )
                value = pre.scaled(slope).plus(LinearForm.variable(variable))
                guard = plan.guard_for(node.name)
                if guard not in _GUARD_ROWS:
                    raise ValueError(f"unknown residual guard mode {guard!r}")
                if guard in {"zero", "both"}:
                    append(value.scaled(-1), Fraction(0), f"{node.name}:y>=0")
                if guard in {"identity", "both"}:
                    append(pre.minus(value), Fraction(0), f"{node.name}:y>=x")
                forms[node.name] = value
                relus.append(
                    RelaxedRelu(
                        node.name,
                        node.source,
                        True,
                        variable,
                        slope,
                        envelope.residual_upper,
                        guard,
                    )
                )
            else:
                envelope = relu_residual_envelope(lower, upper)
                variable = f"y:{node.name}"
                bounds[variable] = (Fraction(0), upper)
                value = LinearForm.variable(variable)
                # The box supplies y>=0.  These are the other two triangle
                # facets used by the current local operator formulation.
                append(pre.minus(value), Fraction(0), f"{node.name}:y>=x")
                append(
                    value.minus(pre.scaled(envelope.slope)),
                    envelope.residual_upper,
                    f"{node.name}:secant",
                )
                forms[node.name] = value
                relus.append(
                    RelaxedRelu(
                        node.name,
                        node.source,
                        False,
                        variable,
                        envelope.slope,
                        envelope.residual_upper,
                        "triangle",
                    )
                )

    objective = forms[dag.output]
    residual_variables = {
        relu.variable for relu in relus if relu.selected
    }
    return FractionRelaxation(
        dag,
        plan,
        dict(bounds),
        tuple(constraints),
        dict(forms),
        tuple(relus),
        objective,
        len(constraints),
        sum(len(constraint.lhs.terms) for constraint in constraints),
        len(residual_variables),
        sum(
            1
            for variable in residual_variables
            if objective.coefficient(variable) != 0
        ),
    )


def _solve_fraction_square(
    rows: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
) -> Optional[Tuple[Fraction, ...]]:
    """Solve a square rational system, returning ``None`` when singular."""

    n = len(rows)
    matrix = [list(row) + [rhs_value] for row, rhs_value in zip(rows, rhs)]
    if any(len(row) != n + 1 for row in matrix):
        raise ValueError("fraction solver requires a square matrix")
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if matrix[row][column] != 0),
            None,
        )
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        pivot_value = matrix[column][column]
        matrix[column] = [value / pivot_value for value in matrix[column]]
        for row in range(n):
            if row == column:
                continue
            scale = matrix[row][column]
            if scale:
                matrix[row] = [
                    left - scale * right
                    for left, right in zip(matrix[row], matrix[column])
                ]
    return tuple(matrix[row][-1] for row in range(n))


@dataclass(frozen=True)
class FractionLPResult:
    lower: Fraction
    upper: Fraction
    lower_witness: Mapping[str, Fraction]
    upper_witness: Mapping[str, Fraction]
    vertices_checked: int


def exact_fraction_lp_range(relaxation: FractionRelaxation) -> FractionLPResult:
    """Solve the tiny bounded rational LP by exhaustive vertex enumeration."""

    variables = tuple(relaxation.variable_bounds)
    if not variables:
        value = relaxation.objective.constant
        return FractionLPResult(value, value, {}, {}, 1)

    rows: list[Tuple[Fraction, ...]] = []
    rhs: list[Fraction] = []
    for constraint in relaxation.constraints:
        coefficients = constraint.lhs.as_dict()
        rows.append(tuple(coefficients.get(name, Fraction(0)) for name in variables))
        rhs.append(constraint.rhs - constraint.lhs.constant)
    for index, name in enumerate(variables):
        lower, upper = relaxation.variable_bounds[name]
        positive = [Fraction(0)] * len(variables)
        positive[index] = Fraction(1)
        rows.append(tuple(positive))
        rhs.append(upper)
        negative = [Fraction(0)] * len(variables)
        negative[index] = Fraction(-1)
        rows.append(tuple(negative))
        rhs.append(-lower)

    objective_coefficients = relaxation.objective.as_dict()
    minimum: Optional[Fraction] = None
    maximum: Optional[Fraction] = None
    minimum_point: Tuple[Fraction, ...] = ()
    maximum_point: Tuple[Fraction, ...] = ()
    vertices_checked = 0
    for active in itertools.combinations(range(len(rows)), len(variables)):
        point = _solve_fraction_square(
            [rows[index] for index in active],
            [rhs[index] for index in active],
        )
        if point is None:
            continue
        if any(
            sum(coefficient * value for coefficient, value in zip(row, point))
            > bound
            for row, bound in zip(rows, rhs)
        ):
            continue
        vertices_checked += 1
        value = relaxation.objective.constant + sum(
            objective_coefficients.get(name, Fraction(0)) * coordinate
            for name, coordinate in zip(variables, point)
        )
        if minimum is None or value < minimum:
            minimum, minimum_point = value, point
        if maximum is None or value > maximum:
            maximum, maximum_point = value, point
    if minimum is None or maximum is None:
        raise ValueError("bounded relaxation has no enumerated feasible vertex")
    return FractionLPResult(
        minimum,
        maximum,
        dict(zip(variables, minimum_point)),
        dict(zip(variables, maximum_point)),
        vertices_checked,
    )


@dataclass(frozen=True)
class ContainmentAudit:
    phase_regions: int
    endpoint_assignments: int
    variable_bound_checks: int
    coupling_checks: int
    output_identity_checks: int


def audit_fraction_phase_containment(
    relaxation: FractionRelaxation,
    *,
    oracle: Optional[FractionPhaseOracle] = None,
) -> ContainmentAudit:
    """Prove containment on every exact phase region.

    Within one phase, every true node, residual assignment, constraint slack,
    and output identity is affine in the scalar input.  Checking both exact
    endpoints therefore covers the entire phase interval.
    """

    oracle = oracle or fraction_phase_oracle(relaxation.dag)
    relu_by_node = {relu.node: relu for relu in relaxation.relus}
    bound_checks = coupling_checks = output_checks = assignments = 0
    for region in oracle.regions:
        for x in dict.fromkeys((region.lower, region.upper)):
            assignments += 1
            values: Dict[str, Fraction] = {
                relaxation.dag.input_name: x
            }
            for node in relaxation.dag.nodes:
                if node.name not in relu_by_node:
                    continue
                relu = relu_by_node[node.name]
                true_y = region.expressions[node.name].evaluate(x)
                if relu.selected:
                    true_pre = region.expressions[relu.source].evaluate(x)
                    values[relu.variable] = true_y - relu.slope * true_pre
                else:
                    values[relu.variable] = true_y

            for variable, (lower, upper) in relaxation.variable_bounds.items():
                bound_checks += 1
                value = values[variable]
                if value < lower or value > upper:
                    raise AssertionError(
                        f"phase containment failed bound {variable}: "
                        f"{value} not in [{lower}, {upper}]"
                    )
            for constraint in relaxation.constraints:
                coupling_checks += 1
                value = constraint.lhs.evaluate(values)
                if value > constraint.rhs:
                    raise AssertionError(
                        f"phase containment failed {constraint.tag}: "
                        f"{value} > {constraint.rhs}"
                    )
            output_checks += 1
            encoded = relaxation.objective.evaluate(values)
            exact = region.expressions[relaxation.dag.output].evaluate(x)
            if encoded != exact:
                raise AssertionError(
                    f"phase containment output mismatch: {encoded} != {exact}"
                )
    return ContainmentAudit(
        len(oracle.regions),
        assignments,
        bound_checks,
        coupling_checks,
        output_checks,
    )


@dataclass(frozen=True)
class ResidualBudgetMetric:
    requested_budget: int
    selected: Tuple[str, ...]
    coupling_rows: int
    coupling_nnz: int
    residual_factor_count: int
    objective_residual_nnz: int
    relaxation_lower: Fraction
    relaxation_upper: Fraction
    true_lower: Fraction
    true_upper: Fraction
    lower_gap: Fraction
    upper_gap: Fraction
    containment_assignments: int


def residual_budget_frontier(
    dag: FractionScalarDAG,
    *,
    max_budget: int,
) -> Tuple[ResidualBudgetMetric, ...]:
    """Audit every nested prefix with exact soundness and tightness oracles."""

    if int(max_budget) < 0:
        raise ValueError("max_budget must be nonnegative")
    oracle = fraction_phase_oracle(dag)
    candidates = property_residual_candidates(dag, oracle)
    metrics = []
    previous_selected: Tuple[str, ...] = ()
    for budget in range(int(max_budget) + 1):
        plan = nested_residual_plan(candidates, budget)
        if plan.selected[: len(previous_selected)] != previous_selected:
            raise AssertionError("residual budgets are not nested")
        previous_selected = plan.selected
        relaxation = encode_fraction_relaxation(dag, plan, oracle=oracle)
        containment = audit_fraction_phase_containment(
            relaxation, oracle=oracle
        )
        lp = exact_fraction_lp_range(relaxation)
        if lp.lower > oracle.true_lower or lp.upper < oracle.true_upper:
            raise AssertionError("relaxation under-approximates the exact oracle")
        metrics.append(
            ResidualBudgetMetric(
                budget,
                plan.selected,
                relaxation.coupling_rows,
                relaxation.coupling_nnz,
                relaxation.residual_factor_count,
                relaxation.objective_residual_nnz,
                lp.lower,
                lp.upper,
                oracle.true_lower,
                oracle.true_upper,
                oracle.true_lower - lp.lower,
                lp.upper - oracle.true_upper,
                containment.endpoint_assignments,
            )
        )
    return tuple(metrics)


@dataclass(frozen=True)
class StopLossDecision:
    accepted: Tuple[ResidualBudgetMetric, ...]
    rejected: Optional[ResidualBudgetMetric]
    reason: str


def strict_stop_loss_prefix(
    frontier: Sequence[ResidualBudgetMetric],
    *,
    max_upper_regression: int | float | Fraction | str = 0,
) -> StopLossDecision:
    """Promote prefixes only while size shrinks and upper tightness is retained."""

    if not frontier:
        raise ValueError("stop-loss requires a nonempty frontier")
    allowance = _q(max_upper_regression)
    if allowance < 0:
        raise ValueError("max_upper_regression must be nonnegative")
    accepted = [frontier[0]]
    for metric in frontier[1:]:
        previous = accepted[-1]
        if metric.selected == previous.selected:
            return StopLossDecision(
                tuple(accepted),
                metric,
                "budget_plateau_no_row_saving_candidate",
            )
        if metric.coupling_rows >= previous.coupling_rows:
            return StopLossDecision(
                tuple(accepted),
                metric,
                "coupling_rows_did_not_decrease",
            )
        if metric.coupling_nnz > previous.coupling_nnz:
            return StopLossDecision(
                tuple(accepted),
                metric,
                "coupling_nnz_increased",
            )
        if metric.relaxation_upper > previous.relaxation_upper + allowance:
            return StopLossDecision(
                tuple(accepted),
                metric,
                "property_upper_regressed",
            )
        accepted.append(metric)
    return StopLossDecision(tuple(accepted), None, "all_prefixes_promoted")
