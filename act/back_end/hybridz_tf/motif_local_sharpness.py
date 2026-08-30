#!/usr/bin/env python3
# ===- motif_local_sharpness.py - bounded local HybridZ cuts ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------====#
"""Exact toy gate for one complementary-ReLU residual motif.

This module deliberately does **not** alter the production operator builder.
It audits the following bounded scalar motif before any graph-level
integration is attempted::

    -1 <= x <= 1
    h1 = ReLU(x)
    h2 = ReLU(-x)
    h3 = ReLU(h1 - h2)
    r  = h1 - h3

The exact graph has ``r == 0`` because
``ReLU(x) - ReLU(-x) == x`` and therefore ``h3 == h1``.  Independent
per-neuron triangle relaxations lose both identities and admit
``-1/2 <= r <= 1/2``.

The minimal two-sided local strengthening is the one-row equality
``h1 - h3 == 0`` (two nonzeros, no new variable or binary).  For a one-sided
proof of ``r <= 0``, the single inequality ``h1 - h3 <= 0`` is sufficient.
Both are semantic cuts: a future matcher must establish the exact signed
motif above; approximate coefficient matching is not a sound precondition.

All decisive values in the gate are computed with :class:`fractions.Fraction`.
LP optima are obtained by exact vertex enumeration, not accepted from a
floating-point solver status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import itertools
from typing import Dict, Optional, Sequence, Tuple


_VARIABLES = ("x", "h1", "h2", "h3")
_ZERO = Fraction(0)
_ONE = Fraction(1)
_HALF = Fraction(1, 2)


class MotifLocalSharpnessError(AssertionError):
    """The bounded motif failed a soundness, tightness, or cost gate."""


@dataclass(frozen=True)
class RationalRow:
    """One exact linear row over ``(x, h1, h2, h3)``."""

    coefficients: Tuple[Fraction, ...]
    rhs: Fraction
    relation: str
    tag: str

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(_VARIABLES):
            raise ValueError(
                f"{self.tag}: expected {len(_VARIABLES)} coefficients"
            )
        if self.relation not in {"<=", "=="}:
            raise ValueError(f"{self.tag}: unsupported relation {self.relation!r}")

    @property
    def nnz(self) -> int:
        return sum(value != 0 for value in self.coefficients)

    def evaluate(self, point: Sequence[Fraction]) -> Fraction:
        if len(point) != len(self.coefficients):
            raise ValueError(f"{self.tag}: point dimension mismatch")
        return sum(
            (
                coefficient * value
                for coefficient, value in zip(self.coefficients, point)
            ),
            _ZERO,
        )


@dataclass(frozen=True)
class ExactLPRange:
    """Exact objective range and deterministic extremal vertices."""

    minimum: Fraction
    maximum: Fraction
    minimum_vertex: Tuple[Fraction, ...]
    maximum_vertex: Tuple[Fraction, ...]
    vertices_checked: int


@dataclass(frozen=True)
class MotifLocalSharpnessReceipt:
    """Machine-readable receipt emitted by the toy stop-loss gate."""

    schema: str
    passed: bool
    variables: Tuple[str, ...]
    phase_patterns_total: int
    phase_patterns_feasible: int
    phase_patterns_nondegenerate: int
    fraction_samples: int
    baseline_lower: Fraction
    baseline_upper: Fraction
    complement_only_lower: Fraction
    complement_only_upper: Fraction
    upper_cut_lower: Fraction
    upper_cut_upper: Fraction
    repeat_equality_lower: Fraction
    repeat_equality_upper: Fraction
    accepted_cut: str
    accepted_rows: int
    accepted_nnz: int
    accepted_new_continuous: int
    accepted_new_binary: int
    inequality_only_rows: int
    inequality_only_nnz: int
    rejected_candidate: str
    rejected_reason: str
    exact_lp_vertices_checked: int

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly receipt without losing rational exactness."""

        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, Fraction):
                result[key] = (
                    str(value.numerator)
                    if value.denominator == 1
                    else f"{value.numerator}/{value.denominator}"
                )
        return result


def complementary_relu_repeat_cut(*, one_sided: bool = False) -> RationalRow:
    """Return the proved motif cut.

    Preconditions for applying the returned row are exact symbolic graph
    identities, not approximate numerical similarities::

        h1 = ReLU(x)
        h2 = ReLU(-x)
        h3 = ReLU(h1 - h2)

    ``one_sided=True`` emits the property-directed upper cut ``h1-h3 <= 0``.
    The default emits the two-sided equality ``h1-h3 == 0``.
    """

    return RationalRow(
        coefficients=(_ZERO, _ONE, _ZERO, -_ONE),
        rhs=_ZERO,
        relation="<=" if one_sided else "==",
        tag=(
            "complementary_relu_repeat_upper_v1"
            if one_sided
            else "complementary_relu_repeat_identity_v1"
        ),
    )


def _complement_identity_cut() -> RationalRow:
    """Return the valid but insufficient intermediate identity h1-h2=x."""

    return RationalRow(
        coefficients=(-_ONE, _ONE, -_ONE, _ZERO),
        rhs=_ZERO,
        relation="==",
        tag="complementary_relu_difference_identity_v1",
    )


def _triangle_rows() -> Tuple[RationalRow, ...]:
    """Independent triangle relaxation for all three ReLUs on [-1, 1]."""

    def le(coefficients: Sequence[Fraction], rhs: Fraction, tag: str) -> RationalRow:
        return RationalRow(tuple(coefficients), rhs, "<=", tag)

    return (
        le((_ONE, _ZERO, _ZERO, _ZERO), _ONE, "x_upper"),
        le((-_ONE, _ZERO, _ZERO, _ZERO), _ONE, "x_lower"),
        # h1 = ReLU(x), l=-1, u=1.
        le((_ZERO, -_ONE, _ZERO, _ZERO), _ZERO, "h1_nonnegative"),
        le((_ONE, -_ONE, _ZERO, _ZERO), _ZERO, "h1_above_x"),
        le((-_HALF, _ONE, _ZERO, _ZERO), _HALF, "h1_triangle"),
        # h2 = ReLU(-x), l=-1, u=1.
        le((_ZERO, _ZERO, -_ONE, _ZERO), _ZERO, "h2_nonnegative"),
        le((-_ONE, _ZERO, -_ONE, _ZERO), _ZERO, "h2_above_neg_x"),
        le((_HALF, _ZERO, _ONE, _ZERO), _HALF, "h2_triangle"),
        # h3 = ReLU(h1-h2), with the valid interval [-1, 1].
        le((_ZERO, _ZERO, _ZERO, -_ONE), _ZERO, "h3_nonnegative"),
        le(
            (_ZERO, _ONE, -_ONE, -_ONE),
            _ZERO,
            "h3_above_h1_minus_h2",
        ),
        le(
            (_ZERO, -_HALF, _HALF, _ONE),
            _HALF,
            "h3_triangle",
        ),
    )


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), _ZERO)


def _solve_square(
    rows: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
) -> Optional[Tuple[Fraction, ...]]:
    """Solve a square rational system; return ``None`` when singular."""

    size = len(rows)
    if size == 0 or len(rhs) != size or any(len(row) != size for row in rows):
        raise ValueError("expected a non-empty square system")
    augmented = [
        [Fraction(value) for value in row] + [Fraction(bound)]
        for row, bound in zip(rows, rhs)
    ]

    for column in range(size):
        pivot = next(
            (row for row in range(column, size)
             if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            return None
        if pivot != column:
            augmented[column], augmented[pivot] = (
                augmented[pivot],
                augmented[column],
            )
        pivot_value = augmented[column][column]
        augmented[column] = [
            value / pivot_value for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    augmented[row], augmented[column]
                )
            ]
    return tuple(augmented[row][-1] for row in range(size))


def _exact_lp_range(
    inequalities: Sequence[RationalRow],
    equalities: Sequence[RationalRow],
    objective: Sequence[Fraction],
) -> ExactLPRange:
    """Optimize a bounded rational polytope by exact vertex enumeration."""

    dimension = len(objective)
    if dimension != len(_VARIABLES):
        raise ValueError("objective dimension mismatch")
    if any(row.relation != "<=" for row in inequalities):
        raise ValueError("inequality list contains a non-inequality")
    if any(row.relation != "==" for row in equalities):
        raise ValueError("equality list contains a non-equality")
    if len(equalities) > dimension:
        raise ValueError("too many equality rows")

    active_count = dimension - len(equalities)
    vertices: Dict[Tuple[Fraction, ...], Fraction] = {}
    for active in itertools.combinations(inequalities, active_count):
        active_rows = tuple(equalities) + tuple(active)
        point = _solve_square(
            [row.coefficients for row in active_rows],
            [row.rhs for row in active_rows],
        )
        if point is None:
            continue
        if any(row.evaluate(point) > row.rhs for row in inequalities):
            continue
        if any(row.evaluate(point) != row.rhs for row in equalities):
            continue
        vertices[point] = _dot(objective, point)

    if not vertices:
        raise MotifLocalSharpnessError(
            "exact LP enumeration found no feasible vertex"
        )
    minimum_vertex = min(vertices, key=lambda point: (vertices[point], point))
    maximum_vertex = max(vertices, key=lambda point: (vertices[point], point))
    return ExactLPRange(
        minimum=vertices[minimum_vertex],
        maximum=vertices[maximum_vertex],
        minimum_vertex=minimum_vertex,
        maximum_vertex=maximum_vertex,
        vertices_checked=len(vertices),
    )


def _phase_interval(
    active1: bool,
    active2: bool,
    active3: bool,
) -> Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
    """Return the exact feasible x interval and affine output coefficients."""

    h1_coefficient = _ONE if active1 else _ZERO
    h2_coefficient = -_ONE if active2 else _ZERO
    pre3_coefficient = h1_coefficient - h2_coefficient
    h3_coefficient = pre3_coefficient if active3 else _ZERO

    lower = -_ONE
    upper = _ONE
    phase_conditions = (
        (_ONE, active1),
        (-_ONE, active2),
        (pre3_coefficient, active3),
    )
    for preactivation_coefficient, active in phase_conditions:
        # Active means a*x >= 0; inactive means a*x <= 0.
        signed = preactivation_coefficient if active else -preactivation_coefficient
        if signed > 0:
            lower = max(lower, _ZERO)
        elif signed < 0:
            upper = min(upper, _ZERO)
    return (
        lower,
        upper,
        h1_coefficient,
        h2_coefficient,
        h3_coefficient,
    )


def _audit_exact_phases(cut: RationalRow) -> Tuple[int, int, int]:
    """Enumerate all 2^3 ReLU phase assignments with exact arithmetic."""

    total = 0
    feasible = 0
    nondegenerate = 0
    for active1, active2, active3 in itertools.product((False, True), repeat=3):
        total += 1
        lower, upper, h1_coefficient, h2_coefficient, h3_coefficient = (
            _phase_interval(active1, active2, active3)
        )
        if lower > upper:
            continue
        feasible += 1
        if lower < upper:
            nondegenerate += 1
        for x_value in (lower, upper):
            point = (
                x_value,
                h1_coefficient * x_value,
                h2_coefficient * x_value,
                h3_coefficient * x_value,
            )
            if cut.evaluate(point) != cut.rhs:
                raise MotifLocalSharpnessError(
                    "repeat equality cut excludes exact phase "
                    f"{(active1, active2, active3)} at x={x_value}"
                )
    return total, feasible, nondegenerate


def _fraction_sample_audit(
    cut: RationalRow,
    *,
    max_denominator: int,
) -> int:
    """Check every unique rational grid point with denominator <= the cap."""

    if max_denominator < 1:
        raise ValueError("max_denominator must be positive")
    samples = {
        Fraction(numerator, denominator)
        for denominator in range(1, max_denominator + 1)
        for numerator in range(-denominator, denominator + 1)
    }
    for x_value in sorted(samples):
        h1 = max(_ZERO, x_value)
        h2 = max(_ZERO, -x_value)
        h3 = max(_ZERO, h1 - h2)
        point = (x_value, h1, h2, h3)
        if h1 - h2 != x_value:
            raise MotifLocalSharpnessError(
                f"complement identity failed at x={x_value}"
            )
        if cut.evaluate(point) != cut.rhs:
            raise MotifLocalSharpnessError(
                f"repeat equality cut failed at x={x_value}"
            )
    return len(samples)


def audit_complementary_relu_repeat_motif(
    *,
    max_denominator: int = 64,
    max_equality_rows: int = 1,
    max_equality_nnz: int = 2,
) -> MotifLocalSharpnessReceipt:
    """Run the bounded soundness/tightness/cost/stop-loss gate.

    The first candidate, ``h1-h2=x``, is sound but deliberately rejected:
    on its own it does not improve the ``r`` LP bound.  The repeat equality is
    accepted only if it closes both sides exactly and stays within the explicit
    one-row/two-nonzero budget.
    """

    triangle_rows = _triangle_rows()
    objective = (_ZERO, _ONE, _ZERO, -_ONE)
    base = _exact_lp_range(triangle_rows, (), objective)

    complement_cut = _complement_identity_cut()
    complement_only = _exact_lp_range(
        triangle_rows,
        (complement_cut,),
        objective,
    )

    upper_cut = complementary_relu_repeat_cut(one_sided=True)
    upper_only = _exact_lp_range(
        tuple(triangle_rows) + (upper_cut,),
        (),
        objective,
    )

    repeat_equality = complementary_relu_repeat_cut()
    tightened = _exact_lp_range(
        triangle_rows,
        (repeat_equality,),
        objective,
    )

    total_phases, feasible_phases, nondegenerate_phases = _audit_exact_phases(
        repeat_equality
    )
    sample_count = _fraction_sample_audit(
        repeat_equality,
        max_denominator=max_denominator,
    )

    expected_base = (-_HALF, _HALF)
    if (base.minimum, base.maximum) != expected_base:
        raise MotifLocalSharpnessError(
            "triangle-gap regression: expected [-1/2,1/2], got "
            f"[{base.minimum},{base.maximum}]"
        )
    if (
        complement_only.minimum != base.minimum
        or complement_only.maximum != base.maximum
    ):
        raise MotifLocalSharpnessError(
            "stop-loss control changed: complement-only candidate unexpectedly "
            "changed this toy's LP range"
        )
    if upper_only.maximum != 0 or upper_only.minimum != base.minimum:
        raise MotifLocalSharpnessError(
            "one-sided repeat cut did not close exactly the requested upper side"
        )
    if tightened.minimum != 0 or tightened.maximum != 0:
        raise MotifLocalSharpnessError(
            "repeat equality did not close the two-sided relaxation gap"
        )
    if repeat_equality.nnz > max_equality_nnz or max_equality_rows < 1:
        raise MotifLocalSharpnessError(
            "repeat equality exceeds the configured local cost budget"
        )
    if total_phases != 8 or feasible_phases != 8 or nondegenerate_phases != 2:
        raise MotifLocalSharpnessError(
            "exact phase census changed: expected total/feasible/nondegenerate "
            f"8/8/2, got {total_phases}/{feasible_phases}/"
            f"{nondegenerate_phases}"
        )

    return MotifLocalSharpnessReceipt(
        schema="hybridz_motif_local_sharpness_toy_v1",
        passed=True,
        variables=_VARIABLES,
        phase_patterns_total=total_phases,
        phase_patterns_feasible=feasible_phases,
        phase_patterns_nondegenerate=nondegenerate_phases,
        fraction_samples=sample_count,
        baseline_lower=base.minimum,
        baseline_upper=base.maximum,
        complement_only_lower=complement_only.minimum,
        complement_only_upper=complement_only.maximum,
        upper_cut_lower=upper_only.minimum,
        upper_cut_upper=upper_only.maximum,
        repeat_equality_lower=tightened.minimum,
        repeat_equality_upper=tightened.maximum,
        accepted_cut=repeat_equality.tag,
        accepted_rows=1,
        accepted_nnz=repeat_equality.nnz,
        accepted_new_continuous=0,
        accepted_new_binary=0,
        inequality_only_rows=2,
        inequality_only_nnz=2 * repeat_equality.nnz,
        rejected_candidate=complement_cut.tag,
        rejected_reason="zero_lp_bound_improvement",
        exact_lp_vertices_checked=(
            base.vertices_checked
            + complement_only.vertices_checked
            + upper_only.vertices_checked
            + tightened.vertices_checked
        ),
    )


def motif_local_sharpness_self_test() -> Dict[str, object]:
    """Run the deterministic toy gate and return its audit receipt."""

    return audit_complementary_relu_repeat_motif().as_dict()


if __name__ == "__main__":
    print(motif_local_sharpness_self_test())
