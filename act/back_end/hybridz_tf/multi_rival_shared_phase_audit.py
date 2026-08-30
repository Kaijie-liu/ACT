"""Exact, candidate-only audit for multi-rival shared phase covers.

This module is deliberately disconnected from the verifier.  It separates
two ideas which are easy to conflate:

* A predicate-selector/disjunctive hull over a *fixed* network relaxation
  cannot improve the worst rival upper bound.  Maximizing a linear epigraph
  coordinate over the convex hull of the rival disjuncts gives exactly
  ``max_r max_{z in P} f_r(z)``.
* An exact phase partition can improve the relaxation inside every child.
  If several rivals request the same semantic phase split, sharing the phase
  tree gives the same bound as separate per-rival trees with fewer network
  nodes.  The saved nodes may buy additional exact depth under a fixed wall
  budget, but sharing alone is not a new bound.

The controlled toy uses two occurrences of the same ReLU value,
``a = b = relu(x)``, over ``x in [-1, 1]`` and two rival predicates

``a - b - 1/10`` and ``b - a - 1/10``.

Independent triangle relaxations lose the duplicate-value correlation and
give ``2/5`` for both rivals.  Splitting the *shared semantic phase* of ``x``
fixes both copies in both children and gives the exact ``-1/10``.  Applying
the split to only one fresh/wrong copy leaves a ``2/5`` relaxation, so a
production implementation must bind the phase to a complete stable-ID
equivalence class and fail closed on a copied ID.

All arithmetic in this file is :class:`fractions.Fraction`.  Results are
research diagnostics with no proof or verdict authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
from typing import Mapping, Optional, Sequence, Tuple


Q = Fraction
Vector = Tuple[Q, ...]
Inequality = Tuple[Vector, Q]
Equality = Tuple[Vector, Q]


@dataclass(frozen=True)
class SharedPhaseAuditResult:
    """Exact bounds and node counts for one controlled shared-phase audit."""

    status: str
    rival_ids: Tuple[int, ...]
    root_upper: Tuple[Q, ...]
    shared_phase_upper: Optional[Tuple[Q, ...]]
    independent_phase_upper: Optional[Tuple[Q, ...]]
    one_sided_copy_upper: Optional[Tuple[Q, ...]]
    predicate_hull_upper: Q
    shared_tree_nodes: int
    independent_tree_nodes: int
    proof_authority: bool = False


def _dot(left: Sequence[Q], right: Sequence[Q]) -> Q:
    return sum((a * b for a, b in zip(left, right)), Q(0))


def _solve_square(
    rows: Sequence[Sequence[Q]], rhs: Sequence[Q]
) -> Optional[Vector]:
    """Solve one exact square system, returning ``None`` when singular."""

    n = len(rows)
    if n == 0 or len(rhs) != n or any(len(row) != n for row in rows):
        raise ValueError("expected one nonempty square system")
    work = [list(row) + [value] for row, value in zip(rows, rhs)]
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            return None
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                left - factor * right
                for left, right in zip(work[row], work[column])
            ]
    return tuple(work[row][-1] for row in range(n))


def _vertices(
    *,
    width: int,
    inequalities: Sequence[Inequality],
    equalities: Sequence[Equality] = (),
) -> Tuple[Vector, ...]:
    """Enumerate vertices of a small bounded rational polytope."""

    if width <= 0:
        raise ValueError("width must be positive")
    if any(len(row) != width for row, _rhs in inequalities):
        raise ValueError("inequality width mismatch")
    if any(len(row) != width for row, _rhs in equalities):
        raise ValueError("equality width mismatch")
    equality_rank = len(equalities)
    if equality_rank > width:
        raise ValueError("too many equalities for the audit polytope")
    active_needed = width - equality_rank
    points = set()
    for chosen in itertools.combinations(inequalities, active_needed):
        rows = [row for row, _rhs in equalities] + [
            row for row, _rhs in chosen
        ]
        rhs = [value for _row, value in equalities] + [
            value for _row, value in chosen
        ]
        point = _solve_square(rows, rhs)
        if point is None:
            continue
        if any(_dot(row, point) > value for row, value in inequalities):
            continue
        if any(_dot(row, point) != value for row, value in equalities):
            continue
        points.add(point)
    if not points:
        raise ValueError("audit polytope has no enumerated vertex")
    return tuple(sorted(points))


def _duplicate_triangle_constraints() -> Tuple[Inequality, ...]:
    """Triangle relaxation in variables ``(x, a, b)``."""

    return (
        ((Q(1), Q(0), Q(0)), Q(1)),
        ((Q(-1), Q(0), Q(0)), Q(1)),
        ((Q(0), Q(-1), Q(0)), Q(0)),
        ((Q(1), Q(-1), Q(0)), Q(0)),
        ((Q(-1), Q(2), Q(0)), Q(1)),
        ((Q(0), Q(0), Q(-1)), Q(0)),
        ((Q(1), Q(0), Q(-1)), Q(0)),
        ((Q(-1), Q(0), Q(2)), Q(1)),
    )


def _phase_constraints(active: bool, *, both_copies: bool) -> Tuple[
    Tuple[Inequality, ...], Tuple[Equality, ...]
]:
    inequalities = list(_duplicate_triangle_constraints())
    if active:
        # x >= 0 and a = x.  A complete semantic split also fixes b = x.
        inequalities.append(((Q(-1), Q(0), Q(0)), Q(0)))
        equalities = [((Q(-1), Q(1), Q(0)), Q(0))]
        if both_copies:
            equalities.append(((Q(-1), Q(0), Q(1)), Q(0)))
    else:
        # x <= 0 and a = 0.  A complete semantic split also fixes b = 0.
        inequalities.append(((Q(1), Q(0), Q(0)), Q(0)))
        equalities = [((Q(0), Q(1), Q(0)), Q(0))]
        if both_copies:
            equalities.append(((Q(0), Q(0), Q(1)), Q(0)))
    return tuple(inequalities), tuple(equalities)


def _upper(
    vertices: Sequence[Vector],
    coefficients: Sequence[Q],
    threshold: Q,
) -> Q:
    return max(_dot(coefficients, point) - threshold for point in vertices)


def predicate_disjunction_hull_upper(
    *,
    relaxation_vertices: Sequence[Sequence[Q]],
    rival_forms: Sequence[Tuple[Sequence[Q], Q]],
) -> Q:
    """Exact support of the rival-selector convex hull.

    The vertices of the disjunctive embedding are
    ``(e_r, z, f_r(z))``.  A linear maximum over their convex hull is attained
    at one such vertex, so enumeration is both an oracle and a concise proof
    that predicate-only convexification cannot beat exact per-rival support
    over the same relaxation.
    """

    points = tuple(
        tuple(Q(value) for value in point)
        for point in relaxation_vertices
    )
    if not points or not rival_forms:
        raise ValueError("at least one relaxation point and rival are required")
    width = len(points[0])
    if width == 0 or any(len(point) != width for point in points):
        raise ValueError("malformed relaxation vertices")
    values = []
    for coefficients, threshold in rival_forms:
        row = tuple(Q(value) for value in coefficients)
        if len(row) != width:
            raise ValueError("rival form width mismatch")
        values.extend(_dot(row, point) - Q(threshold) for point in points)
    return max(values)


def audit_duplicate_relu_shared_phase(
    *,
    rival_ids: Sequence[int] = (10, 20),
    shared_stable_id: bool = True,
) -> SharedPhaseAuditResult:
    """Run the exact two-rival discriminator.

    A false ``shared_stable_id`` models a copied residual/preactivation whose
    semantic equality has not been independently established.  The candidate
    then fails closed and returns no shared or independent phase certificate.
    """

    ids = tuple(int(value) for value in rival_ids)
    if len(ids) != 2 or len(set(ids)) != 2 or min(ids) < 0:
        raise ValueError("the audit requires two unique nonnegative rival ids")
    threshold = Q(1, 10)
    forms_by_id: Mapping[int, Vector] = {
        ids[0]: (Q(0), Q(1), Q(-1)),
        ids[1]: (Q(0), Q(-1), Q(1)),
    }
    root_vertices = _vertices(
        width=3, inequalities=_duplicate_triangle_constraints()
    )
    root = tuple(
        _upper(root_vertices, forms_by_id[rival], threshold)
        for rival in ids
    )
    hull = predicate_disjunction_hull_upper(
        relaxation_vertices=root_vertices,
        rival_forms=[
            (forms_by_id[rival], threshold)
            for rival in ids
        ],
    )
    if not shared_stable_id:
        return SharedPhaseAuditResult(
            status="wrong_copy_fail_closed",
            rival_ids=ids,
            root_upper=root,
            shared_phase_upper=None,
            independent_phase_upper=None,
            one_sided_copy_upper=None,
            predicate_hull_upper=hull,
            shared_tree_nodes=0,
            independent_tree_nodes=0,
        )

    shared_children = []
    one_sided_children = []
    for active in (False, True):
        inequalities, equalities = _phase_constraints(
            active, both_copies=True
        )
        shared_children.append(
            _vertices(
                width=3,
                inequalities=inequalities,
                equalities=equalities,
            )
        )
        inequalities, equalities = _phase_constraints(
            active, both_copies=False
        )
        one_sided_children.append(
            _vertices(
                width=3,
                inequalities=inequalities,
                equalities=equalities,
            )
        )
    shared_upper = tuple(
        max(
            _upper(child, forms_by_id[rival], threshold)
            for child in shared_children
        )
        for rival in ids
    )
    one_sided_upper = tuple(
        max(
            _upper(child, forms_by_id[rival], threshold)
            for child in one_sided_children
        )
        for rival in ids
    )
    # Separate per-rival trees enumerate the same two semantic child domains;
    # only their node objects are duplicated.
    independent_upper = shared_upper
    return SharedPhaseAuditResult(
        status="exact_shared_phase_cover",
        rival_ids=ids,
        root_upper=root,
        shared_phase_upper=shared_upper,
        independent_phase_upper=independent_upper,
        one_sided_copy_upper=one_sided_upper,
        predicate_hull_upper=hull,
        shared_tree_nodes=3,
        independent_tree_nodes=3 * len(ids),
    )


__all__ = [
    "SharedPhaseAuditResult",
    "audit_duplicate_relu_shared_phase",
    "predicate_disjunction_hull_upper",
]
