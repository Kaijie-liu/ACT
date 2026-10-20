"""Sparse margin plan and independent scalar check on original given HZs.

No optimizer or ACT dependency. A positive aggregate is conditional on the
given continuous HZ outer approximation, not a complete routed-network proof.
"""
from fractions import Fraction as F
from functools import cmp_to_key

from scoped_source.hz_request_store import require, rational
from scoped_source.check_margin_support import sigmoid_range, MAX_CELLS
from scoped_source.check_margin_hz_source import inspect_property, check_supports, parameter

PLAN = "SPARSE_MARGIN_PLAN_V1"
CERT = "NATIVE_MARGIN_SCALAR_CERTIFICATE_V1"
GRADE = "CHECKED_MARGIN_LOWER_ON_GIVEN_HZ_CONTINUOUS_RELAXATION"


def stored(value):
    """Serialized plan/claim values keep the original4096-bit contract."""
    return str(rational(str(value)))


def box(row, work):
    center, coefficients = row
    radius = F(0)
    for value in coefficients.values():
        work.visit("coefficient_visits")
        radius = work.exact(radius+abs(value))
    return work.exact(center-radius), work.exact(center+radius)


def median_tilt(rows, t, work):
    """Same closest-to-zero weighted median; sparse, metered exact comparisons.

    Only margin coordinates affect selection. All other coordinates and all
    constraints are retained by the support checker. This is not a new search.
    """
    b, d, m = (rows[k][1] for k in ("b", "d", "m"))
    weights = {}
    for j, value in m.items():
        work.visit("coefficient_visits")
        require(value != 0, "canonical nonzero sparse margin")
        coefficient = work.exact(b.get(j, F(0))+work.exact(t*d.get(j, F(0))))
        ratio = work.exact(-coefficient/value)
        weights[ratio] = work.exact(weights.get(ratio, F(0))+abs(value))
    if not weights:
        return F(0)
    def compare(a, b):
        work.visit("coefficient_visits")
        delta = work.exact(a-b)
        return (delta > 0)-(delta < 0)
    ordered = sorted(weights, key=cmp_to_key(compare))
    total = F(0)
    for weight in weights.values():
        total = work.exact(total+weight)
    cumulative = F(0)
    for i, ratio in enumerate(ordered):
        cumulative = work.exact(cumulative+weights[ratio])
        twice = work.exact(2*cumulative)
        if twice >= total:
            right = ordered[i+1] if twice == total else ratio
            return min(max(F(0), ratio), right)
    raise AssertionError("positive sparse median weights")


def plan(parsed, mode, cells, work):
    require(mode in ("zero", "median"), "finite sparse margin mode")
    require(type(cells) is int and 1 <= cells <= MAX_CELLS, "finite sparse cell capacity")
    lo, hi = box(parsed["rows"]["m"], work)
    l, h = sigmoid_range(lo, hi, work)
    anchors = (l,) if l == h else (l, h)
    goals = []
    for t in anchors:
        goals.append((t, F(0)))
        if mode == "median" and l != h:
            nu = median_tilt(parsed["rows"], t, work)
            if nu:
                goals.append((t, nu))
    count = 1 if lo == hi else cells
    width = work.exact(hi-lo)
    cuts = [work.exact(lo+work.exact(width*F(i, count))) for i in range(count+1)]
    grid = tuple(zip(cuts, cuts[1:]))
    # No unbound margin/cut/range supplied by the candidate can enter this plan.
    result = dict(schema=PLAN, source=parsed["source"], property=parsed["property"],
                  mode=mode, requested_cells=cells, margin=[stored(lo), stored(hi)],
                  anchors=list(map(stored, anchors)),
                  goals=[[stored(t), stored(nu)] for t, nu in goals],
                  grid=[[stored(x), stored(y)] for x, y in grid],
                  gate_enclosure="MONOTONIC_SIGN_AND_QUARTER_LIPSCHITZ_V1",
                  margin_enclosure="ORIGINAL_SHARED_FACTOR_BOX_MINUS_PLUS_ONE")
    work.tick()
    return result, anchors, tuple(goals), grid


def prepare(raw, view, *, expected_source, expected_property, mode, cells, work):
    parsed = inspect_property(raw, view, expected_source=expected_source,
                              expected_property=expected_property, work=work)
    return plan(parsed, mode, cells, work)


def check(raw, view, record, *, expected_source, expected_property,
          expected_mode, expected_cells, work):
    """Reconstruct all source rows, supports and closed cells before accepting."""
    planned, anchors, goals, grid = prepare(raw, view, expected_source=expected_source,
                                           expected_property=expected_property, mode=expected_mode,
                                           cells=expected_cells, work=work)
    require(type(record) is dict and set(record) == {
        "schema", "source", "property", "plan", "supports", "cells", "lower", "grade",
    } and record["schema"] == CERT and record["source"] == expected_source
            and record["property"] == expected_property and record["grade"] == GRADE,
            "scalar certificate identity/grade")
    require(record["plan"] == planned, "caller-bound sparse plan")
    # Recompute exact residuals; neither a producer claim nor an earlier receipt
    # can replace this original-coefficient traversal.
    receipt = check_supports(raw, view, record["supports"], expected_source=expected_source,
                             expected_property=expected_property, expected_goals=goals, work=work)
    supports = [(t, nu, parameter(item["lower"], work))
                for (t, nu), item in zip(goals, receipt["results"])]
    require(type(record["cells"]) is list and len(record["cells"]) == len(grid), "complete scalar cells")
    bounds = []
    for (x, y), item in zip(grid, record["cells"]):
        work.visit("row_visits")
        require(type(item) is dict and set(item) == {"x", "y", "gate", "lower"}, "scalar cell fields")
        require(parameter(item["x"], work) == x and parameter(item["y"], work) == y, "closed scalar coverage")
        gate = sigmoid_range(x, y, work)
        require(type(item["gate"]) is list and len(item["gate"]) == 2
                and tuple(parameter(v, work) for v in item["gate"]) == gate, "scalar sigmoid enclosure")
        require(anchors[0] <= gate[0] <= gate[1] <= anchors[-1], "scalar anchor coverage")
        endpoint_values = []
        for t in anchors:
            endpoint_values.append(max(work.exact(lb+min(work.exact(-nu*x), work.exact(-nu*y)))
                                       for a, nu, lb in supports if a == t))
        if len(anchors) == 1:
            lower = endpoint_values[0]
        else:
            l, h = anchors
            width = work.exact(h-l)
            lower = min(work.exact(work.exact(
                work.exact(work.exact(h-g)*endpoint_values[0])
                +work.exact(work.exact(g-l)*endpoint_values[1]))/width) for g in gate)
        require(parameter(item["lower"], work) == lower, "scalar cell lower mismatch")
        bounds.append(lower)
    lower = min(bounds)
    require(parameter(record["lower"], work) == lower, "scalar aggregate lower mismatch")
    work.tick()
    return dict(source=expected_source, property=expected_property, grade=GRADE,
                lower=stored(lower), status="POSITIVE_GIVEN_SOURCE" if lower > 0 else "NONPOSITIVE_BOUND",
                mode=expected_mode, supports=len(goals), cells=len(grid),
                relaxed_binaries=receipt["relaxed_binaries"], scalar_aggregate_checked=True,
                complete_request=False, route_reachability_checked=False,
                error_bounds=dict(anchor="NOT_QUANTIFIED", finite_tilt="UNKNOWN",
                                  candidate_support="UNKNOWN", sigmoid="VALID_LOOSE_ENCLOSURE",
                                  scalar_minimization="LOWER_ONLY_NO_OPTIMALITY_GAP"))
