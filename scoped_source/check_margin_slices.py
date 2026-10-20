"""Exact closed-slice supports on the original, explicitly relaxed shared HZ.

Standard library only. Rational slab rows never enter a binary64 HZ serializer.
This enables a strong comparator; it is not a new domain or a network proof.
"""
from fractions import Fraction as F

from scoped_source.hz_request_store import require
from scoped_source.check_gate_margin_rows import MARGIN_PROFILE, capacity, freeze
from scoped_source.check_margin_hz_source import (
    inspect_property, parameter, objective as margin_objective, _accumulate,
)
from scoped_source.check_native_margin_support import box, stored
from scoped_source.check_margin_support import sigmoid_range, MAX_CELLS

PLAN = "EXACT_CLOSED_MARGIN_PARTITION_V1"
CERT = "EXACT_CLOSED_MARGIN_SUPPORT_V1"
GRADE = "CHECKED_SLICE_SUPPORT_ON_GIVEN_HZ_CONTINUOUS_RELAXATION"


def partition(parsed, cuts, work):
    """Caller-owned rational interior cuts; cover the entire factor-box range."""
    require(type(cuts) is tuple and len(cuts) < MAX_CELLS, "finite caller-owned slice cuts")
    lo, hi = box(parsed["rows"]["m"], work)
    inside = tuple(parameter(v, work) for v in cuts)
    require(all(lo < v < hi for v in inside)
            and all(x < y for x, y in zip(inside, inside[1:])), "strict ordered interior slice cuts")
    points = (lo,)+inside+(hi,)
    grid = tuple(zip(points, points[1:]))
    hz = parsed["joint"]
    require(len(hz["b"])+len(hz["ub"])+2 <= capacity(MARGIN_PROFILE)[1],
            "derived slice row capacity")
    cells = []
    m0, coefficients = parsed["rows"]["m"]
    for index, (x, y) in enumerate(grid):
        work.visit("row_visits", 2)
        cells.append(dict(index=index, margin=[stored(x), stored(y)],
                          gate=list(map(stored, sigmoid_range(x, y, work))),
                          slab_rhs=[stored(work.exact(y-m0)), stored(work.exact(m0-x))]))
    value = dict(schema=PLAN, source=parsed["source"], property=parsed["property"],
                 cuts=list(map(stored, inside)), margin=[stored(lo), stored(hi)], cells=cells,
                 slab_order="M_LE_UPPER_MINUS_CENTER_THEN_MINUS_M_LE_CENTER_MINUS_LOWER",
                 margin_coefficients=[[j, stored(v)] for j, v in sorted(coefficients.items())],
                 gate_enclosure="MONOTONIC_SIGN_AND_QUARTER_LIPSCHITZ_V1",
                 relaxed_binaries=hz["nb"])
    raw, identity = freeze(value, work, profile=MARGIN_PROFILE)
    return value, identity


def prepare(raw, view, *, expected_source, expected_property, cuts, work):
    parsed = inspect_property(raw, view, expected_source=expected_source,
                              expected_property=expected_property, work=work)
    plan, identity = partition(parsed, cuts, work)
    return parsed, plan, identity


def goal(spec, work):
    require(type(spec) is tuple and len(spec) == 2, "typed slice goal")
    kind, value = spec[0], parameter(spec[1], work)
    require((kind == "mix" and 0 <= value <= 1)
            or (kind == "difference" and value in (-1, 1)), "slice objective kind/range")
    return kind, value


def objective(parsed, spec, work):
    kind, value = goal(spec, work)
    if kind == "mix":
        return margin_objective(parsed, value, F(0), work)
    center, entries = parsed["rows"]["d"]
    result = {}
    for j, coefficient in entries.items():
        work.visit("coefficient_visits")
        result[j] = work.exact(value*coefficient)
    return work.exact(value*center), result


def endpoint_goals(cell, work, *, differences=False):
    lo, hi = (parameter(v, work) for v in cell["gate"])
    result = (("mix", lo),) if lo == hi else (("mix", lo), ("mix", hi))
    return result+(("difference", F(1)), ("difference", F(-1))) if differences else result


def check(raw, view, record, *, expected_source, expected_property, cuts,
          expected_index, expected_goals, work):
    parsed, plan, identity = prepare(raw, view, expected_source=expected_source,
                                    expected_property=expected_property, cuts=cuts, work=work)
    require(type(expected_index) is int and 0 <= expected_index < len(plan["cells"]), "slice index")
    cell = plan["cells"][expected_index]
    require(type(record) is dict and set(record) == {
        "schema", "source", "property", "partition", "cell", "supports", "grade",
    } and record["schema"] == CERT and record["source"] == expected_source
            and record["property"] == expected_property and record["grade"] == GRADE,
            "slice record identity/grade")
    require(record["partition"] == identity and record["cell"] == cell
            and type(record["cell"]["index"]) is int, "caller-bound closed slice")
    require(type(expected_goals) is tuple and 1 <= len(expected_goals) <= 8,
            "finite slice goal roster")
    goals = tuple(goal(s, work) for s in expected_goals)
    require(len(set(goals)) == len(goals), "unique slice goals")
    require(type(record["supports"]) is list and len(record["supports"]) == len(goals),
            "complete slice supports")
    hz, results = parsed["joint"], []
    m0, margin = parsed["rows"]["m"]
    x, y = (parameter(v, work) for v in cell["margin"])
    for spec, item in zip(goals, record["supports"]):
        require(type(item) is dict and set(item) == {"goal", "y", "z", "slice_duals", "lower"}
                and item["goal"] == [spec[0], stored(spec[1])], "slice objective binding")
        lower, residual = objective(parsed, spec, work)
        for key, ck, bk, rhs in (("y", "Auc", "Aub", "ub"), ("z", "Ac", "Ab", "b")):
            require(type(item[key]) is list and len(item[key]) == len(hz[rhs]), "slice original dual rows")
            for i, value in enumerate(item[key]):
                work.visit("row_visits")
                dual = parameter(value, work)
                require(key == "z" or dual <= 0, "slice inequality dual sign")
                lower = work.exact(lower+work.exact(dual*hz[rhs][i]))
                for name, shift in ((ck, 0), (bk, hz["nc"])):
                    for j, a in hz[name][i].items():
                        work.visit("coefficient_visits")
                        _accumulate(residual, shift+j, work.exact(-dual*a), work)
        require(type(item["slice_duals"]) is list and len(item["slice_duals"]) == 2,
                "two ordered slab duals")
        # Independent reconstruction; never trust producer CSR/RHS or saved plan rows.
        for value, sign, rhs in zip(item["slice_duals"], (1, -1),
                                    (work.exact(y-m0), work.exact(m0-x))):
            work.visit("row_visits")
            dual = parameter(value, work)
            require(dual <= 0, "slice inequality dual sign")
            lower = work.exact(lower+work.exact(dual*rhs))
            for j, a in margin.items():
                work.visit("coefficient_visits")
                _accumulate(residual, j, work.exact(-sign*work.exact(dual*a)), work)
        for value in residual.values():
            lower = work.exact(lower-abs(value))
        require(parameter(item["lower"], work) == lower, "slice exact residual lower mismatch")
        results.append(dict(goal=item["goal"], lower=stored(lower)))
    work.tick()
    return dict(source=expected_source, property=expected_property, partition=identity,
                cell=cell, results=results, grade=GRADE, relaxed_binaries=hz["nb"],
                complete_request=False, scalar_aggregate_checked=False, feasibility_checked=False)


def aggregate(raw, view, records, *, expected_source, expected_property, cuts,
              differences, work):
    """All closed cells, no infeasible-cell omission; optional typed N1 ranges."""
    require(type(differences) is bool, "explicit slice range roster")
    parsed, plan, identity = prepare(raw, view, expected_source=expected_source,
                                    expected_property=expected_property, cuts=cuts, work=work)
    require(type(records) is list and len(records) == len(plan["cells"]), "complete closed partition")
    results = []
    for index, (cell, record) in enumerate(zip(plan["cells"], records)):
        goals = endpoint_goals(cell, work, differences=differences)
        receipt = check(raw, view, record, expected_source=expected_source,
                        expected_property=expected_property, cuts=cuts, expected_index=index,
                        expected_goals=goals, work=work)
        lowers = [parameter(s["lower"], work) for s in receipt["results"]]
        lower = min(v for spec, v in zip(goals, lowers) if spec[0] == "mix")
        entry = dict(index=index, lower=stored(lower))
        if differences:
            dlo, dhi = box(parsed["rows"]["d"], work)
            dlo, dhi = max(dlo, lowers[-2]), min(dhi, -lowers[-1])
            require(dlo <= dhi, "inverted difference range requires separate infeasibility proof")
            entry["difference_range"] = [stored(dlo), stored(dhi)]
        results.append(entry)
    lower = min(parameter(v["lower"], work) for v in results)
    work.tick()
    return dict(source=expected_source, property=expected_property, partition=identity,
                cells=results, lower=stored(lower),
                status="POSITIVE_GIVEN_SOURCE" if lower > 0 else "NONPOSITIVE_BOUND",
                grade="CHECKED_SLICED_MARGIN_LOWER_ON_GIVEN_HZ_CONTINUOUS_RELAXATION",
                complete_request=False, scalar_aggregate_checked=True,
                route_reachability_checked=False, feasibility_checked=False)
