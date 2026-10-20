"""Optional exact finite margin support prototype; not wired into verification.

Loaded directly for stdlib controls, avoiding ACT package numerical imports.
Caller-supplied candidate multipliers need independent exact checking. This
producer has no solver, source-lowering authority, or request SAFE transition.
"""
from fractions import Fraction as F
from scoped_source.hz_request_store import require
from scoped_source.check_margin_support import (
    CERT, GRADE, inspect, plan, sigmoid_range, number,
)


def build(raw_source, *, expected_source, mode, cells, duals, work):
    source = inspect(raw_source, expected_source, work)
    anchors, goals, grid = plan(source, mode, cells, work)
    require(type(duals) is list and len(duals) == len(goals), "candidate count")
    supports = []
    for (t, nu), candidate in zip(goals, duals):
        require(type(candidate) is tuple and len(candidate) == 2, "candidate fields")
        y, z = candidate
        require(len(y) == len(source["le"]) and len(z) == len(source["eq"])
                and all(type(v) is F for v in (*y, *z)), "candidate multiplier type/shape")
        for v in (*y, *z):
            work.exact(v)
        # Producer substitution; checker separately reconstructs the residual.
        target = [work.exact(b+work.exact(t*d)+work.exact(nu*m))
                  for b, d, m in zip(source["b"], source["d"], source["m"])]
        for multipliers, rows in ((y, source["le"]), (z, source["eq"])):
            for multiplier, row in zip(multipliers, rows):
                target[0] = work.exact(target[0]+work.exact(multiplier*row[0]))
                for i in range(1, len(target)):
                    target[i] = work.exact(target[i]-work.exact(multiplier*row[i]))
        value = target[0]
        for coefficient, (lo, hi) in zip(target[1:], source["bounds"]):
            chosen = lo if coefficient >= 0 else hi
            value = work.exact(value+work.exact(coefficient*chosen))
        supports.append(dict(t=str(t), nu=str(nu), y=list(map(str, y)), z=list(map(str, z)),
                             lower=str(value)))
    result_cells = []
    for x, y in grid:
        work.visit("row_visits")
        gl, gh = sigmoid_range(x, y, work)
        values = []
        for t in anchors:
            candidates = []
            for row in supports:
                if number(row["t"], work) == t:
                    nu, lower = number(row["nu"], work), number(row["lower"], work)
                    endpoint = y if nu >= 0 else x
                    candidates.append(work.exact(lower-work.exact(nu*endpoint)))
            values.append(max(candidates))
        if len(anchors) == 1:
            lower = values[0]
        else:
            l, h = anchors
            slope = work.exact(work.exact(values[1]-values[0])/work.exact(h-l))
            chosen = gl if slope >= 0 else gh
            lower = work.exact(values[0]+work.exact(work.exact(chosen-l)*slope))
        result_cells.append(dict(x=str(x), y=str(y), gate=[str(gl), str(gh)], lower=str(lower)))
    work.tick()
    return dict(schema=CERT, source=expected_source, mode=mode, cell_count=cells,
                supports=supports, cells=result_cells,
                lower=str(min(number(v["lower"], work) for v in result_cells)), grade=GRADE)
