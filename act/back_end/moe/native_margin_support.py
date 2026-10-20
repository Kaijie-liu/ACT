"""Default-off sparse conditional-support aggregation, no production dispatch.

Uses the existing fixed CPU candidate generator on the original P; the new
composition joins checked supports to the already specified scalar cell rule.
"""
from scoped_source.hz_request_store import require
from scoped_source.check_margin_support import sigmoid_range
from scoped_source.check_margin_hz_source import parameter
from scoped_source import check_native_margin_support as checker
from act.back_end.moe.margin_hz_source import propose_supports


def build(plan, candidates, work):
    """Candidate scalar record; independent reception does not trust this math."""
    anchors = tuple(parameter(v, work) for v in plan["anchors"])
    require(len(anchors) in (1, 2) and len(candidates) == len(plan["goals"]), "finite producer roster")
    supports = []
    for goal, item in zip(plan["goals"], candidates):
        t, nu = (parameter(v, work) for v in goal)
        require(parameter(item["t"], work) == t and parameter(item["nu"], work) == nu, "producer support order")
        supports.append((t, nu, parameter(item["lower"], work)))
    result_cells = []
    for cell in plan["grid"]:
        work.visit("row_visits")
        x, y = (parameter(v, work) for v in cell)
        gl, gh = sigmoid_range(x, y, work)
        values = []
        for t in anchors:
            values.append(max(work.exact(lower-work.exact(nu*(y if nu >= 0 else x)))
                              for a, nu, lower in supports if a == t))
        if len(anchors) == 1:
            lower = values[0]
        else:
            l, h = anchors
            slope = work.exact(work.exact(values[1]-values[0])/work.exact(h-l))
            g = gl if slope >= 0 else gh
            lower = work.exact(values[0]+work.exact(work.exact(g-l)*slope))
        result_cells.append(dict(x=checker.stored(x), y=checker.stored(y),
                                 gate=[checker.stored(gl), checker.stored(gh)], lower=checker.stored(lower)))
    require(bool(result_cells), "nonempty covered scalar grid")
    lower = min(parameter(item["lower"], work) for item in result_cells)
    work.tick()
    return dict(schema=checker.CERT, source=plan["source"], property=plan["property"], plan=plan,
                supports=candidates, cells=result_cells,
                lower=checker.stored(lower), grade=checker.GRADE)


def support_property(bound, view, *, expected_property, mode, cells, work):
    """One source, property, candidate roster and scalar check under caller budget.

    Caller installs hard supervision and owns complete routed request coverage.
    This API never sets the verifier's SAFE/UNSAFE status or changes solver gates.
    """
    import time
    start = time.monotonic()
    plan, _, goals, _ = checker.prepare(bound.payload, view, expected_source=bound.sha256,
                                        expected_property=expected_property, mode=mode, cells=cells, work=work)
    planned = time.monotonic()
    generated = propose_supports(bound, view, expected_property=expected_property, goals=goals, work=work)
    supported = time.monotonic()
    record = build(plan, generated["candidates"], work)
    built = time.monotonic()
    receipt = checker.check(bound.payload, view, record, expected_source=bound.sha256,
                            expected_property=expected_property, expected_mode=mode,
                            expected_cells=cells, work=work)
    checked = time.monotonic()
    work.tick()
    return dict(record=record, receipt=receipt, cost_seconds=dict(plan=planned-start,
                supports_and_first_check=supported-planned, scalar_build=built-supported,
                full_independent_check=checked-built, total=checked-start),
                native_candidate_calls=1, objective_slots=len(goals),
                algorithm=generated["algorithm"], iterations=generated["iterations"],
                hard_budget_supervision=False, complete_request=False)
