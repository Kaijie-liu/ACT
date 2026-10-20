"""Default-off rational N1/M3 LP adapter; existing candidate kernel unchanged."""
from fractions import Fraction as F

from act.back_end.moe.margin_hz_source import _csr
from act.back_end.moe.gate_margin_rows import snapshot
from scoped_source.hz_request_store import require
from scoped_source.check_gate_margin_rows import MARGIN_PROFILE, freeze
from scoped_source.check_margin_hz_source import parameter
from scoped_source.check_native_margin_support import stored
from scoped_source import check_margin_product as checker


def build(parsed, cell, difference, sign, work):
    """Bounded version of the existing direct rational McCormick recipe."""
    hz = parsed["joint"]; n = hz["nc"]+hz["nb"]
    d0, d = parsed["rows"]["d"]; m0, m = parsed["rows"]["m"]
    u0, u = parsed["rows"]["b"]
    a, b = (parameter(v, work) for v in cell["gate"]); l, h = difference
    x, y = (parameter(v, work) for v in cell["margin"])
    A, E = _csr(hz, "Auc", "Aub", work), _csr(hz, "Ac", "Ab", work)
    A["shape"][1] = E["shape"][1] = n+2
    rhs = list(map(stored, hz["ub"]))
    def append(row, value):
        work.visit("row_visits")
        for j, v in sorted(row.items()):
            work.visit("coefficient_visits")
            if v: A["indices"].append(j); A["data"].append(stored(v))
        A["indptr"].append(len(A["data"])); A["shape"][0] += 1
        rhs.append(stored(value))
    append(m, work.exact(y-m0)); append({j: -v for j, v in m.items()}, work.exact(m0-x))
    for s, t, orientation in ((a, l, 1), (b, h, 1), (b, l, -1), (a, h, -1)):
        row = {n: work.exact(orientation*t), n+1: F(-orientation)}
        for j, v in d.items():
            work.visit("coefficient_visits"); row[j] = work.exact(orientation*work.exact(s*v))
        append(row, work.exact(orientation*work.exact(s*work.exact(t-d0))))
    if sign:
        side = sign["side"]; row = {n: F(side)}
        for j, v in m.items():
            work.visit("coefficient_visits"); row[j] = work.exact(-side*v/4)
        append(row, work.exact(side*work.exact(F(1, 2)+work.exact(m0/4))))
        append({n: F(-side)}, F(-side, 2))
    corners = [work.exact(s*t) for s in (a, b) for t in (l, h)]
    work.visit("coefficient_visits", 3*(n+2))
    return dict(matrix_format="csr_v1", A=A, E=E, b=rhs, h=list(map(stored, hz["b"])),
                c=[stored(u.get(j, F(0))) for j in range(n)]+["0", "1"], offset=stored(u0),
                lower=["-1"]*n+[stored(a), stored(min(corners))],
                upper=["1"]*n+[stored(b), stored(max(corners))])


def proposed_lower(lp, y, z, work):
    """Producer finite-box arithmetic, independently repeated on reconstructed LP."""
    r = [parameter(v, work) for v in lp["c"]]; value = parameter(lp["offset"], work)
    for key, rhs, multipliers in (("A", "b", y), ("E", "h", z)):
        require(len(multipliers) == len(lp[rhs]), "producer product dual coverage")
        a = lp[key]
        for i, multiplier in enumerate(multipliers):
            work.visit("row_visits"); multiplier = parameter(multiplier, work)
            require(key == "E" or multiplier <= 0, "producer product dual sign")
            value = work.exact(value+work.exact(multiplier*parameter(lp[rhs][i], work)))
            for pos in range(a["indptr"][i], a["indptr"][i+1]):
                work.visit("coefficient_visits"); j = a["indices"][pos]
                r[j] = work.exact(r[j]-work.exact(multiplier*parameter(a["data"][pos], work)))
    for j, v in enumerate(r):
        work.visit("coefficient_visits")
        endpoint = lp["lower"][j] if v >= 0 else lp["upper"][j]
        value = work.exact(value+work.exact(v*parameter(endpoint, work)))
    return parameter(value, work)


def propose(bound, view, ranges, *, expected_property, cuts, index, mode, work,
            provider="subgradient_128", query_seconds=None, observe=None):
    require(provider in ("subgradient_128", "highs")
            and (provider == "highs" or (query_seconds is None and observe is None)),
            "explicit fixed product provider")
    parsed, plan, identity, difference, sign = checker.premises(bound.payload, view, ranges,
        expected_source=bound.sha256, expected_property=expected_property, cuts=cuts, index=index, mode=mode, work=work)
    require(snapshot(bound.output_hz, work, profile=MARGIN_PROFILE) == parsed["obj"]["augmented"],
            "live product source changed before proposal")
    cell = plan["cells"][index]; lp = build(parsed, cell, difference, sign, work)
    _, lp_id = freeze(lp, work, profile=MARGIN_PROFILE)
    if provider == "highs":
        from act.back_end.moe.margin_lp_candidates import columns
        exact = [(parameter(lp["offset"], work), [parameter(v, work) for v in lp["c"]])]
        ys, zs, native_report = columns(lp, exact, work=work, query_seconds=query_seconds, observe=observe)
    else:
        from act.back_end.moe.batched_support import _candidate_columns
        native = dict(lp, **{key: [float(F(v)) for v in lp[key]] for key in ("b", "h", "lower", "upper")})
        ys, zs = _candidate_columns(native, [[float(F(v)) for v in lp["c"]]],
                                    [float(F(lp["offset"]))], deadline=work.deadline)
    require(len(ys) == len(zs) == 1, "complete product candidate")
    y, z = ys[0], zs[0]; lower = proposed_lower(lp, y, z, work)
    zero_y, zero_z = [0]*len(y), [0]*len(z)
    zero = proposed_lower(lp, zero_y, zero_z, work)
    choice = "NATIVE_DUAL"
    if zero > lower:
        y, z, lower = zero_y, zero_z, zero
        choice = "CHECKED_ZERO_DUAL_STRONGER"
    cert = dict(lp_sha256=lp_id, y=[stored(parameter(v, work)) for v in y],
                z=[stored(parameter(v, work)) for v in z], lower=stored(lower))
    record = dict(schema=checker.SCHEMA, source=bound.sha256, property=expected_property,
                  partition=identity, cell=cell, mode=mode, ranges=ranges,
                  difference=list(map(stored, difference)), sign=sign, lp=lp, certificate=cert, grade=checker.GRADE)
    receipt = checker.check(bound.payload, view, record, expected_source=bound.sha256,
        expected_property=expected_property, cuts=cuts, index=index, mode=mode, work=work)
    require(snapshot(bound.output_hz, work, profile=MARGIN_PROFILE) == parsed["obj"]["augmented"],
            "live product source changed during proposal")
    work.tick()
    result = dict(record=record, receipt=receipt, native_candidate_calls=1, objective_slots=1,
                iterations=128, device="cpu", hard_budget_supervision=False)
    if provider == "highs":
        result.update(iterations=None, native_report=native_report, selected_candidates=[choice])
    return result
