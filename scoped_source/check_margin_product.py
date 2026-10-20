"""Bounded original-source check of sliced McCormick and M3 product LPs.

No native libraries. This checks a given-HZ continuous relaxation, not network
lowering. Lambda/product coordinates use their own finite bounds, NOT [-1,1].
"""
from fractions import Fraction as F

from scoped_source.hz_request_store import require
from scoped_source.check_gate_margin_rows import MARGIN_PROFILE, capacity, freeze, sign_premise
from scoped_source.check_margin_hz_source import parameter
from scoped_source.check_native_margin_support import box, stored
from scoped_source import check_margin_slices as slices

SCHEMA = "CHECKED_SLICED_PRODUCT_LP_V1"
GRADE = "CHECKED_PRODUCT_LOWER_ON_GIVEN_HZ_CONTINUOUS_RELAXATION"
RANGE_GOALS = (("difference", F(1)), ("difference", F(-1)))


def premises(raw, view, ranges, *, expected_source, expected_property, cuts, index, mode, work):
    require(mode in ("N1", "M3"), "fixed product mode")
    parsed, plan, identity = slices.prepare(raw, view, expected_source=expected_source,
        expected_property=expected_property, cuts=cuts, work=work)
    require(type(index) is int and 0 <= index < len(plan["cells"]), "product cell index")
    checked = slices.check(raw, view, ranges, expected_source=expected_source,
        expected_property=expected_property, cuts=cuts, expected_index=index,
        expected_goals=RANGE_GOALS, work=work)
    dlo, dhi = box(parsed["rows"]["d"], work)
    dlo = max(dlo, parameter(checked["results"][0]["lower"], work))
    dhi = min(dhi, -parameter(checked["results"][1]["lower"], work))
    require(dlo <= dhi, "product range inverted without infeasibility proof")
    cell = plan["cells"][index]
    x, y = (parameter(v, work) for v in cell["margin"])
    m0, margin = parsed["rows"]["m"]
    sign = None
    if mode == "M3":
        sign = (dict(side=1, kind="CLOSED_CELL") if x >= 0 else
                dict(side=-1, kind="CLOSED_CELL") if y <= 0 else
                sign_premise(parsed["joint"], m0, margin, work))
    hz = parsed["joint"]
    require(len(hz["b"])+len(hz["ub"])+6+(2 if sign else 0) <= capacity(MARGIN_PROFILE)[1],
            "derived product row capacity")
    return parsed, plan, identity, (dlo, dhi), sign


def expected_lp(parsed, cell, difference, sign, work):
    """Checker-only reconstruction; never calls the producer builder."""
    hz = parsed["joint"]; n = hz["nc"]+hz["nb"]
    center, b = parsed["rows"]["b"]
    d0, d = parsed["rows"]["d"]; m0, m = parsed["rows"]["m"]
    a, h = (parameter(v, work) for v in cell["gate"])
    lo, hi = difference
    x, y = (parameter(v, work) for v in cell["margin"])
    rows, rhs = {}, {}
    for label, ck, bk, key in (("A", "Auc", "Aub", "ub"), ("E", "Ac", "Ab", "b")):
        rows[label], rhs[label] = [], list(hz[key])
        for cr, br in zip(hz[ck], hz[bk]):
            work.visit("row_visits")
            row = {}
            for block, shift in ((cr, 0), (br, hz["nc"])):
                for j, v in block.items():
                    work.visit("coefficient_visits"); row[j+shift] = v
            rows[label].append(row)
    rows["A"].extend((dict(m), {j: -v for j, v in m.items()}))
    rhs["A"].extend((work.exact(y-m0), work.exact(m0-x)))
    # Independently derive the two lower and two upper planes.
    factors = ((a, lo, -1), (h, hi, -1), (-h, -lo, 1), (-a, -hi, 1))
    offsets = (work.exact(a*work.exact(lo-d0)), work.exact(h*work.exact(hi-d0)),
               work.exact(h*work.exact(d0-lo)), work.exact(a*work.exact(d0-hi)))
    for (scale, lam, product), offset in zip(factors, offsets):
        work.visit("row_visits")
        row = {n: lam, n+1: F(product)}
        for j, v in d.items():
            work.visit("coefficient_visits"); row[j] = work.exact(scale*v)
        rows["A"].append(row); rhs["A"].append(offset)
    if sign:
        s = sign["side"]
        row = {n: F(s)}
        for j, v in m.items():
            work.visit("coefficient_visits"); row[j] = work.exact(-s*v/4)
        rows["A"].extend((row, {n: F(-s)}))
        rhs["A"].extend((work.exact(F(s, 2)+work.exact(s*m0/4)), F(-s, 2)))
    corners = [work.exact(g*v) for g in (a, h) for v in (lo, hi)]
    def csr(items):
        data, ids, ptr = [], [], [0]
        for row in items:
            work.visit("row_visits")
            for j, v in sorted(row.items()):
                work.visit("coefficient_visits")
                if v: ids.append(j); data.append(stored(v))
            ptr.append(len(data))
        return dict(shape=[len(items), n+2], data=data, indices=ids, indptr=ptr)
    work.visit("coefficient_visits", 3*(n+2))
    return dict(matrix_format="csr_v1", c=[stored(b.get(j, F(0))) for j in range(n)]+["0", "1"],
                offset=stored(center), lower=["-1"]*n+[stored(a), stored(min(corners))],
                upper=["1"]*n+[stored(h), stored(max(corners))],
                A=csr(rows["A"]), b=list(map(stored, rhs["A"])),
                E=csr(rows["E"]), h=list(map(stored, rhs["E"])))


def check(raw, view, record, *, expected_source, expected_property, cuts, index, mode, work):
    require(type(record) is dict and set(record) == {"schema", "source", "property", "partition", "cell",
        "mode", "ranges", "difference", "sign", "lp", "certificate", "grade"}
        and record["schema"] == SCHEMA and record["source"] == expected_source
        and record["property"] == expected_property and record["mode"] == mode
        and record["grade"] == GRADE, "product record identity/mode/grade")
    # Bound untrusted structures/bit lengths before comparison/traversal.
    freeze(record, work, profile=MARGIN_PROFILE)
    parsed, plan, identity, difference, sign = premises(raw, view, record["ranges"],
        expected_source=expected_source, expected_property=expected_property,
        cuts=cuts, index=index, mode=mode, work=work)
    cell = plan["cells"][index]
    require(record["partition"] == identity and record["cell"] == cell
            and type(record["cell"]["index"]) is int
            and record["difference"] == list(map(stored, difference))
            and freeze(record["sign"], work, profile=MARGIN_PROFILE)[0]
            == freeze(sign, work, profile=MARGIN_PROFILE)[0],
            "product domain/range/sign binding")
    lp = expected_lp(parsed, cell, difference, sign, work)
    raw_lp, lp_id = freeze(lp, work, profile=MARGIN_PROFILE)
    require(freeze(record["lp"], work, profile=MARGIN_PROFILE)[0] == raw_lp, "product exact LP construction")
    cert = record["certificate"]
    require(type(cert) is dict and set(cert) == {"lp_sha256", "y", "z", "lower"}
            and cert["lp_sha256"] == lp_id, "product LP certificate identity")
    residual = [parameter(v, work) for v in lp["c"]]
    lower = parameter(lp["offset"], work)
    for name, right, dualkey in (("A", "b", "y"), ("E", "h", "z")):
        matrix = lp[name]
        require(type(cert[dualkey]) is list and len(cert[dualkey]) == len(lp[right]), "product complete dual rows")
        for i, val in enumerate(cert[dualkey]):
            work.visit("row_visits"); dual = parameter(val, work)
            require(dualkey == "z" or dual <= 0, "product inequality dual sign")
            lower = work.exact(lower+work.exact(dual*parameter(lp[right][i], work)))
            for pos in range(matrix["indptr"][i], matrix["indptr"][i+1]):
                work.visit("coefficient_visits"); j = matrix["indices"][pos]
                residual[j] = work.exact(residual[j]-work.exact(dual*parameter(matrix["data"][pos], work)))
    for r, lo, hi in zip(residual, lp["lower"], lp["upper"]):
        work.visit("coefficient_visits")
        lower = work.exact(lower+min(work.exact(r*parameter(lo, work)), work.exact(r*parameter(hi, work))))
    require(parameter(cert["lower"], work) == lower, "product exact finite-box residual")
    work.tick()
    return dict(source=expected_source, property=expected_property, partition=identity, cell=index,
                mode=mode, lp_sha256=lp_id, lower=stored(lower), difference=list(map(stored, difference)),
                sign=sign, grade=GRADE, relaxed_binaries=parsed["joint"]["nb"],
                status="POSITIVE_GIVEN_SOURCE" if lower > 0 else "NONPOSITIVE_BOUND",
                complete_request=False, scalar_aggregate_checked=False, feasibility_checked=False)


def aggregate(raw, view, records, *, expected_source, expected_property, cuts, mode, work):
    parsed, plan, identity = slices.prepare(raw, view, expected_source=expected_source,
        expected_property=expected_property, cuts=cuts, work=work)
    require(type(records) is list and len(records) == len(plan["cells"]), "complete product partition")
    receipts = [check(raw, view, record, expected_source=expected_source, expected_property=expected_property,
                      cuts=cuts, index=index, mode=mode, work=work) for index, record in enumerate(records)]
    lower = min(parameter(r["lower"], work) for r in receipts)
    return dict(source=expected_source, property=expected_property, partition=identity, mode=mode,
                cells=receipts, lower=stored(lower), grade=GRADE,
                status="POSITIVE_GIVEN_SOURCE" if lower > 0 else "NONPOSITIVE_BOUND",
                complete_request=False, scalar_aggregate_checked=True, route_reachability_checked=False)
