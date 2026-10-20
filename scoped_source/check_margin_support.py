"""Exact checks of finite margin-conditioned supports on a GIVEN affine source.

No network lowering, optimizer, numerical dependency or production SAFE policy.
The caller supplies the immutable source identity and one shared Work/deadline.
"""
from fractions import Fraction as F
from scoped_source.hz_request_store import (
    compact, decode, digest, rational, require, sha,
)

SCHEMA = "MARGIN_SUPPORT_SOURCE_V1"
CERT = "MARGIN_SUPPORT_CERTIFICATE_V1"
GRADE = "CHECKED_LOWER_BOUND_ON_GIVEN_CONTINUOUS_SOURCE"
MAX_FACTORS, MAX_ROWS, MAX_CLASSES, MAX_CELLS = 16, 32, 8, 128
MEMBER_BYTES = 2**20


def number(x, work):
    work.visit("coefficient_visits")
    return work.exact(rational(x))


def dot(a, b, work):
    require(len(a) == len(b), "dot dimension")
    total = F(0)
    for x, y in zip(a, b):
        total = work.exact(total + work.exact(x*y))
    return total


def affine_box(row, bounds, work):
    lo = hi = row[0]
    for a, (l, h) in zip(row[1:], bounds):
        x, y = work.exact(a*l), work.exact(a*h)
        lo, hi = work.exact(lo+min(x, y)), work.exact(hi+max(x, y))
    return lo, hi


def inspect(raw, expected_source, work):
    """Parse all source rows. Immutable bytes avoid live alias consumption."""
    work.tick()
    require(type(raw) is bytes and len(raw) <= MEMBER_BYTES, "source bytes capacity")
    work.visit("coefficient_visits", (len(raw)+4095)//4096)
    require(digest(raw) == sha(expected_source), "source identity")
    obj = decode(raw)
    require(compact(obj) == raw, "canonical source")
    require(type(obj) is dict and set(obj) == {
        "schema", "request", "frame", "pair", "legal_pair", "factors",
        "le", "eq", "expert_a", "expert_b", "router_a", "router_b",
        "q", "constant", "semantics",
    } and obj["schema"] == SCHEMA, "source schema")
    sha(obj["request"])
    require(type(obj["frame"]) is str and 0 < len(obj["frame"]) <= 64, "frame identity")
    pair = obj["pair"]
    require(type(pair) is list and len(pair) == 2
            and all(type(i) is int and 0 <= i < 8 for i in pair)
            and pair[0] != pair[1] and obj["legal_pair"] == sorted(pair), "pair orientation")
    require(obj["semantics"] == "GIVEN_CONTINUOUS_SELECTED_SOFTMAX_TOP2", "source semantics")
    factors = obj["factors"]
    require(type(factors) is list and 1 <= len(factors) <= MAX_FACTORS, "factor capacity")
    ids, bounds = set(), []
    for factor in factors:
        work.visit("row_visits")
        require(type(factor) is list and len(factor) == 4, "factor fields")
        name, role, lower, upper = factor
        require(type(name) is str and 0 < len(name) <= 64 and name not in ids
                and role in ("shared", "expert_a", "expert_b"), "factor identity/role")
        ids.add(name)
        l, h = number(lower, work), number(upper, work)
        require(l <= h, "factor bounds")
        bounds.append((l, h))
    n = len(bounds)

    def row(value):
        work.visit("row_visits")
        require(type(value) is list and len(value) == n+1, "affine row dimension")
        return tuple(number(v, work) for v in value)

    constraints = {}
    require(type(obj["le"]) is list and type(obj["eq"]) is list
            and len(obj["le"])+len(obj["eq"]) <= MAX_ROWS, "constraint capacity")
    # Stored constraints are [rhs, coefficients...], unlike affine maps.
    for key in ("le", "eq"):
        constraints[key] = tuple(row(v) for v in obj[key])
    require(type(obj["q"]) is list and 1 <= len(obj["q"]) <= MAX_CLASSES, "property capacity")
    q = tuple(number(v, work) for v in obj["q"])
    k = len(q)
    for key in ("expert_a", "expert_b"):
        require(type(obj[key]) is list and len(obj[key]) == k, "expert/class dimension")
    a, b = tuple(map(row, obj["expert_a"])), tuple(map(row, obj["expert_b"]))
    ra, rb = row(obj["router_a"]), row(obj["router_b"])
    for j, (_, role, _, _) in enumerate(factors, 1):
        work.visit("coefficient_visits")
        require(role == "shared" or ra[j] == rb[j] == 0, "router private factor")
        require(role != "expert_b" or all(v[j] == 0 for v in a), "expert a foreign private factor")
        require(role != "expert_a" or all(v[j] == 0 for v in b), "expert b foreign private factor")
    c = number(obj["constant"], work)
    # Reconstruct directly from the original shared outputs, in exact arithmetic.
    base, difference, margin = [], [], []
    for j in range(n+1):
        bv = dot(q, [v[j] for v in b], work)
        av = dot(q, [v[j] for v in a], work)
        base.append(work.exact(bv+(c if j == 0 else F(0))))
        difference.append(work.exact(av-bv))
        margin.append(work.exact(ra[j]-rb[j]))
    return dict(obj=obj, bounds=tuple(bounds), le=constraints["le"], eq=constraints["eq"],
                b=tuple(base), d=tuple(difference), m=tuple(margin), source=expected_source)


def sigmoid_range(x, y, work):
    """Rational enclosure from monotonicity, sigma(0)=1/2 and sigma'<=1/4."""
    require(x <= y, "margin cell order")
    lower = F(1, 2) if x >= 0 else max(F(0), work.exact(F(1, 2)+x/4))
    upper = F(1, 2) if y <= 0 else min(F(1), work.exact(F(1, 2)+y/4))
    return lower, upper


def median_tilt(source, t, work):
    values = {}
    for i, (l, h) in enumerate(source["bounds"], 1):
        work.visit("coefficient_visits")
        m = source["m"][i]
        if l == h or not m:
            continue
        a = work.exact(source["b"][i]+work.exact(t*source["d"][i]))
        weight = work.exact(work.exact(work.exact(h-l)*abs(m))/2)
        ratio = work.exact(-a/m)
        values[ratio] = work.exact(values.get(ratio, F(0))+weight)
    if not values:
        return F(0)
    # Small explicit capacity; comparisons are charged as conservative n^2 visits.
    work.visit("coefficient_visits", len(values)**2)
    ordered = sorted(values)
    total = F(0)
    for value in values.values():
        total = work.exact(total+value)
    cumulative = F(0)
    for index, r in enumerate(ordered):
        cumulative = work.exact(cumulative+values[r])
        twice = work.exact(cumulative*2)
        if twice >= total:
            right = ordered[index+1] if twice == total else r
            return min(max(F(0), r), right)
    raise AssertionError("positive median weight")


def plan(source, mode, cells, work):
    require(mode in ("zero", "median"), "explicit finite policy")
    require(type(cells) is int and 1 <= cells <= MAX_CELLS, "cell capacity")
    lo, hi = affine_box(source["m"], source["bounds"], work)
    l, h = sigmoid_range(lo, hi, work)
    anchors = (l,) if l == h else (l, h)
    goals = []
    for t in anchors:
        tilts = [F(0)]
        if mode == "median" and l != h:
            v = median_tilt(source, t, work)
            if v:
                tilts.append(v)
        goals.extend((t, nu) for nu in tilts)
    count = 1 if lo == hi else cells
    width = work.exact(hi-lo)
    cuts = [work.exact(lo+work.exact(width*F(i, count))) for i in range(count+1)]
    return anchors, tuple(goals), tuple(zip(cuts, cuts[1:]))


def checked_support(source, t, nu, item, work):
    require(type(item) is dict and set(item) == {"t", "nu", "y", "z", "lower"}, "support fields")
    require(number(item["t"], work) == t and number(item["nu"], work) == nu, "objective binding")
    require(type(item["y"]) is list and len(item["y"]) == len(source["le"])
            and type(item["z"]) is list and len(item["z"]) == len(source["eq"]), "dual dimensions")
    y = tuple(number(v, work) for v in item["y"])
    z = tuple(number(v, work) for v in item["z"])
    require(all(v <= 0 for v in y), "inequality dual sign")
    coeff = tuple(work.exact(b+work.exact(t*d)+work.exact(nu*m))
                  for b, d, m in zip(source["b"], source["d"], source["m"]))
    lower = coeff[0]
    for dual, row in list(zip(y, source["le"]))+list(zip(z, source["eq"])):
        lower = work.exact(lower+work.exact(dual*row[0]))
    for j, (l, h) in enumerate(source["bounds"], 1):
        residual = coeff[j]
        for dual, row in list(zip(y, source["le"]))+list(zip(z, source["eq"])):
            residual = work.exact(residual-work.exact(dual*row[j]))
        lower = work.exact(lower+min(work.exact(residual*l), work.exact(residual*h)))
    require(number(item["lower"], work) == lower, "support residual lower mismatch")
    return lower


def check(raw_source, record, *, expected_source, expected_mode, expected_cells, work):
    """Full independent reconstruction; no producer-selected cells may disappear."""
    source = inspect(raw_source, expected_source, work)
    require(type(record) is dict and set(record) == {
        "schema", "source", "mode", "cell_count", "supports", "cells", "lower", "grade",
    } and record["schema"] == CERT and record["source"] == expected_source
            and record["grade"] == GRADE, "certificate identity/grade")
    require(record["mode"] == expected_mode and record["cell_count"] == expected_cells,
            "frozen policy binding")
    anchors, goals, cells = plan(source, record["mode"], record["cell_count"], work)
    require(type(record["supports"]) is list and len(record["supports"]) == len(goals),
            "complete support obligations")
    supports = [(t, nu, checked_support(source, t, nu, item, work))
                for (t, nu), item in zip(goals, record["supports"])]
    require(type(record["cells"]) is list and len(record["cells"]) == len(cells), "complete cells")
    bounds = []
    for (x, y), item in zip(cells, record["cells"]):
        work.visit("row_visits")
        require(type(item) is dict and set(item) == {"x", "y", "gate", "lower"}, "cell fields")
        require(number(item["x"], work) == x and number(item["y"], work) == y, "cell coverage")
        gate = sigmoid_range(x, y, work)
        require(type(item["gate"]) is list and len(item["gate"]) == 2
                and tuple(number(v, work) for v in item["gate"]) == gate, "sigmoid premise")
        require(anchors[0] <= gate[0] <= gate[1] <= anchors[-1], "gate anchor coverage")
        lines = []
        for t in anchors:
            lines.append(max(work.exact(lb+min(work.exact(-nu*x), work.exact(-nu*y)))
                             for a, nu, lb in supports if a == t))
        if len(anchors) == 1:
            lower = lines[0]
        else:
            l, h = anchors
            width = work.exact(h-l)
            lower = min(work.exact(work.exact(
                work.exact(work.exact(h-g)*lines[0])+work.exact(work.exact(g-l)*lines[1])) / width)
                for g in gate)
            lower = work.exact(lower)
        require(number(item["lower"], work) == lower, "cell bound mismatch")
        bounds.append(lower)
    lower = min(bounds)
    require(number(record["lower"], work) == lower, "aggregate lower mismatch")
    work.tick()
    return dict(lower=str(lower), grade=GRADE, supports=len(goals), cells=len(cells),
                status="POSITIVE_GIVEN_SOURCE" if lower > 0 else "NONPOSITIVE_BOUND")
