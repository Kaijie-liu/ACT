"""Exact property views of supplied shared-input HZs, standard library only.

Checks source composition and original coefficients, NOT network lowering,
route reachability, or complete-request safety. The caller anchors source and
property identities independently of the untrusted candidate. No optimizer.
"""
from fractions import Fraction as F

from scoped_source.hz_request_store import POLICY, decode, digest, rational, require, sha
from scoped_source.check_gate_margin_rows import (
    MARGIN_PROFILE, _row_equal, _same_fields, freeze, number, source,
    validate_binding,
)

SCHEMA = "MARGIN_NATIVE_JOINT_SOURCE_V1"
VIEW = "MARGIN_NATIVE_PROPERTY_VIEW_V1"
GRADE = "CHECKED_SUPPORT_ON_GIVEN_HZ_CONTINUOUS_RELAXATION"


def parameter(value, work):
    """Accept the existing exact planner's Fractions without widening bit limits."""
    if type(value) is F:
        work.visit("coefficient_visits")
        return work.exact(rational(str(value)))
    return number(value, work)


def owned(raw, anchor, work):
    work.tick()
    require(type(raw) is bytes and len(raw) <= POLICY["total_bytes"], "owned source bytes")
    work.visit("coefficient_visits", (len(raw)+4095)//4096)
    require(digest(raw) == sha(anchor), "external owned-source identity")
    value = decode(raw)
    require(freeze(value, work, profile=MARGIN_PROFILE)[0] == raw, "canonical owned source")
    return value


def inspect(raw, *, expected_source, work):
    """Reconstruct shared/private composition and the two retained router rows."""
    obj = owned(raw, expected_source, work)
    require(type(obj) is dict and set(obj) == {
        "schema", "profile", "scope", "base", "augmented", "router_rows", "relaxation",
    } and obj["schema"] == SCHEMA and obj["profile"] == MARGIN_PROFILE, "margin source schema")
    scope = obj["scope"]
    require(type(scope) is dict and set(scope) == {"request", "domain", "guard", "model", "input"},
            "complete caller-owned source scope")
    for value in scope.values():
        sha(value)
    base = obj["base"]
    base_hash = freeze(base, work, profile=MARGIN_PROFILE)[1]
    parsed = validate_binding(base, expected_hash=base_hash, work=work, profile=MARGIN_PROFILE)
    require(all(base["context"][k] == scope[k] for k in ("request", "domain")), "base scope binding")
    joint, router = parsed["joint"], parsed["router"]
    aug = source(obj["augmented"], work, profile=MARGIN_PROFILE)
    _same_fields(joint, aug, ("nc", "nb", "frame", "b", "ub", "Ac", "Ab", "Auc", "Aub"), work)
    pair = base["context"]["pair"]
    start = len(joint["c"])
    require(type(obj["router_rows"]) is list and obj["router_rows"] == [start, start+1]
            and all(type(i) is int for i in obj["router_rows"]), "router output row binding")
    require(aug["c"] == joint["c"] + tuple(router["c"][i] for i in pair)
            and aug["exact"] == (joint["exact"] and router["exact"]), "augmented centers/exact flag")
    for key in ("Gc", "Gb"):
        require(len(aug[key]) == start+2, "all expert and router outputs retained")
        for before, after in zip(joint[key], aug[key][:start]):
            _row_equal(before, after, work)
        for output, router_index in zip(obj["router_rows"], pair):
            # Separate continuous/binary blocks; no accidental router.n_cont shift.
            _row_equal(aug[key][output], router[key][router_index], work)
    require(obj["relaxation"] == {
        "kind": "BINARY_MINUS_PLUS_ONE_TO_CONTINUOUS_BOX", "count": joint["nb"],
    } and type(obj["relaxation"]["count"]) is int, "explicit binary relaxation")
    work.tick()
    return dict(obj=obj, joint=aug, n_classes=start//2, source=expected_source)


def _accumulate(target, j, value, work):
    result = work.exact(target.get(j, F(0)) + value)
    if result:
        target[j] = result
    else:
        target.pop(j, None)


def project(hz, weights, offset, work):
    """Exact projection from original binary64 rows; never a float dot product."""
    require(len(weights) == len(hz["c"]), "complete output property dimension")
    center, coefficients = work.exact(offset), {}
    for i, weight in enumerate(weights):
        work.visit("row_visits")
        center = work.exact(center + work.exact(weight*hz["c"][i]))
        for name, shift in (("Gc", 0), ("Gb", hz["nc"])):
            for j, value in hz[name][i].items():
                work.visit("coefficient_visits")
                _accumulate(coefficients, shift+j, work.exact(weight*value), work)
    return center, coefficients


def encoded(row, work):
    center, coeff = row
    work.visit("coefficient_visits", len(coeff))
    # Intermediates may use work_bits, but serialized values retain stored_bits.
    for value in (center, *coeff.values()):
        rational(str(value))
    return dict(constant=str(center), entries=[[j, str(v)] for j, v in sorted(coeff.items())])


def projected_property(src, q, constant, work):
    k, hz = src["n_classes"], src["joint"]
    require(type(q) is list and len(q) == k, "lossless complete class property")
    weights = tuple(parameter(v, work) for v in q)
    offset = parameter(constant, work)
    zeros = (F(0),)*k
    b = project(hz, zeros+weights+(F(0), F(0)), offset, work)
    d = project(hz, weights+tuple(-v for v in weights)+(F(0), F(0)), F(0), work)
    m = project(hz, zeros+zeros+(F(1), F(-1)), F(0), work)
    return weights, offset, dict(b=b, d=d, m=m)


def make_property(raw, *, expected_source, q, constant, work):
    """Produce a view; the independent receiver rebuilds it from the full source."""
    src = inspect(raw, expected_source=expected_source, work=work)
    weights, offset, rows = projected_property(src, q, constant, work)
    value = dict(schema=VIEW, source=expected_source, q=list(map(str, weights)),
                 constant=str(offset), rows={k: encoded(v, work) for k, v in rows.items()})
    return freeze(value, work, profile=MARGIN_PROFILE)


def inspect_property(raw, view, *, expected_source, expected_property, work):
    src = inspect(raw, expected_source=expected_source, work=work)
    obj = owned(view, expected_property, work)
    require(type(obj) is dict and set(obj) == {"schema", "source", "q", "constant", "rows"}
            and obj["schema"] == VIEW and obj["source"] == expected_source, "property source binding")
    q, constant, rows = projected_property(src, obj["q"], obj["constant"], work)
    require(obj["q"] == list(map(str, q)) and obj["constant"] == str(constant)
            and obj["rows"] == {k: encoded(v, work) for k, v in rows.items()},
            "original-coefficient property reconstruction")
    return dict(**src, q=q, constant=constant, rows=rows, property=expected_property)


def objective(view, t, nu, work):
    """Assemble a sparse exact objective in the unchanged factor frame."""
    t, nu = parameter(t, work), parameter(nu, work)
    require(0 <= t <= 1, "normalized gate anchor")
    total, coeff = F(0), {}
    for key, weight in (("b", F(1)), ("d", t), ("m", nu)):
        center, entries = view["rows"][key]
        total = work.exact(total + work.exact(weight*center))
        for j, value in entries.items():
            work.visit("coefficient_visits")
            _accumulate(coeff, j, work.exact(weight*value), work)
    return total, coeff


def check_supports(raw, view, candidates, *, expected_source, expected_property,
                   expected_goals, work):
    """All explicitly frozen objectives checked; no producer-selected omissions.

    This is a support receipt, not the scalar margin aggregation or a MoE SAFE.
    Candidate multipliers may come from the existing floating batch generator.
    """
    src = inspect_property(raw, view, expected_source=expected_source,
                           expected_property=expected_property, work=work)
    require(type(expected_goals) is tuple and 1 <= len(expected_goals) <= 8,
            "finite caller-owned objective roster")
    require(type(candidates) is list and len(candidates) == len(expected_goals), "complete candidates")
    hz = src["joint"]
    results = []
    for goal, item in zip(expected_goals, candidates):
        require(type(goal) is tuple and len(goal) == 2, "objective roster entry")
        t, nu = (parameter(v, work) for v in goal)
        require(type(item) is dict and set(item) == {"t", "nu", "y", "z", "lower"}
                and number(item["t"], work) == t and number(item["nu"], work) == nu,
                "support objective binding")
        lower, residual = objective(src, str(t), str(nu), work)
        for key, ck, bk, rhs in (("y", "Auc", "Aub", "ub"), ("z", "Ac", "Ab", "b")):
            require(type(item[key]) is list and len(item[key]) == len(hz[rhs]), "complete dual rows")
            for i, value in enumerate(item[key]):
                work.visit("row_visits")
                dual = number(value, work)
                require(key == "z" or dual <= 0, "inequality dual sign")
                lower = work.exact(lower + work.exact(dual*hz[rhs][i]))
                for name, shift in ((ck, 0), (bk, hz["nc"])):
                    for j, a in hz[name][i].items():
                        work.visit("coefficient_visits")
                        _accumulate(residual, shift+j, work.exact(-dual*a), work)
        # All HZ factors use [-1,1]; exact nonzero residuals suffice. Omitted
        # coefficients are verified zero, not unchecked or dropped source rows.
        for value in residual.values():
            lower = work.exact(lower-abs(value))
        require(number(item["lower"], work) == lower, "exact residual lower mismatch")
        results.append(dict(t=str(t), nu=str(nu), lower=str(lower)))
    work.tick()
    return dict(grade=GRADE, source=expected_source, property=expected_property,
                results=results, relaxed_binaries=hz["nb"],
                scalar_aggregate_checked=False, complete_request=False)
