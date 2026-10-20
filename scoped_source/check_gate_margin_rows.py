"""Independent, bounded checking of two gate/margin rows on supplied HZs.

Standard library only. All work continues a caller-owned hz_request_store.Work.
This checks given source composition and added rows, NOT network lowering, gate
range provenance, original F0 lowering, solver accuracy or route coverage.
"""
from fractions import Fraction as F
import math

from scoped_source.hz_request_store import (
    POLICY, compact, decode, digest, rational, require, sha,
)

SCHEMA = "GATE_MARGIN_ROWS_V1"
BINDING = "GATE_MARGIN_SOURCES_V1"
GATE_SEMANTICS = "SELECTED_SOFTMAX_TOP2_UNIT_TEMPERATURE"
SCOPE = "CHECKED_ADDED_ROWS_GIVEN_SHARED_ROUTER_AND_F0"
VECTORS = ("c", "b", "ub")
MATRICES = ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
# The existing finite pair-map boundary, not eight-expert capacity admission.
FACTORS, ROWS, OUTPUTS = 128, 256, 128


def freeze(value, work):
    """Bound the JSON tree and a conservative encoded-size sum before copying."""
    stack, size = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        work.visit("coefficient_visits")
        require(depth <= 12, "record depth")
        if type(item) is dict:
            require(len(item) <= 32 and all(type(k) is str and len(k) <= 64 for k in item), "record keys")
            size += 2 + 4 * len(item) + sum(12 * len(k) + 2 for k in item)
            stack.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            require(len(item) <= (FACTORS + 2) * (ROWS + 6), "record list capacity")
            size += 2 + len(item)
            stack.extend((v, depth + 1) for v in item)
        elif type(item) is str:
            require(len(item) <= 2470, "record string capacity")
            size += 12 * len(item) + 2
        elif type(item) is int:
            require(item.bit_length() <= POLICY["stored_bits"], "record integer capacity")
            size += 2 + item.bit_length()
        elif type(item) is float:
            require(math.isfinite(item), "record finite float")
            size += 32
        elif item is None or type(item) is bool:
            size += 5
        else:
            raise ValueError("record JSON types")
        require(size <= POLICY["total_bytes"], "record byte capacity")
    work.tick()
    raw = compact(value)
    require(len(raw) <= size <= POLICY["total_bytes"], "record encoded-size contract")
    work.tick()
    return raw, digest(raw)


def number(value, work):
    work.visit("coefficient_visits")
    if type(value) is str:
        value = rational(value)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite binary64 source")
        value = F(value)
    elif type(value) is int:
        require(value.bit_length() <= POLICY["stored_bits"], "stored integer capacity")
        value = F(value)
    else:
        raise ValueError("finite stored rational required")
    return work.exact(value)


def _vector(xs, limit, work):
    require(type(xs) is list and len(xs) <= limit, "finite vector capacity")
    return tuple(number(x, work) for x in xs)


def _binary64(value, work):
    try:
        stored = float(value)
    except OverflowError:
        raise ValueError("nonfinite binary64 representation") from None
    require(math.isfinite(stored) and work.exact(F(stored)) == value, "non-binary64 stored source")


def source(raw, work, *, f0=False):
    """No floating normalization or duplicate-column summation."""
    work.tick()
    require(type(raw) is dict and set(raw) == set(VECTORS + MATRICES + ("frame_id", "exact")),
            "complete source fields")
    require(type(raw["frame_id"]) is int and type(raw["exact"]) is bool, "source frame")
    nc, nb = raw["Gc"]["shape"][1], raw["Gb"]["shape"][1]
    require(type(nc) is int and type(nb) is int and min(nc, nb) >= 0
            and nc + nb <= FACTORS + (2 if f0 else 0), "finite factor capacity")
    h = {k: _vector(raw[k], OUTPUTS if k == "c" else ROWS + (6 if f0 else 0), work)
         for k in VECTORS}
    require(0 < len(h["c"]) <= OUTPUTS, "source outputs")
    require(len(h["b"]) + len(h["ub"]) <= ROWS + (6 if f0 else 0), "total constraint capacity")
    for values in h.values():
        for value in values:
            _binary64(value, work)
    h.update(nc=nc, nb=nb, frame=raw["frame_id"], exact=raw["exact"])
    for name in MATRICES:
        nr = len(h["c" if name.startswith("G") else "b" if name in ("Ac", "Ab") else "ub"])
        width = nc if name.endswith("c") else nb
        matrix = raw[name]
        require(type(matrix) is dict and set(matrix) == {"shape", "data", "indices", "indptr"},
                "canonical CSR fields")
        shape, data, ids, ptr = (matrix[k] for k in ("shape", "data", "indices", "indptr"))
        require(type(shape) is list and shape == [nr, width]
                and all(type(x) is int for x in shape), "CSR shape")
        require(type(data) is list and type(ids) is list and type(ptr) is list
                and len(data) == len(ids) <= nr * width and len(ptr) == nr + 1
                and all(type(x) is int for x in ptr) and ptr[0] == 0 and ptr[-1] == len(data),
                "CSR capacity/pointers")
        result = []
        for i in range(nr):
            work.visit("row_visits")
            lo, hi = ptr[i:i + 2]
            require(0 <= lo <= hi <= len(data), "CSR pointer order")
            row, previous = {}, -1
            for p in range(lo, hi):
                j = ids[p]
                require(type(j) is int and previous < j < width, "CSR column order/alias")
                previous = j
                value = number(data[p], work)
                _binary64(value, work)
                if value:
                    row[j] = value
            result.append(row)
        h[name] = tuple(result)
    return h


def _row_equal(a, b, work):
    work.visit("row_visits")
    work.visit("coefficient_visits", len(a) + len(b))
    require(a == b, "retained coefficient row mismatch")


def _mapped(row, mapping, work):
    work.visit("coefficient_visits", len(row))
    return {mapping[j]: v for j, v in row.items()}


def _same_fields(a, b, names, work):
    for name in names:
        work.tick()
        av, bv = a[name], b[name]
        if name in MATRICES:
            require(len(av) == len(bv), "matrix height changed")
            for x, y in zip(av, bv):
                _row_equal(x, y, work)
        else:
            if name in VECTORS:
                work.visit("coefficient_visits", len(av) + len(bv))
            require(av == bv, "source field changed: " + name)


def _prefix(a, b, work):
    for left, right, rhs in (("Ac", "Ab", "b"), ("Auc", "Aub", "ub")):
        n = len(a[rhs])
        require(len(b[rhs]) >= n and b[rhs][:n] == a[rhs], "source RHS prefix changed")
        work.visit("coefficient_visits", 2 * n)
        for i in range(n):
            _row_equal(a[left][i], b[left][i], work)
            _row_equal(a[right][i], b[right][i], work)


def _metadata(value, entry, a, b, work):
    expected = {
        "a_rows": list(range(len(a["c"]))),
        "b_rows": list(range(len(a["c"]), 2 * len(a["c"]))),
        "shared_continuous": entry["nc"], "shared_binary": entry["nb"],
        "a_private_continuous": a["nc"] - entry["nc"],
        "b_private_continuous": b["nc"] - entry["nc"],
        "a_private_binary": a["nb"] - entry["nb"],
        "b_private_binary": b["nb"] - entry["nb"], "relation_mode": "shared_input",
    }
    work.tick()
    require(value == expected and all(type(x) is int for k in ("a_rows", "b_rows") for x in value[k]),
            "shared/private metadata or expert output order")
    for key in expected.keys() - {"a_rows", "b_rows", "relation_mode"}:
        require(type(value[key]) is int, "factor count type")


def context(value):
    require(type(value) is dict and set(value) == {"request", "domain", "pair"}, "context fields")
    sha(value["request"]); sha(value["domain"])
    pair = value["pair"]
    require(type(pair) is list and len(pair) == 2
            and all(type(x) is int and x >= 0 for x in pair) and pair[0] < pair[1],
            "canonical pair direction")


def validate_binding(binding, *, expected_hash, work):
    """Independently reconstruct given entry/expert composition and router map."""
    require(type(binding) is dict and set(binding) == {
        "schema", "gate_semantics", "context", "sources", "metadata", "router_eq_rows", "router_le_rows"},
        "binding fields")
    context(binding["context"])
    require(binding["schema"] == BINDING, "binding schema")
    require(binding["gate_semantics"] == GATE_SEMANTICS, "declared gate semantics")
    require(type(binding["sources"]) is dict and set(binding["sources"]) ==
            {"entry", "a", "b", "joint", "input", "router"}, "source roster")
    parsed = {k: source(v, work) for k, v in binding["sources"].items()}
    work.tick()
    require(freeze(binding, work)[1] == sha(expected_hash), "external source binding")
    entry, a, b, joint, inp, router = (parsed[k] for k in ("entry", "a", "b", "joint", "input", "router"))
    require(len(a["c"]) == len(b["c"]), "expert output widths")
    for expert in (a, b):
        require(expert["frame"] == entry["frame"] and expert["nc"] >= entry["nc"]
                and expert["nb"] >= entry["nb"], "expert/common factor frame")
        _prefix(entry, expert, work)
    _metadata(binding["metadata"], entry, a, b, work)
    nc, nb = a["nc"] + b["nc"] - entry["nc"], a["nb"] + b["nb"] - entry["nb"]
    require((joint["nc"], joint["nb"]) == (nc, nb) and joint["frame"] == entry["frame"],
            "joint dimensions/frame")
    require(joint["c"] == a["c"] + b["c"]
            and joint["exact"] == all(x["exact"] for x in (entry, a, b)), "joint outputs/exact")
    cm = {"entry": list(range(entry["nc"])), "a": list(range(a["nc"])),
          "b": list(range(entry["nc"])) + list(range(a["nc"], nc))}
    bm = {"entry": list(range(entry["nb"])), "a": list(range(a["nb"])),
          "b": list(range(entry["nb"])) + list(range(a["nb"], nb))}
    for key, maps in (("Gc", cm), ("Gb", bm)):
        expected = tuple(_mapped(row, maps[name], work) for name in ("a", "b") for row in parsed[name][key])
        require(len(expected) == len(joint[key]), "joint output height")
        for x, y in zip(expected, joint[key]):
            _row_equal(x, y, work)
    for ck, bk, rhs in (("Ac", "Ab", "b"), ("Auc", "Aub", "ub")):
        pos = 0
        for name in ("entry", "a", "b"):
            h = parsed[name]; start = 0 if name == "entry" else len(entry[rhs])
            for i in range(start, len(h[rhs])):
                require(pos < len(joint[rhs]) and joint[rhs][pos] == h[rhs][i], "joint RHS/coverage")
                _row_equal(joint[ck][pos], _mapped(h[ck][i], cm[name], work), work)
                _row_equal(joint[bk][pos], _mapped(h[bk][i], bm[name], work), work)
                pos += 1
        require(pos == len(joint[rhs]), "unexpected joint rows")
    require((inp["nc"], inp["nb"]) == (nc, nb), "padded input dimensions")
    _same_fields(entry, inp, VECTORS + MATRICES + ("frame", "exact"), work)
    require(router["frame"] == entry["frame"] and router["nc"] <= entry["nc"]
            and router["nb"] <= entry["nb"] and binding["context"]["pair"][1] < len(router["c"]),
            "router prefix/pair")
    for ck, bk, rhs, label in (("Ac", "Ab", "b", "router_eq_rows"), ("Auc", "Aub", "ub", "router_le_rows")):
        mapping = binding[label]
        require(type(mapping) is list and len(mapping) == len(router[rhs])
                and all(type(i) is int and 0 <= i < len(entry[rhs]) for i in mapping)
                and len(set(mapping)) == len(mapping), "router retained row map")
        for i, j in enumerate(mapping):
            require(router[rhs][i] == entry[rhs][j], "router retained RHS")
            _row_equal(router[ck][i], entry[ck][j], work)
            _row_equal(router[bk][i], entry[bk][j], work)
    work.tick()
    return parsed


def active_sources(binding, active, *, expected_hash, work):
    original = validate_binding(binding, expected_hash=expected_hash, work=work)
    require(type(active) is dict and set(active) == {"router", "joint", "input"}, "active source roster")
    current = {k: source(v, work) for k, v in active.items()}
    additions = []
    for name in ("router", "joint", "input"):
        old, new = original[name], current[name]
        _same_fields(old, new, ("nc", "nb", "frame", "c", "Gc", "Gb", "Ac", "Ab", "b"), work)
        _prefix(old, new, work)
        count = len(new["ub"]) - len(old["ub"])
        require(count in (0, 2), "only unchanged or two retained rows supported")
        additions.append(count)
        require(new["exact"] in (old["exact"], False), "active exact flag upgraded")
    require(len(set(additions)) == 1, "slice rows omitted from a domain")
    if additions[0]:
        for j in (-2, -1):
            for other in ("joint", "input"):
                require(current["router"]["ub"][j] == current[other]["ub"][j], "slice RHS mismatch")
                for key in ("Auc", "Aub"):
                    _row_equal(current["router"][key][j], current[other][key][j], work)
    router = original["router"]; pair = binding["context"]["pair"]
    m0 = work.exact(router["c"][pair[0]] - router["c"][pair[1]])
    v = {}
    ncont = current["joint"]["nc"]
    for key, shift in (("Gc", 0), ("Gb", ncont)):
        for sign, i in ((1, pair[0]), (-1, pair[1])):
            for j, val in router[key][i].items():
                v[j + shift] = work.exact(v.get(j + shift, F(0)) + sign * val)
    return current, m0, {j: val for j, val in v.items() if val}


def sign_premise(hz, m0, v, work):
    """Search only box sign or one exactly proportional retained inequality."""
    radius = F(0)
    for value in v.values():
        radius = work.exact(radius + abs(value))
    if work.exact(m0 - radius) >= 0:
        return {"side": 1, "kind": "BOX"}
    if work.exact(m0 + radius) <= 0:
        return {"side": -1, "kind": "BOX"}
    if not v:
        return None
    first = min(v)
    for i, (cr, br, rhs) in enumerate(zip(hz["Auc"], hz["Aub"], hz["ub"])):
        work.visit("row_visits")
        work.visit("coefficient_visits", len(cr) + len(br))
        row = dict(cr)
        row.update({hz["nc"] + j: x for j, x in br.items()})
        if set(row) != set(v):
            continue
        scale = work.exact(row[first] / v[first])
        if not scale:
            continue
        if all(row[j] == work.exact(scale * v[j]) for j in v):
            # scale*m <= rhs+scale*m0; select a half-line only if this
            # actually entails its sign. No tolerance or numeric range test.
            limit = work.exact(rhs + work.exact(scale * m0))
            if limit <= 0:
                return {"side": -1 if scale > 0 else 1, "kind": "ROW", "row": i, "scale": str(scale)}
    return None


def _f0_base(active, raw, work):
    f0 = source(raw, work, f0=True); base = active["joint"]
    require((f0["nc"], f0["nb"]) == (base["nc"] + 2, base["nb"])
            and f0["frame"] == base["frame"] and len(f0["c"]) == 1 and not f0["exact"],
            "given F0 dimensions")
    require(len(f0["ub"]) == len(base["ub"]) + 4 and len(f0["b"]) == len(base["b"]),
            "given F0 McCormick row count")
    _prefix(base, f0, work)
    return f0


def inspect(binding, active, old_f0, parameters, *, expected_binding, work):
    """Shared parsing/structural premise checks, not the new-row construction."""
    current, m0, v = active_sources(binding, active, expected_hash=expected_binding, work=work)
    f0 = _f0_base(current, old_f0, work)
    require(type(parameters) is dict and set(parameters) == {
        "lambda_slot", "product_slot", "center", "radius", "gate", "q", "offset"}, "parameter fields")
    n = current["joint"]["nc"]
    require(type(parameters["lambda_slot"]) is int and parameters["lambda_slot"] == n
            and type(parameters["product_slot"]) is int and parameters["product_slot"] == n + 1,
            "actual F0 factor slots")
    classes = len(binding["metadata"]["a_rows"])
    q = _vector(parameters["q"], classes, work)
    require(len(q) == classes, "property width")
    number(parameters["offset"], work)
    C, R = number(parameters["center"], work), number(parameters["radius"], work)
    bounds = _vector(parameters["gate"], 2, work)
    require(len(bounds) == 2 and 0 <= bounds[0] <= bounds[1] <= 1, "gate interval")
    for value in (*bounds, C, R):
        _binary64(value, work)
    # Bind the actual, versioned native recipe; this does NOT make its
    # interval embedding outward or reprove the old McCormick lowering.
    rounded_sum = work.exact(F(float(work.exact(bounds[0] + bounds[1]))))
    rounded_diff = work.exact(F(float(work.exact(bounds[1] - bounds[0]))))
    require(C == work.exact(F(float(work.exact(rounded_sum / 2))))
            and R == work.exact(F(float(work.exact(rounded_diff / 2)))),
            "lambda representation differs from native recipe")
    embedding = R >= 0 and work.exact(C - R) <= bounds[0] and work.exact(C + R) >= bounds[1]
    premise = sign_premise(current["joint"], m0, v, work) if embedding else None
    return current, f0, m0, v, C, R, premise, embedding


def check(record, *, expected_binding, expected_context, expected_property, work):
    """Reconstruct ideal rows and check final F0 differs only by these rows."""
    work.tick()
    require(type(expected_property) in (tuple, list) and len(expected_property) == 2,
            "external property fields")
    expected = decode(freeze({"context": expected_context,
                              "property": list(expected_property)}, work)[0])
    context(expected["context"])
    require(type(record) is dict, "relation record type")
    original = record
    raw, before_hash = freeze(original, work)
    record = decode(raw)
    require(set(record) == {"schema", "binding", "active", "old_f0", "parameters",
                           "premise", "status", "new_f0"}, "complete relation record")
    require(record["schema"] == SCHEMA, "relation schema")
    require(record["binding"]["context"] == expected["context"], "request/domain/pair binding")
    require([record["parameters"]["q"], record["parameters"]["offset"]] == expected["property"],
            "property identity")
    work.tick()
    active, old, m0, v, C, R, premise, embedding = inspect(
        record["binding"], record["active"], record["old_f0"], record["parameters"],
        expected_binding=expected_binding, work=work)
    require(record["premise"] == premise, "independent sign premise mismatch")
    expected_status = "APPLIED" if embedding and premise else "NO_SIGN_PREMISE" if embedding else "GATE_EMBEDDING_REFUSED"
    require(record["status"] == expected_status, "relation status mismatch")
    new = source(record["new_f0"], work, f0=True)
    _same_fields(old, new, ("nc", "nb", "frame", "exact", "c", "Gc", "Gb", "Ac", "Ab", "b"), work)
    _prefix(old, new, work)
    added = 2 if expected_status == "APPLIED" else 0
    require(len(new["ub"]) == len(old["ub"]) + added, "added relation row coverage")
    if added:
        side = premise["side"]; nc = old["nc"]
        # Independently assemble coefficients of s*(lambda-m/4) and
        # -s*lambda, including the actual stored centers.
        margin = {j if j < active["joint"]["nc"] else j + 2: x for j, x in v.items()}
        linear = {j: work.exact(-side * x / 4) for j, x in margin.items()}
        linear[record["parameters"]["lambda_slot"]] = work.exact(side * R)
        half = {record["parameters"]["lambda_slot"]: work.exact(-side * R)}
        targets = (
            (linear, work.exact(F(side, 2) - work.exact(side * C) + work.exact(side * m0 / 4))),
            (half, work.exact(work.exact(side * C) - F(side, 2))),
        )
        for i, (ideal, rhs) in enumerate(targets, len(old["ub"])):
            got = dict(new["Auc"][i])
            got.update({nc + j: x for j, x in new["Aub"][i].items()})
            require(set(got) <= set(ideal), "nonzero private/product or foreign relation coefficient")
            error = F(0)
            for j in set(got) | set(ideal):
                error = work.exact(error + abs(work.exact(got.get(j, F(0)) - ideal.get(j, F(0)))))
            require(new["ub"][i] >= work.exact(rhs + error), "inward relation row")
    work.tick()
    require(freeze(original, work)[1] == before_hash, "relation record changed during check")
    return {"status": expected_status, "scope": SCOPE, "source_complete": False,
            "record_sha256": before_hash, "hard_budget_supervision": False,
            "deployed_float_SAFE": False,
            "trusted_premises": ["network_and_guard_lowering", "active_slice_coverage",
                                 "canonical_expert_slot_identity", "original_gate_range_and_F0_lowering",
                                 "selected_softmax_unit_temperature_of_supplied_router"],
            "binding_sha256": expected_binding, "old_f0_sha256": freeze(record["old_f0"], work)[1],
            "new_f0_sha256": freeze(record["new_f0"], work)[1], "added_rows": added}
