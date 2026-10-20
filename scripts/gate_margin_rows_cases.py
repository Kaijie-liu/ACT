"""Literal M3 native constructions and exact saved-record controls.

Imported only by the frozen control adapter. No optimizer or model execution.
The receiver functions have standard-library dependencies only.
"""
from fractions import Fraction as F
import copy
import hashlib
import time

SPECS = ["P0", "P1", "N", "Z", "E", "Q", "X", "R"]
REFERENCES = ["B0", "B1", "B2", "B3"]
FAULTS = [
    "schema", "extra_field", "request", "domain", "pair", "property",
    "gate_semantics", "source_center", "router_map", "private_metadata",
    "missing_active_rows", "sign_premise", "inward_row", "lambda_slot",
    "product_slot", "binary_shift", "old_output", "missing_new_row",
    "missing_control_record", "deadline_binding", "ledger_reset",
    "expired_work", "stored_bits", "csr_index_dtype", "csr_duplicate",
    "binding_digest", "binding_oversize", "independent_mode",
]
ERRORS = [
    "relation schema", "complete relation record", "request/domain/pair binding",
    "request/domain/pair binding", "request/domain/pair binding", "property identity",
    "declared gate semantics", "source field changed: c", "router retained row map",
    "shared/private metadata or expert output order", "slice rows omitted from a domain",
    "independent sign premise mismatch", "inward relation row", "actual F0 factor slots",
    "actual F0 factor slots", "nonzero private/product or foreign relation coefficient",
    "source field changed: c", "added relation row coverage", "complete control roster",
    "same deadline", "ledger reset or omitted work", "store deadline",
    "stored integer capacity", "native CSR integral index buffers",
    "native CSR duplicate/value", "binding bytes/hash", "bounded binding header",
    "independent input gate relation refused",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(label):
    return hashlib.sha256(("m3-native-control/" + label).encode()).hexdigest()


def expected_context(name):
    return {"request": identity(name), "domain": identity("domain/" + name), "pair": [0, 1]}


def ledger(counts):
    require(type(counts) is dict and set(counts) == {
        "row_visits", "coefficient_visits", "arithmetic"}
        and all(type(v) is int and v >= 0 for v in counts.values())
        and sum(counts.values()) <= 8000000, "cumulative Work")


def inventory(records, deadline, reported_deadline, final_counts):
    require(reported_deadline == deadline, "same deadline")
    require([v["case"] for v in records] == SPECS + REFERENCES, "complete control roster")
    ledger(final_counts)
    previous = None
    for row in records:
        ledger(row["before"]); ledger(row["after"])
        require(previous is None or previous == row["before"], "contiguous control ledger")
        require(all(row["before"][k] <= row["after"][k] <= final_counts[k]
                    for k in final_counts), "ledger reset or omitted work")
        previous = row["after"]


def all_faults(records, faults, final_counts):
    require([v["name"] for v in faults] == FAULTS, "complete fault roster")
    previous = records[-1]["after"]
    for i, fault in enumerate(faults):
        ledger(fault["before"]); ledger(fault["after"])
        require(fault["before"] == previous and fault["status"] == "PASS"
                and fault["error"] == ERRORS[i], "fault prefix/identity/rejection")
        require(all(fault["after"][k] >= previous[k] for k in previous), "fault ledger decrease")
        previous = fault["after"]
    require(previous == final_counts, "fault terminal ledger")


def dot(row, z, work):
    total = F(0)
    for j, value in row.items():
        total = work.exact(total + work.exact(value * z[j]))
    return total


def flattened(h, name):
    c, b = ("Gc", "Gb") if name == "output" else ("Auc", "Aub") if name == "le" else ("Ac", "Ab")
    return [dict(list(cr.items()) + [(h["nc"] + j, v) for j, v in br.items()])
            for cr, br in zip(h[c], h[b])]


def feasible_value(h, point, work):
    z = tuple(F(x) for x in point)
    require(len(z) == h["nc"] + h["nb"] and all(abs(x) <= 1 for x in z)
            and all(x in (-1, 1) for x in z[h["nc"]:]), "exact point factors")
    require(all(dot(row, z, work) == b for row, b in zip(flattened(h, "eq"), h["b"])),
            "exact point equality")
    require(all(dot(row, z, work) <= b for row, b in zip(flattened(h, "le"), h["ub"])),
            "exact point inequality")
    return work.exact(h["c"][0] + dot(flattened(h, "output")[0], z, work))


def lower(h, weights, work):
    """Exact LP dual residual compensation on the stored F0, not a solve."""
    rows = flattened(h, "le")
    objective = flattened(h, "output")[0]
    residual = dict(objective)
    value = h["c"][0]
    for i, weight in weights:
        y = F(weight)
        require(type(i) is int and 0 <= i < len(rows) and y <= 0, "dual sign/index")
        value = work.exact(value + work.exact(y * h["ub"][i]))
        for j, coeff in rows[i].items():
            residual[j] = work.exact(residual.get(j, F(0)) - work.exact(y * coeff))
    for v in residual.values():
        value = work.exact(value - abs(v))
    return value


def literal_native_form(name, raw, work):
    """Registered objectives and selected constant-difference MC/slice rows."""
    from scoped_source.check_gate_margin_rows import source
    h = source(raw, work, f0=True)
    require(h["frame"] == 901 + (SPECS + REFERENCES).index(name)
            and h["exact"] is False, "registered native frame")
    if name == "Q":
        require(h["c"] == (F(5, 16),) and h["Gc"] == ({0: F(1, 8), 2: F(-1, 8), 4: F(7, 16)},)
                and h["Gb"] == ({0: F(1, 8), 2: F(-1, 8)},), "registered private objective")
        require(flattened(h, "eq") == [{0: F(1), 5: F(1, 2)}, {1: F(1), 6: F(-1)},
                                     {2: F(1), 7: F(-1)}] and h["b"] == (F(0),)*3,
                "registered private equalities")
        require(flattened(h, "le")[:5] == [{0: F(1)}, {1: F(1)}, {2: F(-1)},
                                          {0: F(1), 5: F(1, 4)}, {0: F(-1), 5: F(-1, 4)}]
                and h["ub"][:5] == (F(1), F(1, 2), F(3, 4), F(1), F(0)),
                "registered private retained rows")
        return h
    if name in ("P0", "P1", "N") or name in REFERENCES:
        upper = F(5, 8) if name == "P0" else F(3, 4)
        lower_gate = F(1, 2)
        if name in REFERENCES:
            upper = F(2303, 4096) + F(REFERENCES.index(name), 16)
        if name == "N":
            lower_gate, upper = F(1, 4), F(1, 2)
        C, R = (lower_gate + upper)/2, (upper-lower_gate)/2
        constant = F(-5, 16)+C if name == "N" else F(11, 16)-C
        require(h["nc"] == 3 and h["nb"] == 0 and not h["b"]
                and len(h["ub"]) == (4 if name == "N" else 6)
                and h["c"] == (constant,) and h["Gc"] == ({0: F(1, 8), 2: R},)
                and h["Gb"] == ({},), "registered constant-difference objective")
        sign = F(1) if name == "N" else F(-1)
        mc = {1: sign*R, 2: -R}
        require(flattened(h, "le")[-4:] == [mc, mc, {j: -v for j, v in mc.items()},
                                           {j: -v for j, v in mc.items()}]
                and h["ub"][-4:] == (F(0),)*4, "registered exact product rows")
        if name != "N":
            l = F(0) if name == "P0" else F(1, 2) if name == "P1" else F(REFERENCES.index(name), 4)
            u = l + (F(1, 2) if name in ("P0", "P1") else F(1, 4))
            require(flattened(h, "le")[:2] == [{0: F(1, 2)}, {0: F(-1, 2)}]
                    and h["ub"][:2] == (u-F(1, 2), F(1, 2)-l), "registered closed slice")
    return h


def algebra(name, record, work):
    """Check registered negative relaxed points / positive exact LP bounds."""
    from scoped_source.check_gate_margin_rows import source
    if name in REFERENCES:
        h = literal_native_form(name, record["off"], work)
        i = REFERENCES.index(name)
        # Two retained slice rows, then four McCormick rows.
        bound = lower(h, [(2, -1), (1, "-1/4")], work)
        value = feasible_value(h, [F(i, 2)-1, 1, -1], work)
        require(bound == value == F(1, 4096), "four-slice baseline exact bound")
        return {"scope": "given_native_F0_LP", "lower": str(bound),
                "exact_relaxation_minimum": str(value), "optimizer_calls": 0}
    proof = record["relation"]
    old, new = literal_native_form(name, proof["old_f0"], work), source(proof["new_f0"], work, f0=True)
    require(record["off"] == proof["old_f0"], "off/on original F0 difference")
    if name in ("P0", "P1", "N"):
        point = [-1, 1, -1] if name == "P0" else [0, 1, -1] if name == "P1" else [-1, -1, -1]
        value = feasible_value(old, point, work)
        expected = F(-3, 16) if name == "N" else F(-1, 16)
        require(value == expected, "registered negative relaxed point")
        base_rows = len(old["ub"])-4
        bound = lower(new, [(base_rows, -1), (len(old["ub"]), -1)], work)
        require(bound == F(1, 16), "native positive exact residual bound")
        return {"scope": "given_native_F0_LP", "old_relaxed_point": list(map(str, point)),
                "old_relaxed_value": str(value), "new_checked_lower": str(bound),
                "optimizer_calls": 0, "network_counterexample": False}
    if name in ("Z", "E"):
        bound = lower(new, [], work)
        require(bound == (F(1, 2) if name == "Z" else F(1, 8)), "degenerate exact box bound")
        return {"scope": "given_native_F0_LP", "unchanged_output_lower": str(bound)}
    return {"scope": "structural_mapping_or_fallback_only"}


def verify_row(row, work):
    from scoped_source.check_gate_margin_rows import check
    name = row["case"]
    if name in SPECS:
        proof = row["relation"]
        receipt = check(proof, expected_binding=row["binding_sha256"],
                        expected_context=expected_context(name), expected_property=([1.0], 0.0), work=work)
        expected = "NO_SIGN_PREMISE" if name == "X" else "GATE_EMBEDDING_REFUSED" if name == "R" else "APPLIED"
        require(receipt == row["receipt"] and receipt["status"] == expected, "fixed row outcome")
        if name == "Q":
            require(proof["premise"] == {"side": 1, "kind": "ROW", "row": 4, "scale": "-1"},
                    "private-row order/sign premise")
            h = proof["new_f0"]
            require(h["Gc"]["shape"][1] == 5 and h["Gb"]["shape"][1] == 3
                    and len(h["ub"]) == 11 and len(h["b"]) == 3, "private native dimensions")
        require(receipt["added_rows"] == (2 if expected == "APPLIED" else 0), "row count")
    return algebra(name, row, work)


def run(root, destination, deadline, phase_deadline, tick):
    import numpy as np
    import scipy.sparse as sp
    import torch
    from dataclasses import replace
    from act.back_end.solver.solver_hz import SparseHZono, HZSupportBoundsResult, Bounds
    from act.back_end.moe.weighted_top2 import (
        shared_input_pair_hz, WeightedTop2GateRange, WeightedTop2DifferenceRange, build_weighted_top2_f0)
    from act.back_end.moe.conditioned_difference_support import condition_on_affine_path_interval
    from act.back_end.moe.gate_margin_rows import bind_sources, snapshot, GateMarginBinding, apply_rows
    from scoped_source.check_gate_margin_rows import GATE_SEMANTICS, check, freeze, number
    from scoped_source.hz_request_store import Work
    from scoped_proof.io import save

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    work = Work(deadline)
    result = {"status": "FAIL", "records": [], "faults": [], "counts": {
        "native_F0_attempted": 0, "native_F0_returned": 0,
        "support_queries": 0, "optimizer_calls": 0, "model_forwards": 0},
        "shared_deadline": deadline}
    receipts = []
    prefix_index = 0

    def save_prefix():
        nonlocal prefix_index
        work.tick()
        ledger(work.counts)
        result["work_counts"] = dict(work.counts)
        save(destination / f"prefix_{prefix_index:02d}.json", result)
        prefix_index += 1
        tick()

    def H(c, gc, gb=None, *, ac=None, ab=None, b=(), auc=None, aub=None, ub=(), frame=901):
        gc = sp.csr_matrix(gc, dtype=np.float64)
        gb = sp.csr_matrix((len(c), 0), dtype=np.float64) if gb is None else sp.csr_matrix(gb, dtype=np.float64)
        nc, nb = gc.shape[1], gb.shape[1]
        matrix = lambda x, n, width: sp.csr_matrix((n, width), dtype=np.float64) if x is None else sp.csr_matrix(x, dtype=np.float64)
        return SparseHZono(c=np.asarray(c, dtype=np.float64), Gc=gc, Gb=gb,
                           Ac=matrix(ac, len(b), nc), Ab=matrix(ab, len(b), nb), b=np.asarray(b, dtype=np.float64),
                           Auc=matrix(auc, len(ub), nc), Aub=matrix(aub, len(ub), nb), ub=np.asarray(ub, dtype=np.float64),
                           frame_id=frame, exact=True)

    def fixture(name):
        frame = 901 + (SPECS + REFERENCES).index(name)
        h = lambda *a, **kw: H(*a, frame=frame, **kw)
        entry = h([.5], [[.5]])
        router = h([.5, 0], [[.5], [0]])
        a, b = h([-5/16], [[1/8]]), h([11/16], [[1/8]])
        scalar, interval, maps = entry, None, []
        margin, gate, difference = (0, 1), (.5, .75), (-1, -1)
        if name in ("P0", "P1") or name in REFERENCES:
            if name in REFERENCES:
                i = REFERENCES.index(name)
                interval = (i/4, (i+1)/4)
                gate = (.5, 2303/4096 + i/16)
            else:
                interval = (0, .5) if name == "P0" else (.5, 1)
                gate = (.5, .625) if name == "P0" else (.5, .75)
            margin = interval
        elif name == "N":
            router = h([-.5, 0], [[-.5], [0]])
            a, b = b, a
            margin, gate, difference = (-1, 0), (.25, .5), (1, 1)
        elif name == "Z":
            router = h([0, 0], [[0], [0]])
            a, b = h([0], [[0]]), h([1], [[0]])
            margin, gate = (0, 0), (.5, .5)
        elif name == "E":
            a, b = h([.25], [[.125]]), h([.25], [[.125]])
            difference = (0, 0)
        elif name == "Q":
            common = dict(ac=[[1]], ab=[[.5]], b=[0], auc=[[1]], aub=[[0]], ub=[1])
            entry = h([0], [[1]], [[0]], **common)
            router = h([0, 0], [[1], [0]], [[.25], [0]], **common)
            scalar = h([0], [[1]], [[.25]], **common)
            a = h([0], [[.125, .125]], [[.125, .125]], ac=[[1, 0], [0, 1]],
                  ab=[[.5, 0], [0, -1]], b=[0, 0], auc=[[1, 0], [0, 1]],
                  aub=[[0, 0], [0, 0]], ub=[1, .5])
            b = h([1], [[.125, -.125]], [[.125, -.125]], ac=[[1, 0], [0, 1]],
                  ab=[[.5, 0], [0, -1]], b=[0, 0], auc=[[1, 0], [0, -1]],
                  aub=[[0, 0], [0, 0]], ub=[1, .75])
            interval, maps, difference = (0, 1), [0], (-1.5, -.5)
        elif name == "X":
            entry = h([0], [[1]])
            router = h([0, 0], [[1], [0]])
            a, b = h([0], [[0]]), h([1], [[0]])
            margin, gate = (-1, 1), (.25, .75)
        elif name == "R":
            router = h([-2, 0], [[0], [0]])
            a, b = h([0], [[0]]), h([1], [[0]])
            margin, gate = (-2, -2), (.1, .2)
        pair = shared_input_pair_hz(entry, a, b)
        binding = None
        if name in SPECS:
            binding = bind_sources(entry=entry, expert_a=a, expert_b=b, router=router,
                                   pair_hz=pair, caller_context=expected_context(name), gate_semantics=GATE_SEMANTICS,
                                   router_eq_rows=maps, router_le_rows=maps, work=work)
        if interval is not None:
            cut = lambda target: condition_on_affine_path_interval(target, scalar, *interval)
            pair = replace(pair, output_hz=cut(pair.output_hz), input_hz=cut(pair.input_hz))
            router = cut(router)
        def support(bounds):
            return HZSupportBoundsResult(rows=(0,), bounds=Bounds(lb=torch.tensor([bounds[0]], dtype=torch.float64),
                ub=torch.tensor([bounds[1]], dtype=torch.float64)), lower_status=("supplied_literal",),
                upper_status=("supplied_literal",), solver_gap=(None,), elapsed=0., solves=0, exact=False)
        gr = WeightedTop2GateRange((0, 1), router, router.frame_id, 2, margin, gate, support(margin))
        dr = WeightedTop2DifferenceRange((0, 1), pair.output_hz, pair.output_hz.frame_id, 2,
                                         (1.,), difference, support(difference))
        return pair, router, binding, gr, dr

    native = {}
    try:
        for name in SPECS + REFERENCES:
            tick(); work.tick()
            before = dict(work.counts); started = time.monotonic()
            pair, router, binding, gr, dr = fixture(name)
            prepared = time.monotonic()
            off = on = None
            # Predetermined alternating order, not chosen by outcome.
            order = (False,) if name in REFERENCES else ((False, True) if SPECS.index(name) % 2 == 0 else (True, False))
            times = {}
            for enabled in order:
                work.tick(); tick()
                result["counts"]["native_F0_attempted"] += 1
                t = time.monotonic()
                value = build_weighted_top2_f0(pair, router, (0, 1), [1.], 0.,
                    difference_time_limit=0., gate_range=gr, difference_range=dr,
                    gate_margin_binding=binding if enabled else None, gate_margin_work=work if enabled else None)
                result["counts"]["native_F0_returned"] += 1
                times["on" if enabled else "off"] = time.monotonic()-t
                if enabled:
                    on = value
                else:
                    off = value
                    require(off.gate_margin_evidence is None, "default off must not emit relation evidence")
            row = dict(case=name, before=before, off=snapshot(off.output_hz, work, f0=True),
                       preparation_seconds=prepared-started, construction_seconds=times)
            if name in SPECS:
                row.update(relation=on.gate_margin_evidence["record"], receipt=on.gate_margin_evidence["receipt"],
                           binding_sha256=binding.sha256)
                native[name] = (pair, router, binding, off)
            row["algebra"] = verify_row(row, work)
            row["seconds_through_worker_check"] = time.monotonic()-started
            row["after"] = dict(work.counts)
            result["records"].append(row)
            receipts.append(save(destination / (name + ".json"), row))
            save_prefix()

        require(result["counts"]["native_F0_returned"] == 20, "exact native construction count")
        record = result["records"][0]
        private = result["records"][SPECS.index("Q")]

        def receive(changed, expected=None):
            return check(changed, expected_binding=record["binding_sha256"] if expected is None else expected,
                         expected_context=expected_context("P0" if expected is None else "Q"),
                         expected_property=([1.], 0.), work=work)

        def mutation(name):
            p = copy.deepcopy(private["relation"] if name in ("router_map", "private_metadata", "binary_shift") else record["relation"])
            anchor = private["binding_sha256"] if name in ("router_map", "private_metadata", "binary_shift") else None
            if name == "schema": p["schema"] = "OTHER"
            elif name == "extra_field": p["unexpected"] = True
            elif name == "request": p["binding"]["context"]["request"] = "0"*64
            elif name == "domain": p["binding"]["context"]["domain"] = "0"*64
            elif name == "pair": p["binding"]["context"]["pair"] = [1, 0]
            elif name == "property": p["parameters"]["q"] = [-1.]
            elif name == "gate_semantics": p["binding"]["gate_semantics"] = "ARBITRARY_CONVEX_GATE"
            elif name == "source_center": p["active"]["joint"]["c"][0] += 1
            elif name == "router_map": p["binding"]["router_eq_rows"] = []
            elif name == "private_metadata": p["binding"]["metadata"]["a_private_binary"] = 0
            elif name == "missing_active_rows":
                p["active"]["router"] = copy.deepcopy(p["binding"]["sources"]["router"])
            elif name == "sign_premise": p["premise"]["side"] = -1
            elif name == "inward_row": p["new_f0"]["ub"][-2] -= .125
            elif name == "lambda_slot": p["parameters"]["lambda_slot"] += 1
            elif name == "product_slot": p["parameters"]["product_slot"] -= 1
            elif name == "binary_shift": p["new_f0"]["Aub"]["indices"][-1] = 1
            elif name == "old_output": p["new_f0"]["c"][0] += 1
            elif name == "missing_new_row": p["new_f0"] = copy.deepcopy(p["old_f0"])
            else: raise AssertionError("unknown mutation")
            if name in ("router_map", "private_metadata"):
                # Deliberately rebind the malformed source: exercise structural
                # rejection, not merely an out-of-date hash.
                anchor = freeze(p["binding"], work)[1]
            receive(p, anchor)

        for name in FAULTS:
            tick(); before = dict(work.counts); start = time.monotonic()
            try:
                if name in FAULTS[:18]:
                    mutation(name)
                elif name == "missing_control_record":
                    inventory(result["records"][:-1], deadline, deadline, work.counts)
                elif name == "deadline_binding":
                    inventory(result["records"], deadline, deadline+1, work.counts)
                elif name == "ledger_reset":
                    inventory(result["records"], deadline, deadline, dict.fromkeys(work.counts, 0))
                elif name == "expired_work":
                    Work(time.monotonic()-1)
                elif name == "stored_bits":
                    number(1 << 4096, work)
                elif name in ("csr_index_dtype", "csr_duplicate"):
                    hz = copy.deepcopy(native["P0"][0].output_hz)
                    if name == "csr_index_dtype":
                        hz.Gc.indices = hz.Gc.indices.astype(np.float64)+.25
                    else:
                        hz.Gc = sp.csr_matrix((np.asarray([.125, .125]),
                            np.asarray([0, 0]), np.asarray([0, 2, 2])), shape=(2, 1))
                    snapshot(hz, work)
                else:
                    pair, router, binding, off = native["P0"]
                    params = record["relation"]["parameters"]
                    if name == "binding_digest":
                        binding = GateMarginBinding(binding.payload, "0"*64)
                    elif name == "binding_oversize":
                        binding = GateMarginBinding(b" "*(8*2**20+1), "0"*64)
                    elif name == "independent_mode":
                        pair = replace(pair, relation_mode="independent_inputs")
                    else:
                        raise AssertionError("unknown fault")
                    apply_rows(binding=binding, pair_hz=pair, router=router, pair=(0, 1), q=[1.], offset=0.,
                        old_f0=off.output_hz, lambda_slot=params["lambda_slot"], product_slot=params["product_slot"],
                        lambda_center=params["center"], lambda_radius=params["radius"], gate=params["gate"], work=work)
            except (ValueError, TimeoutError) as exc:
                if name == "expired_work":
                    require(type(exc) is TimeoutError, "deadline must be explicit")
                else:
                    require(type(exc) is ValueError, "unexpected deadline in nondeadline fault")
                require(str(exc) == ERRORS[FAULTS.index(name)], "wrong rejection cause: " + name + ": " + str(exc))
                result["faults"].append(dict(name=name, status="PASS", error=str(exc),
                    before=before, after=dict(work.counts), seconds=time.monotonic()-start))
            else:
                raise AssertionError("fault was accepted: " + name)
            save_prefix()
        inventory(result["records"], deadline, deadline, work.counts)
        all_faults(result["records"], result["faults"], work.counts)
        # Rational four-slice comparison: sigma(1/4) <= 1849/3289 < 2303/4096.
        require(F(1849, 3289) < F(2303, 4096) < F(9, 16), "fixed baseline interval")
        result["status"] = "PASS"
    except BaseException as exc:
        result["error"] = repr(exc)
    result.update(work_counts=dict(work.counts), record_receipts=receipts,
                  guarantee="conditional given-HZ row checks and exact synthetic LP bounds; no network SAFE",
                  hard_deadline=deadline, phase_deadline=phase_deadline)
    save_prefix()
    return result
