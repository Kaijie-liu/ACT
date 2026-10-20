"""Fixed stdlib cases, not a numerical benchmark or a trained request."""
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import time

from scoped_source.hz_request_store import Work, compact, digest, require
from scoped_source import check_margin_support as check

ROOT = Path("/data1/Kane/MOE/ACT")
CASES = (
    "nonconstant_shared_private", "diamond_nonpositive", "pair_reversal",
    "multiclass_constant", "fixed_gate", "median_tie_and_representation",
    "nonzero_dual_constraints", "source_binding_faults", "support_faults",
    "cell_coverage_faults", "immutable_capture", "capacity_faults",
    "deadline_and_work_faults",
)


def load_producer():
    spec = importlib.util.spec_from_file_location(
        "finite_margin_producer", ROOT/"act/back_end/moe/margin_conditioned_support.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source(n, a, b, ra, rb, *, q=None, constant="0", le=(), eq=(), bounds=None):
    return dict(
        schema=check.SCHEMA, request="a"*64, frame="synthetic_shared_frame",
        pair=[0, 1], legal_pair=[0, 1],
        factors=[[f"z{i}", "shared", *(bounds[i] if bounds else ("-1", "1"))] for i in range(n)],
        le=list(le), eq=list(eq), expert_a=a, expert_b=b,
        router_a=ra, router_b=rb, q=["1"] if q is None else q, constant=constant,
        semantics="GIVEN_CONTINUOUS_SELECTED_SOFTMAX_TOP2")


def main_source():
    value = source(3, [["9/32", "1/2", "1/256", "0"]],
                   [["-7/32", "-1/2", "0", "1/256"]],
                   ["0", "1/2", "0", "0"], ["0", "-1/2", "0", "0"])
    value["factors"][1][1], value["factors"][2][1] = "expert_a", "expert_b"
    return value


def diamond_source():
    rows = [["1", str(a), str(b)] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    return source(2, [["3/16", "0", "1/2"]], [["3/16", "0", "-1/2"]],
                  ["0", "1/2", "0"], ["0", "-1/2", "0"], le=rows)


def registered_sources():
    """Literal sources only; may be hashed at freeze without running controls."""
    reverse = main_source()
    reverse["pair"] = [1, 0]
    reverse["expert_a"], reverse["expert_b"] = reverse["expert_b"], reverse["expert_a"]
    reverse["router_a"], reverse["router_b"] = reverse["router_b"], reverse["router_a"]
    reverse["factors"][1][1], reverse["factors"][2][1] = "expert_b", "expert_a"
    multi = main_source()
    for key in ("expert_a", "expert_b"):
        multi[key] = [multi[key][0], ["1/8", "0", "0", "0"]]
    multi["q"], multi["constant"] = ["1", "-2"], "1/4"
    fixed = source(1, [["1", "1"]], [["1", "-1"]], ["0", "0"], ["0", "0"])
    ine = source(1, [["0", "-1"]], [["0", "-1"]], ["0", "0"], ["0", "0"], le=[["0", "1"]])
    eq = deepcopy(ine); eq["le"], eq["eq"] = [], [["0", "1"]]
    sliced = diamond_source(); sliced["le"].append(["-1/2", "1", "0"])
    return dict(main_zero=main_source(), main_median=main_source(), diamond=diamond_source(),
                reversed=reverse, two_classes_nonzero_constant=multi, fixed_half_gate=fixed,
                inequality_dual=ine, equality_dual=eq, slice_support=sliced)


def run(deadline, tick):
    work = Work(deadline)
    producer = load_producer()
    rows, certificates = [], []
    result = dict(status="FAIL", cases=rows, certificates=certificates,
                  deadline=deadline, work_counts=work.counts)
    current = None

    def checked(obj, mode="median", cells=128, candidates=None, label="control"):
        raw = compact(obj); key = digest(raw)
        parsed = check.inspect(raw, key, work)
        _, goals, _ = check.plan(parsed, mode, cells, work)
        if candidates is None:
            candidates = [(tuple(F(0) for _ in parsed["le"]),
                           tuple(F(0) for _ in parsed["eq"])) for _ in goals]
        record = producer.build(raw, expected_source=key, mode=mode, cells=cells,
                                duals=candidates, work=work)
        receipt = check.check(raw, record, expected_source=key,
                              expected_mode=mode, expected_cells=cells, work=work)
        certificates.append(dict(label=label, source=obj, source_sha256=key,
                                 certificate=record, receipt=receipt))
        return record, receipt, raw

    def reject(call, expected):
        try:
            call()
        except (ValueError, TimeoutError) as exc:
            require(expected in str(exc), f"wrong refusal: {exc!r}; need {expected}")
            return str(exc)
        raise AssertionError("fault accepted: "+expected)

    saved = {}

    def positive():
        off, _, raw = checked(main_source(), "zero", label="main_zero")
        on, receipt, _ = checked(main_source(), label="main_median")
        require([r["lower"] for r in off["supports"]] == ["-89/256", "-25/256"],
                "zero-tilt exact supports")
        require([(r["nu"], r["lower"]) for r in on["supports"]] ==
                [("0", "-89/256"), ("1/4", "-25/256"), ("0", "-25/256"), ("-1/4", "39/256")],
                "selected tilted supports")
        require(F(off["lower"]) < 0 < F(on["lower"]) and F(on["lower"]) >= F(3, 512),
                "finite scalar separation")
        saved.update(raw=raw, record=on, receipt=receipt)
        return dict(off_lower=off["lower"], on_lower=on["lower"],
                    off_objective_slots=2, on_objective_slots=4, actual_native_calls=0)

    def diamond():
        record, receipt, _ = checked(diamond_source(), label="diamond")
        require(record["lower"] == "-1/16" and receipt["status"] == "NONPOSITIVE_BOUND"
                and len(record["supports"]) == 2, "diamond no rescue")
        return receipt

    def reversal():
        obj = main_source()
        obj["pair"] = [1, 0]
        obj["expert_a"], obj["expert_b"] = obj["expert_b"], obj["expert_a"]
        obj["router_a"], obj["router_b"] = obj["router_b"], obj["router_a"]
        obj["factors"][1][1], obj["factors"][2][1] = "expert_b", "expert_a"
        record, receipt, _ = checked(obj, label="reversed")
        require(record["lower"] == saved["record"]["lower"], "pair reversal equality")
        return receipt

    def multiclass():
        obj = main_source()
        for key in ("expert_a", "expert_b"):
            obj[key] = [obj[key][0], ["1/8", "0", "0", "0"]]
        obj["q"], obj["constant"] = ["1", "-2"], "1/4"
        record, receipt, _ = checked(obj, label="two_classes_nonzero_constant")
        require(record["lower"] == saved["record"]["lower"], "property constant counted once")
        return receipt

    def fixed():
        obj = source(1, [["1", "1"]], [["1", "-1"]], ["0", "0"], ["0", "0"])
        record, receipt, _ = checked(obj, cells=8, label="fixed_half_gate")
        require(record["lower"] == "1" and len(record["supports"]) == len(record["cells"]) == 1,
                "fixed actual half gate")
        return receipt

    def median_cases():
        obj = source(2, [["0", "1", "-2"]], [["0", "1", "-2"]],
                     ["0", "1", "1"], ["0", "0", "0"])
        raw = compact(obj); p = check.inspect(raw, digest(raw), work)
        require(check.median_tilt(p, F(1, 2), work) == 0, "closest zero median interval")
        original = source(1, [["0", "-1"]], [["0", "-1"]], ["0", "1"], ["0", "0"])
        duplicate = source(2, [["0", "0", "-1"]], [["0", "0", "-1"]],
                           ["0", "1", "0"], ["0", "0", "0"], eq=[["0", "1", "-1"]])
        tilts = []
        for item in (original, duplicate):
            raw = compact(item)
            tilts.append(str(check.median_tilt(check.inspect(raw, digest(raw), work), F(1, 2), work)))
        require(tilts == ["1", "0"], "retained equivalent-representation limitation")
        return dict(tied_median="0", equivalent_source_tilts=tilts)

    def constraints():
        # Same fixed-half-gate objective -z, with z<=0. Correct dual is -1.
        obj = source(1, [["0", "-1"]], [["0", "-1"]], ["0", "0"], ["0", "0"],
                     le=[["0", "1"]])
        record, receipt, _ = checked(obj, candidates=[((F(-1),), ())], label="inequality_dual")
        require(record["lower"] == "0", "nonzero inequality dual")
        eq = deepcopy(obj); eq["le"], eq["eq"] = [], [["0", "1"]]
        record, _, _ = checked(eq, candidates=[((), (F(-1),))], label="equality_dual")
        require(record["lower"] == "0", "nonzero free equality dual")
        # Explicit diamond slice certificate, not merely a zero-dual box proof.
        obj = diamond_source(); obj["le"].append(["-1/2", "1", "0"])
        raw = compact(obj); key = digest(raw); parsed = check.inspect(raw, key, work)
        item = dict(t="1/4", nu="0", y=["0", "0", "-1/4", "0", "-1/4"], z=[], lower="1/16")
        bound = check.checked_support(parsed, F(1, 4), F(0), item, work)
        require(bound == F(1, 16), "retained slice support")
        saved.update(constrained_source=obj, constrained_support=item)
        return dict(inequality=receipt, slice_lower=str(bound), slice_source=obj, slice_support=item)

    def sources():
        errors = []
        mutations = (
            lambda o: o.update(request="b"*64),
            lambda o: o.update(pair=[1, 0]),
            lambda o: o.update(constant="1"),
            lambda o: o["factors"][1].__setitem__(0, "foreign_factor"),
            lambda o: o["router_a"].__setitem__(0, "1/8"),
            lambda o: o["expert_a"][0].__setitem__(2, "0"),
        )
        for mutate in mutations:
            obj = main_source(); mutate(obj)
            errors.append(reject(lambda: check.check(compact(obj), saved["record"],
                                                      expected_source=digest(saved["raw"]),
                                                      expected_mode="median", expected_cells=128, work=work),
                                 "source identity"))
        obj = saved["constrained_source"]
        raw = compact(obj); key = digest(raw)
        for operation in ("delete", "reorder"):
            mutated = deepcopy(obj)
            if operation == "delete":
                mutated["le"].pop()
            else:
                mutated["le"].reverse()
            errors.append(reject(lambda: check.inspect(compact(mutated), key, work), "source identity"))
        # Even with a newly bound hash, incompatible declared ownership is invalid.
        for key, index, reason in (("router_a", 2, "router private factor"),
                                   ("expert_a", 3, "expert a foreign private factor")):
            changed = main_source()
            row = changed[key][0] if key.startswith("expert") else changed[key]
            row[index] = "1"
            data = compact(changed)
            errors.append(reject(lambda: check.inspect(data, digest(data), work), reason))
        return dict(refusals=errors)

    def supports():
        errors = []
        for mutation, reason in (
            (lambda r: r["supports"].pop(), "complete support"),
            (lambda r: r["supports"][1].update(nu="-1/4"), "objective binding"),
            (lambda r: r["supports"][1].update(lower="1"), "support residual"),
            (lambda r: r.update(source="c"*64), "certificate identity"),
            (lambda r: r.update(mode="zero"), "frozen policy binding"),
            (lambda r: r.update(cell_count=64), "frozen policy binding"),
        ):
            record = deepcopy(saved["record"]); mutation(record)
            errors.append(reject(lambda: check.check(saved["raw"], record,
                                                    expected_source=digest(saved["raw"]),
                                                    expected_mode="median", expected_cells=128, work=work), reason))
        obj = saved["constrained_source"]; raw = compact(obj)
        parsed = check.inspect(raw, digest(raw), work)
        item = deepcopy(saved["constrained_support"]); item["y"][2] = "1/4"
        errors.append(reject(lambda: check.checked_support(parsed, F(1, 4), F(0), item, work),
                             "dual sign"))
        return dict(refusals=errors)

    def coverage():
        errors = []
        for mutation, reason in (
            (lambda r: r["cells"].pop(64), "complete cells"),
            (lambda r: r["cells"].__setitem__(64, deepcopy(r["cells"][63])), "cell coverage"),
            (lambda r: r["cells"][0].update(gate=["1/2", "1/2"]), "sigmoid premise"),
            (lambda r: r["cells"][0].update(lower="1"), "cell bound"),
            (lambda r: r.update(lower="1"), "aggregate lower"),
        ):
            record = deepcopy(saved["record"]); mutation(record)
            errors.append(reject(lambda: check.check(saved["raw"], record,
                                                    expected_source=digest(saved["raw"]),
                                                    expected_mode="median", expected_cells=128, work=work), reason))
        return dict(refusals=errors)

    def aliases():
        obj = main_source(); raw = compact(obj); key = digest(raw)
        obj["expert_a"][0][0] = "999"
        receipt = check.check(raw, saved["record"], expected_source=key,
                              expected_mode="median", expected_cells=128, work=work)
        error = reject(lambda: check.check(compact(obj), saved["record"], expected_source=key,
                                          expected_mode="median", expected_cells=128, work=work),
                       "source identity")
        return dict(immutable_receipt=receipt, changed_live_source=error)

    def capacity():
        obj = main_source(); obj["factors"] *= 6
        raw = compact(obj)
        errors = [reject(lambda: check.inspect(raw, digest(raw), work), "factor capacity")]
        obj = main_source(); obj["constant"] = str(2**4096)
        raw = compact(obj)
        errors.append(reject(lambda: check.inspect(raw, digest(raw), work), "rational capacity"))
        obj = main_source(); obj["constant"] = "NaN"; raw = compact(obj)
        errors.append(reject(lambda: check.inspect(raw, digest(raw), work), "canonical stored rational"))
        r = deepcopy(saved["record"]); r["cell_count"] = 129
        errors.append(reject(lambda: check.check(saved["raw"], r, expected_source=digest(saved["raw"]),
                                                expected_mode="median", expected_cells=129, work=work), "cell capacity"))
        # This is a real checked intermediate overflow, not a quota reset.
        errors.append(reject(lambda: work.exact(F(2**8192)), "intermediate rational capacity"))
        return dict(refusals=errors)

    def timing():
        class InjectedStop:
            """Private fault adapter: same actual ledger, no clock/module patch."""
            def __init__(self, after, error):
                self.left, self.error = after, error
            def tick(self):
                work.tick()
                self.left -= 1
                if self.left < 0:
                    raise self.error
            def visit(self, *args):
                work.visit(*args); self.tick()
            def exact(self, q):
                value = work.exact(q); self.tick(); return value
        errors = []
        for count, exc in ((0, TimeoutError("injected deadline")),
                           (1500, TimeoutError("injected deadline")),
                           (100, ValueError("injected Work stop"))):
            errors.append(reject(
                lambda: check.check(saved["raw"], saved["record"],
                                    expected_source=digest(saved["raw"]),
                                    expected_mode="median", expected_cells=128,
                                    work=InjectedStop(count, exc)), str(exc)))
        return dict(refusals=errors, real_work_never_reset=True)

    functions = (positive, diamond, reversal, multiclass, fixed, median_cases, constraints,
                 sources, supports, coverage, aliases, capacity, timing)
    try:
        for current, call in zip(CASES, functions):
            tick(); started = time.monotonic()
            detail = call(); tick()
            rows.append(dict(name=current, status="PASS", seconds=time.monotonic()-started, detail=detail))
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(error=repr(exc), failed_case=current)
    result["work_counts"] = dict(work.counts)
    return result
