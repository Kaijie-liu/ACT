"""New sparse scalar integration roster, not a rerun of native source controls."""
import copy
import hashlib
import time
from fractions import Fraction as F

from scoped_source.development_work import DevelopmentWork, POLICY
from scoped_source.hz_request_store import compact, decode, digest, require
from scoped_source import check_margin_hz_source as source_check
from scoped_source import check_native_margin_support as scalar
from margin_native_cases import fixtures, PROPERTIES, rejected

CELLS = 8
PROPERTY = PROPERTIES[0]
MODES = ("zero", "median", "matched_zero")
SCOPE = {k: hashlib.sha256(("A1_SCALAR_20261021_"+k).encode()).hexdigest()
         for k in ("request", "domain", "guard", "model", "input")}
FAULTS = ("plan_mode", "grid_cut", "missing_cell", "duplicate_cell", "cell_gate",
          "cell_lower", "aggregate_lower", "missing_support", "property_binding")
REASONS = ("caller-bound sparse plan", "caller-bound sparse plan", "complete scalar cells",
           "closed scalar coverage", "scalar sigmoid enclosure", "scalar cell lower mismatch",
           "scalar aggregate lower mismatch", "complete candidates", "scalar certificate identity/grade")


def receive(value, work):
    require(value["status"] == "PASS" and value["policy"] == POLICY, "successful scalar worker")
    require(value["full_calls_attempted"] == 2 and value["native_candidate_calls"] == 2
            and value["objective_slots"] == 6,
            "frozen native scalar call roster")
    require(set(value["faults"]) == set(FAULTS) and value["expired_reception"]["type"] == "TimeoutError",
            "complete scalar fault roster")
    require(all(value["faults"][name]["type"] == "ValueError"
                and reason in value["faults"][name]["reason"] for name, reason in zip(FAULTS, REASONS)),
            "specific scalar rejection reasons")
    require(value["source"]["scope"] == SCOPE, "frozen scalar scope")
    base = value["source"]["base"]
    require(base["context"]["pair"] == [0, 2]
            and all(base["sources"][k] == v for k, v in fixtures().items())
            and base["router_eq_rows"] == [1] and base["router_le_rows"] == [1], "original scalar source")
    raw = compact(value["source"])
    require(digest(raw) == value["source_sha256"], "scalar source anchor")
    require(value["view"]["q"] == PROPERTY["q"]
            and value["view"]["constant"] == PROPERTY["constant"], "complete ten-class property")
    require(set(value["records"]) == set(MODES), "all three scalar records")
    receipts = {}
    for name in MODES:
        receipt = scalar.check(raw, compact(value["view"]), value["records"][name]["record"],
                               expected_source=value["source_sha256"],
                               expected_property=value["property_sha256"],
                               expected_mode="median" if name == "median" else "zero",
                               expected_cells=CELLS, work=work)
        require(receipt == value["records"][name]["receipt"], "scalar independent receipt")
        receipts[name] = receipt
    median_supports = value["records"]["median"]["record"]["supports"]
    common = [s for s in median_supports if F(s["nu"]) == 0]
    require(common == value["records"]["matched_zero"]["record"]["supports"], "identical common supports")
    require(F(receipts["median"]["lower"]) >= F(receipts["matched_zero"]["lower"]),
            "finite tilt monotonicity with identical zero certificates")
    work.tick()
    return receipts


def run(deadline, publish):
    work = DevelopmentWork(deadline, policy=POLICY)
    start = time.monotonic()
    result = dict(status="FAIL", policy=POLICY, deadline=deadline, records={}, faults={},
                  native_candidate_calls=0, objective_slots=0, full_calls_attempted=0, complete_request=False)
    try:
        import importlib.metadata
        import numpy as np
        import scipy.sparse as sp
        import torch
        from act.back_end.solver.solver_hz import SparseHZono
        from act.back_end.moe.margin_hz_source import bind_source
        from act.back_end.moe.native_margin_support import support_property, build
        torch.set_num_threads(1)
        result["native_import_seconds"] = time.monotonic()-start
        result["versions"] = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch")}
        def native(obj):
            args = dict(obj)
            for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                m = args[key]
                args[key] = sp.csr_matrix((m["data"], m["indices"], m["indptr"]),
                                          shape=m["shape"], dtype=np.float64)
            return SparseHZono(**args)
        begun = time.monotonic()
        live = {k: native(v) for k, v in fixtures().items()}
        bound = bind_source(entry=live["entry"], expert_a=live["a"], expert_b=live["b"],
                            router=live["router"], pair=(0, 2), scope=SCOPE,
                            router_eq_rows=(1,), router_le_rows=(1,), work=work)
        view, prop_hash = source_check.make_property(bound.payload, expected_source=bound.sha256,
                                                     q=PROPERTY["q"], constant=PROPERTY["constant"], work=work)
        result.update(source=decode(bound.payload), source_sha256=bound.sha256,
                      view=decode(view), property_sha256=prop_hash,
                      source_and_property_seconds=time.monotonic()-begun)
        for mode, slots in (("zero", 2), ("median", 4)):
            result["full_calls_attempted"] += 1
            record = support_property(bound, view, expected_property=prop_hash,
                                      mode=mode, cells=CELLS, work=work)
            result["native_candidate_calls"] += record["native_candidate_calls"]
            result["objective_slots"] += record["objective_slots"]
            plan = record["record"]["plan"]
            require(plan["margin"] == ["-5/16", "21/16"] and plan["anchors"] == ["27/64", "53/64"],
                    "analytic source range and anchors")
            expected = [[t, nu] for t in ("27/64", "53/64")
                        for nu in (("0", "-1/128") if mode == "median" else ("0",))]
            require(plan["goals"] == expected and record["objective_slots"] == slots,
                    "analytic sparse median and full roster")
            result["records"][mode] = record
            publish(result, work)
        plan = result["records"]["zero"]["record"]["plan"]
        common = [s for s in result["records"]["median"]["record"]["supports"] if F(s["nu"]) == 0]
        begun = time.monotonic()
        record = build(plan, common, work)
        receipt = scalar.check(bound.payload, view, record, expected_source=bound.sha256,
                                expected_property=prop_hash, expected_mode="zero", expected_cells=CELLS, work=work)
        result["records"]["matched_zero"] = dict(record=record, receipt=receipt,
                                                   scalar_only_seconds=time.monotonic()-begun,
                                                   native_candidate_calls=0, objective_slots=0)
        def accept(record):
            return scalar.check(bound.payload, view, record, expected_source=bound.sha256,
                                expected_property=prop_hash, expected_mode="median", expected_cells=CELLS, work=work)
        for name, reason in zip(FAULTS, REASONS):
            bad = copy.deepcopy(result["records"]["median"]["record"])
            if name == "plan_mode": bad["plan"]["mode"] = "zero"
            elif name == "grid_cut": bad["plan"]["grid"][0][1] = "0"
            elif name == "missing_cell": bad["cells"].pop()
            elif name == "duplicate_cell": bad["cells"][1] = copy.deepcopy(bad["cells"][0])
            elif name == "cell_gate": bad["cells"][0]["gate"][0] = "0"
            elif name == "cell_lower": bad["cells"][0]["lower"] = "123"
            elif name == "aggregate_lower": bad["lower"] = "123"
            elif name == "missing_support": bad["supports"].pop()
            elif name == "property_binding": bad["property"] = "0"*64
            result["faults"][name] = rejected(lambda: accept(bad), reason)
        # Expired continuation preserves counts and never resets the live account.
        # Its deliberate 60ms wait is inside the outer batch budget.
        expired = DevelopmentWork(min(deadline, time.monotonic()+.05), policy=POLICY)
        expired.continue_from(work.counts)
        time.sleep(.06)
        result["expired_reception"] = rejected(lambda: scalar.check(
            bound.payload, view, record, expected_source=bound.sha256,
            expected_property=prop_hash, expected_mode="zero", expected_cells=CELLS, work=expired), "deadline")
        require(expired.counts == work.counts, "expired scalar reception did no unmetered traversal")
        result["status"] = "PASS"
        result["worker_reception"] = receive(result, work)
        result["separate_native_arm_delta"] = str(F(result["records"]["median"]["receipt"]["lower"])
                                                   -F(result["records"]["zero"]["receipt"]["lower"]))
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    result.update(work_counts=dict(work.counts), elapsed_before_worker_export=time.monotonic()-start)
    return result
