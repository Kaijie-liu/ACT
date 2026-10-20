"""Frozen new rational-slab interface controls; not an A2 performance study."""
import copy
import hashlib
import time
from fractions import Fraction as F

from scoped_source.development_work import DevelopmentWork, POLICY
from scoped_source.hz_request_store import compact, decode, digest, require
from scoped_source import check_margin_hz_source as source_check
from scoped_source import check_margin_slices as check
from margin_native_cases import fixtures, PROPERTIES, csr, rejected

CUTS = (F(1, 3),)
PROPERTY = PROPERTIES[0]
SCOPE = {k: hashlib.sha256(("SLICE_INTERFACE_20261021_"+k).encode()).hexdigest()
         for k in ("request", "domain", "guard", "model", "input")}
FAULTS = ("slab_rhs", "cell_margin", "cell_index", "partition", "source", "property",
          "missing_support", "missing_slab", "swap_slab", "positive_slab", "objective",
          "lower", "missing_original_row", "missing_cell", "reordered_cell", "invalid_cut")
REASONS = ("caller-bound closed slice",)*4+("slice record identity/grade",)*2+(
    "complete slice supports", "two ordered slab duals", "slice exact residual lower mismatch",
    "slice inequality dual sign", "slice objective binding", "slice exact residual lower mismatch",
    "slice original dual rows", "complete closed partition", "caller-bound closed slice",
    "strict ordered interior slice cuts")


def originals(mode):
    value = fixtures()
    if mode in ("constant_nonzero_margin", "zero_margin"):
        r = value["router"]
        r["Gc"] = csr(3, r["Gc"]["shape"][1], [])
        r["Gb"] = csr(3, r["Gb"]["shape"][1], [])
        if mode == "zero_margin": r["c"][2] = r["c"][0]
    return value


def direct_lower(source, view, cell, item, work):
    """Second coordinate formulation from all original output rows, not b/d/m."""
    hz = source["augmented"]
    nc, nb = hz["Gc"]["shape"][1], hz["Gb"]["shape"][1]
    q = list(map(F, view["q"])); k = len(q)
    kind, value = item["goal"][0], F(item["goal"][1])
    weights = ([value*v for v in q]+[(1-value)*v for v in q]+[F(0)]*2 if kind == "mix"
               else [value*v for v in q]+[-value*v for v in q]+[F(0)]*2)
    center = F(view["constant"]) if kind == "mix" else F(0)
    residual = [F(0)]*(nc+nb); margin = [F(0)]*(nc+nb)
    def add_row(matrix, i, multiplier, target, shift):
        for p in range(matrix["indptr"][i], matrix["indptr"][i+1]):
            work.visit("coefficient_visits")
            j = matrix["indices"][p]+shift
            target[j] = work.exact(target[j]+work.exact(multiplier*F(matrix["data"][p])))
    for i, weight in enumerate(weights):
        work.visit("row_visits")
        center = work.exact(center+work.exact(weight*F(hz["c"][i])))
        for key, shift in (("Gc", 0), ("Gb", nc)):
            add_row(hz[key], i, weight, residual, shift)
    m0 = work.exact(F(hz["c"][2*k])-F(hz["c"][2*k+1]))
    for i, weight in ((2*k, F(1)), (2*k+1, F(-1))):
        for key, shift in (("Gc", 0), ("Gb", nc)):
            add_row(hz[key], i, weight, margin, shift)
    for duals, ck, bk, rhs in ((item["y"], "Auc", "Aub", "ub"), (item["z"], "Ac", "Ab", "b")):
        require(len(duals) == len(hz[rhs]), "direct original row count")
        for i, value in enumerate(duals):
            work.visit("row_visits")
            multiplier = F(value)
            center = work.exact(center+work.exact(multiplier*F(hz[rhs][i])))
            add_row(hz[ck], i, -multiplier, residual, 0)
            add_row(hz[bk], i, -multiplier, residual, nc)
    p, qdual = map(F, item["slice_duals"])
    x, y = map(F, cell["margin"])
    center = work.exact(center+work.exact(p*work.exact(y-m0))+work.exact(qdual*work.exact(m0-x)))
    for j in range(nc+nb):
        work.visit("coefficient_visits")
        residual[j] = work.exact(residual[j]+work.exact((qdual-p)*margin[j]))
        center = work.exact(center-abs(residual[j]))
    return center


def receive(value, work):
    require(value["status"] == "PASS" and value["policy"] == POLICY, "successful slice worker")
    require(value["full_calls_attempted"] == value["native_candidate_calls"] == 2
            and value["objective_slots"] == 8, "fixed slice native call roster")
    require(set(value["faults"]) == set(FAULTS) and value["expired_reception"]["type"] == "TimeoutError",
            "complete new slice fault roster")
    for name, reason in zip(FAULTS, REASONS):
        require(value["faults"][name]["type"] == "ValueError" and reason in value["faults"][name]["reason"],
                "specific slice rejection reason")
    require(set(value["objects"]) == {"native", "constant_nonzero_margin", "zero_margin", "constant_property"},
            "fixed slice object roster")
    receipts = {}
    for name, obj in value["objects"].items():
        require(type(obj["records"]) is list and len(obj["records"]) == (2 if name == "native" else 1),
                "exact per-object slice record roster")
        require(all(item["native_candidate_calls"] == (1 if name == "native" else 0)
                    and (name != "native" or item["objective_slots"] == 4) for item in obj["records"]),
                "record-level slice call accounting")
        src, prop = obj["source"], obj["view"]
        require(src["scope"] == dict(SCOPE, request=hashlib.sha256((SCOPE["request"]+name).encode()).hexdigest())
                and src["base"]["context"]["pair"] == [0, 2]
                and src["base"]["router_eq_rows"] == [1] and src["base"]["router_le_rows"] == [1]
                and all(src["base"]["sources"][k] == v for k, v in originals(name).items()), "fixed slice source")
        require(prop["q"] == (["0"]*10 if name == "constant_property" else PROPERTY["q"])
                and prop["constant"] == PROPERTY["constant"], "full slice property binding")
        raw, view = compact(src), compact(prop)
        require(digest(raw) == obj["source_sha256"] and digest(view) == obj["property_sha256"], "slice source hash")
        cuts = CUTS if name in ("native", "constant_property") else ()
        args = dict(expected_source=obj["source_sha256"], expected_property=obj["property_sha256"],
                    cuts=cuts, work=work)
        parsed, plan, identity = check.prepare(raw, view, **args)
        if name == "native":
            receipt = check.aggregate(raw, view, [s["record"] for s in obj["records"]], differences=True, **args)
        else:
            goals = check.endpoint_goals(plan["cells"][0], work, differences=True)
            receipt = check.check(raw, view, obj["records"][0]["record"], expected_index=0,
                                  expected_goals=goals, **args)
        require(receipt == obj["receipt"], "independent slice receipt")
        for item in obj["records"]:
            for support in item["record"]["supports"]:
                require(direct_lower(src, prop, item["record"]["cell"], support, work) == F(support["lower"]),
                        "direct coordinate slice residual")
        receipts[name] = receipt
    obj = value["objects"]["native"]
    raw, view = compact(obj["source"]), compact(obj["view"])
    args = dict(expected_source=obj["source_sha256"], expected_property=obj["property_sha256"],
                cuts=CUTS, work=work)
    _, plan, _ = check.prepare(raw, view, **args)
    fixed = value["fixed_nonzero_slab_record"]
    check.check(raw, view, fixed, expected_index=0,
                expected_goals=check.endpoint_goals(plan["cells"][0], work, differences=True), **args)
    for support in fixed["supports"]:
        require(direct_lower(obj["source"], obj["view"], fixed["cell"], support, work) == F(support["lower"]),
                "fixed slab direct differential")
    return receipts


def run(deadline, publish):
    work = DevelopmentWork(deadline, policy=POLICY); start = time.monotonic()
    result = dict(status="FAIL", policy=POLICY, deadline=deadline, objects={}, faults={}, records={},
                  native_candidate_calls=0, full_calls_attempted=0, objective_slots=0, complete_request=False)
    try:
        import importlib.metadata
        import numpy as np
        import scipy.sparse as sp
        import torch
        from act.back_end.solver.solver_hz import SparseHZono
        from act.back_end.moe.margin_hz_source import bind_source
        from act.back_end.moe.margin_slice_support import propose
        torch.set_num_threads(1)
        result["native_import_seconds"] = time.monotonic()-start
        result["versions"] = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch")}
        def native(obj):
            args = dict(obj)
            for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                m = args[key]
                args[key] = sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=m["shape"], dtype=np.float64)
            return SparseHZono(**args)
        for name in ("native", "constant_nonzero_margin", "zero_margin", "constant_property"):
            begun = time.monotonic()
            live = {k: native(v) for k, v in originals(name).items()}
            scope = dict(SCOPE, request=hashlib.sha256((SCOPE["request"]+name).encode()).hexdigest())
            bound = bind_source(entry=live["entry"], expert_a=live["a"], expert_b=live["b"], router=live["router"],
                                pair=(0, 2), scope=scope, router_eq_rows=(1,), router_le_rows=(1,), work=work)
            view, propid = source_check.make_property(bound.payload, expected_source=bound.sha256,
                q=["0"]*10 if name == "constant_property" else PROPERTY["q"], constant=PROPERTY["constant"], work=work)
            args = dict(expected_source=bound.sha256, expected_property=propid,
                        cuts=CUTS if name in ("native", "constant_property") else (), work=work)
            parsed, plan, identity = check.prepare(bound.payload, view, **args)
            obj = dict(source=decode(bound.payload), view=decode(view), source_sha256=bound.sha256,
                       property_sha256=propid, records=[])
            if name == "native":
                require(plan["margin"] == ["-5/16", "21/16"]
                        and plan["cells"][0]["slab_rhs"] == ["-1/6", "13/16"], "nonbinary rational slab offsets")
                for index, cell in enumerate(plan["cells"]):
                    result["full_calls_attempted"] += 1
                    item = propose(bound, view, expected_property=propid, cuts=CUTS, index=index,
                                   goals=check.endpoint_goals(cell, work, differences=True), work=work)
                    result["native_candidate_calls"] += item["native_candidate_calls"]
                    result["objective_slots"] += item["objective_slots"]
                    obj["records"].append(item)
                    result["objects"][name] = obj
                    result["records"][str(index)] = "completed_native_slice"
                    publish(result, work)
                obj["receipt"] = check.aggregate(bound.payload, view, [s["record"] for s in obj["records"]],
                                                  differences=True, **args)
            else:
                cell = plan["cells"][0]; goals = check.endpoint_goals(cell, work, differences=True)
                require(len(plan["cells"]) == (2 if name == "constant_property" else 1), "constant margin coverage")
                if name == "constant_nonzero_margin": require(cell["gate"] == ["1/2", "5/8"], "nonzero point loose gate")
                if name == "zero_margin": require(cell["gate"] == ["1/2", "1/2"], "zero point exact gate")
                supports = []
                for spec in goals:
                    item = dict(goal=[spec[0], str(spec[1])], y=["0"]*len(parsed["joint"]["ub"]),
                                z=["0"]*len(parsed["joint"]["b"]),
                                slice_duals=["0", "0"] if name == "constant_property" else ["-1/3", "-1/5"])
                    item["lower"] = str(direct_lower(obj["source"], obj["view"], cell, item, work))
                    supports.append(item)
                record = dict(schema=check.CERT, source=bound.sha256, property=propid, partition=identity,
                              cell=cell, supports=supports, grade=check.GRADE)
                obj["records"] = [dict(record=record, native_candidate_calls=0)]
                obj["receipt"] = check.check(bound.payload, view, record, expected_index=0, expected_goals=goals, **args)
                if name == "constant_property":
                    require([s["lower"] for s in supports] == ["1/8", "1/8", "0", "0"],
                            "constant property and zero differences exact")
            obj["construction_proposal_check_seconds"] = time.monotonic()-begun
            result["objects"][name] = obj
        obj = result["objects"]["native"]
        raw, view = compact(obj["source"]), compact(obj["view"])
        args = dict(expected_source=obj["source_sha256"], expected_property=obj["property_sha256"], cuts=CUTS, work=work)
        _, plan, _ = check.prepare(raw, view, **args)
        goals = check.endpoint_goals(plan["cells"][0], work, differences=True)
        # A fixed nonzero dual record makes swapped slab signs observable even
        # if the optimizer selected zero multipliers. This is not a new query.
        fixed = copy.deepcopy(obj["records"][0]["record"])
        for s in fixed["supports"]:
            s["slice_duals"] = ["-1/3", "-1/5"]
            s["lower"] = str(direct_lower(obj["source"], obj["view"], fixed["cell"], s, work))
        check.check(raw, view, fixed, expected_index=0, expected_goals=goals, **args)
        result["fixed_nonzero_slab_record"] = fixed
        for name, reason in zip(FAULTS, REASONS):
            bad = copy.deepcopy(fixed)
            if name == "slab_rhs": bad["cell"]["slab_rhs"][0] = "1/6"
            elif name == "cell_margin": bad["cell"]["margin"].reverse()
            elif name == "cell_index": bad["cell"]["index"] = 1
            elif name == "partition": bad["partition"] = "0"*64
            elif name in ("source", "property"): bad[name] = "0"*64
            elif name == "missing_support": bad["supports"].pop()
            elif name == "missing_slab": bad["supports"][0]["slice_duals"].pop()
            elif name == "swap_slab": bad["supports"][0]["slice_duals"].reverse()
            elif name == "positive_slab": bad["supports"][0]["slice_duals"][0] = "1"
            elif name == "objective": bad["supports"][0]["goal"] = ["difference", "1"]
            elif name == "lower": bad["supports"][0]["lower"] = "123"
            elif name == "missing_original_row": bad["supports"][0]["y"].pop()
            if name in ("missing_cell", "reordered_cell"):
                records = [r["record"] for r in obj["records"]]
                records = records[:-1] if name == "missing_cell" else records[::-1]
                result["faults"][name] = rejected(lambda: check.aggregate(raw, view, records, differences=True, **args), reason)
            elif name == "invalid_cut":
                result["faults"][name] = rejected(lambda: check.prepare(raw, view, **dict(args, cuts=(F(2),))), reason)
            else:
                result["faults"][name] = rejected(lambda: check.check(raw, view, bad, expected_index=0,
                                                                      expected_goals=goals, **args), reason)
        expired = DevelopmentWork(min(deadline, time.monotonic()+.05), policy=POLICY)
        expired.continue_from(work.counts); time.sleep(.06)
        result["expired_reception"] = rejected(lambda: check.check(raw, view, fixed, expected_index=0,
            expected_goals=goals, **dict(args, work=expired)), "deadline")
        require(expired.counts == work.counts, "expired slice receipt did no traversal")
        result["status"] = "PASS"
        result["worker_reception"] = receive(result, work)
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    result.update(work_counts=dict(work.counts), elapsed_before_worker_export=time.monotonic()-start)
    return result
