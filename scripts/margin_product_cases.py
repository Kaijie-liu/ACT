"""Finite new direct-product interface roster, not an algorithm performance table."""
import copy
import hashlib
import time
from fractions import Fraction as F

from scoped_source.development_work import DevelopmentWork, POLICY
from scoped_source.hz_request_store import compact, decode, digest, require
from scoped_source.check_gate_margin_rows import freeze, MARGIN_PROFILE
from scoped_source import check_margin_hz_source as source_check
from scoped_source import check_margin_slices as slices
from scoped_source import check_margin_product as check
from scoped_source.check_native_margin_support import box
from margin_native_cases import fixtures, PROPERTIES, csr, rejected
from margin_slice_cases import direct_lower as direct_range_lower

CUTS = (F(0),)
MODES = ("N1", "M3")
NAMES = ("native", "fixed_gate", "zero_difference", "positive_difference", "negative_difference", "cross_zero")
SCOPE = {k: hashlib.sha256(("PRODUCT_INTERFACE_20261021_"+k).encode()).hexdigest()
         for k in ("request", "domain", "guard", "model", "input")}
FAULTS = ("mode", "source", "property", "cell", "difference", "sign", "bool_sign", "range_order",
          "range_bound", "range_cell", "missing_range", "mc_sign", "mc_offset", "lambda_box", "w_box",
          "missing_mc", "m3_sign", "certificate_id", "missing_dual", "positive_dual", "lower",
          "missing_cell", "reordered_cell")
REASONS = ("product record identity/mode/grade",)*3+("product domain/range/sign binding",)*4+(
    "slice objective binding", "slice exact residual lower mismatch", "caller-bound closed slice",
    "complete slice supports")+("product exact LP construction",)*6+(
    "product LP certificate identity", "product complete dual rows", "product inequality dual sign",
    "product exact finite-box residual", "complete product partition", "caller-bound closed slice")


def originals(name):
    value = fixtures()
    if name == "fixed_gate":
        r = value["router"]
        r["Gc"] = csr(3, r["Gc"]["shape"][1], []); r["Gb"] = csr(3, r["Gb"]["shape"][1], [])
        r["c"][2] = r["c"][0]
    if name in ("positive_difference", "negative_difference"):
        for key in ("a", "b"):
            for matrix in ("Gc", "Gb"):
                value[key][matrix] = csr(10, value[key][matrix]["shape"][1], [])
    return value


def prop(name):
    p = copy.deepcopy(PROPERTIES[0])
    if name == "zero_difference": p["q"] = ["0"]*10
    if name == "negative_difference": p["q"] = [str(-F(v)) for v in p["q"]]
    return p


def cuts(name):
    return () if name in ("fixed_gate", "cross_zero") else CUTS


def direct_product_lower(lp, cert, work):
    """Second finite-box residual implementation, not the product receiver."""
    coefficients = list(map(F, lp["c"]))
    constant = F(lp["offset"])
    for matrix, rhs, dual in ((lp["A"], lp["b"], cert["y"]), (lp["E"], lp["h"], cert["z"])):
        require(len(dual) == len(rhs), "direct product rows")
        for i, multiplier in enumerate(map(F, dual)):
            work.visit("row_visits"); constant = work.exact(constant+work.exact(multiplier*F(rhs[i])))
            for k in range(matrix["indptr"][i], matrix["indptr"][i+1]):
                work.visit("coefficient_visits"); j = matrix["indices"][k]
                coefficients[j] = work.exact(coefficients[j]-work.exact(multiplier*F(matrix["data"][k])))
    for coefficient, lo, hi in zip(coefficients, map(F, lp["lower"]), map(F, lp["upper"])):
        work.visit("coefficient_visits")
        constant = work.exact(constant+min(work.exact(coefficient*lo), work.exact(coefficient*hi)))
    return constant


def receive(value, work):
    require(value["status"] == "PASS" and value["policy"] == POLICY, "successful product worker")
    require(value["full_calls_attempted"] == value["native_candidate_calls"] == value["objective_slots"] == 4,
            "fixed product call roster")
    require(value["pending_call"] is None, "no unfinished product call accepted")
    require(set(value["objects"]) == set(NAMES) and set(value["faults"]) == set(FAULTS)
            and value["expired_reception"]["type"] == "TimeoutError", "complete product object/fault roster")
    for name, reason in zip(FAULTS, REASONS):
        require(value["faults"][name]["type"] == "ValueError" and reason in value["faults"][name]["reason"],
                "specific product rejection")
    receipts = {}
    for name in NAMES:
        obj = value["objects"][name]; src, view = obj["source"], obj["view"]
        require(src["scope"] == dict(SCOPE, request=hashlib.sha256((SCOPE["request"]+name).encode()).hexdigest())
                and src["base"]["context"]["pair"] == [0, 2]
                and src["base"]["router_eq_rows"] == [1] and src["base"]["router_le_rows"] == [1]
                and all(src["base"]["sources"][k] == v for k, v in originals(name).items()), "frozen product source")
        p = prop(name)
        require(view["q"] == p["q"] and view["constant"] == p["constant"], "full product property")
        raw, pv = compact(src), compact(view)
        require(digest(raw) == obj["source_sha256"] and digest(pv) == obj["property_sha256"], "product source hash")
        args = dict(expected_source=obj["source_sha256"], expected_property=obj["property_sha256"],
                    cuts=cuts(name), work=work)
        require(set(obj["records"]) == (set(MODES) if name == "native" else {"M3"}), "product mode roster")
        receipts[name] = {}
        for mode, records in obj["records"].items():
            require(type(records) is list and len(records) == (2 if name == "native" else 1)
                    and all(r["native_candidate_calls"] == (1 if name == "native" else 0)
                            and (name != "native" or r["objective_slots"] == 1) for r in records),
                    "exact product record/call roster")
            if name == "native":
                receipt = check.aggregate(raw, pv, [r["record"] for r in records], mode=mode, **args)
            else:
                receipt = check.check(raw, pv, records[0]["record"], index=0, mode=mode, **args)
            require(receipt == obj["receipts"][mode], "independent product receipt")
            for item in records:
                r = item["record"]
                require(direct_product_lower(r["lp"], r["certificate"], work) == F(r["certificate"]["lower"]),
                        "direct finite-box differential")
                for support in r["ranges"]["supports"]:
                    require(direct_range_lower(src, view, r["cell"], support, work) == F(support["lower"]),
                            "direct product range differential")
            receipts[name][mode] = receipt
    return receipts


def run(deadline, publish):
    work = DevelopmentWork(deadline, policy=POLICY); start = time.monotonic()
    result = dict(status="FAIL", policy=POLICY, deadline=deadline, objects={}, records={}, faults={},
                  native_candidate_calls=0, full_calls_attempted=0, objective_slots=0,
                  pending_call=None, complete_request=False)
    try:
        import importlib.metadata
        import numpy as np
        import scipy.sparse as sp
        import torch
        from act.back_end.solver.solver_hz import SparseHZono
        from act.back_end.moe.margin_hz_source import bind_source
        from act.back_end.moe.margin_product_support import propose, build, proposed_lower
        torch.set_num_threads(1)
        result["native_import_seconds"] = time.monotonic()-start
        result["versions"] = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch")}
        def native(obj):
            args = dict(obj)
            for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                m = args[key]
                args[key] = sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=m["shape"], dtype=np.float64)
            return SparseHZono(**args)
        for name in NAMES:
            begun = time.monotonic(); live = {k: native(v) for k, v in originals(name).items()}
            scope = dict(SCOPE, request=hashlib.sha256((SCOPE["request"]+name).encode()).hexdigest())
            bound = bind_source(entry=live["entry"], expert_a=live["a"], expert_b=live["b"], router=live["router"],
                                pair=(0, 2), scope=scope, router_eq_rows=(1,), router_le_rows=(1,), work=work)
            pv, pid = source_check.make_property(bound.payload, expected_source=bound.sha256, **prop(name), work=work)
            args = dict(expected_source=bound.sha256, expected_property=pid, cuts=cuts(name), work=work)
            parsed, plan, identity = slices.prepare(bound.payload, pv, **args)
            obj = dict(source=decode(bound.payload), view=decode(pv), source_sha256=bound.sha256,
                       property_sha256=pid, records={}, receipts={})
            result["objects"][name] = obj
            for mode in MODES if name == "native" else ("M3",):
                records = []; obj["records"][mode] = records
                for index in range(2 if name == "native" else 1):
                    cell = plan["cells"][index]; supports = []
                    for goal in check.RANGE_GOALS:
                        row = slices.objective(parsed, goal, work); lower = box(row, work)[0]
                        supports.append(dict(goal=[goal[0], str(goal[1])], y=["0"]*len(parsed["joint"]["ub"]),
                            z=["0"]*len(parsed["joint"]["b"]), slice_duals=["0", "0"], lower=str(lower)))
                    ranges = dict(schema=slices.CERT, source=bound.sha256, property=pid, partition=identity,
                                  cell=cell, supports=supports, grade=slices.GRADE)
                    if name == "native":
                        result["full_calls_attempted"] += 1
                        result["pending_call"] = dict(object=name, mode=mode, cell=index,
                                                      native_execution_state="NOT_YET_CONFIRMED_RETURNED")
                        publish(result, work)
                        item = propose(bound, pv, ranges, expected_property=pid, cuts=cuts(name), index=index, mode=mode, work=work)
                        result["native_candidate_calls"] += item["native_candidate_calls"]
                        result["objective_slots"] += item["objective_slots"]
                        result["pending_call"] = None
                    else:
                        _, _, _, difference, sign = check.premises(bound.payload, pv, ranges, index=index, mode=mode, **args)
                        lp = build(parsed, cell, difference, sign, work)
                        _, lp_id = freeze(lp, work, profile=MARGIN_PROFILE)
                        y, z = ["0"]*len(lp["b"]), ["0"]*len(lp["h"])
                        cert = dict(lp_sha256=lp_id, y=y, z=z, lower=str(proposed_lower(lp, y, z, work)))
                        record = dict(schema=check.SCHEMA, source=bound.sha256, property=pid, partition=identity,
                            cell=cell, mode=mode, ranges=ranges, difference=list(map(str, difference)), sign=sign,
                            lp=lp, certificate=cert, grade=check.GRADE)
                        receipt = check.check(bound.payload, pv, record, index=index, mode=mode, **args)
                        item = dict(record=record, receipt=receipt, native_candidate_calls=0)
                    records.append(item)
                    if name == "native":
                        result["records"][str(len(result["records"]))] = mode+":"+str(index)
                        publish(result, work)
                obj["receipts"][mode] = (check.aggregate(bound.payload, pv, [r["record"] for r in records], mode=mode, **args)
                    if name == "native" else records[0]["receipt"])
            obj["construction_proposal_check_seconds"] = time.monotonic()-begun
        require(result["objects"]["fixed_gate"]["records"]["M3"][0]["record"]["cell"]["gate"] == ["1/2", "1/2"],
                "fixed gate product")
        for name, expected in (("zero_difference", "0"), ("positive_difference", "5/128"), ("negative_difference", "-5/128")):
            require(result["objects"][name]["records"]["M3"][0]["record"]["difference"] == [expected]*2,
                    "fixed signed difference product")
        require(result["objects"]["cross_zero"]["records"]["M3"][0]["record"]["sign"] is None,
                "crossing cell no unproved M3 rows")
        obj = result["objects"]["native"]; raw, pv = compact(obj["source"]), compact(obj["view"])
        args = dict(expected_source=obj["source_sha256"], expected_property=obj["property_sha256"], cuts=CUTS, work=work)
        original = obj["records"]["M3"][0]["record"]
        for name, reason in zip(FAULTS, REASONS):
            test_index = 1 if name == "bool_sign" else 0
            bad = copy.deepcopy(obj["records"]["M3"][test_index]["record"])
            if name == "mode": bad["mode"] = "N1"
            elif name in ("source", "property"): bad[name] = "0"*64
            elif name == "cell": bad["cell"]["index"] = 1
            elif name == "difference": bad["difference"][0] = "-9"
            elif name == "sign": bad["sign"]["side"] *= -1
            elif name == "bool_sign": bad["sign"]["side"] = True
            elif name == "range_order": bad["ranges"]["supports"].reverse()
            elif name == "range_bound": bad["ranges"]["supports"][0]["lower"] = "123"
            elif name == "range_cell": bad["ranges"]["cell"]["index"] = 1
            elif name == "missing_range": bad["ranges"]["supports"].pop()
            elif name == "mc_sign":
                start = bad["lp"]["A"]["indptr"][len(obj["source"]["augmented"]["ub"])+2]
                bad["lp"]["A"]["data"][start] = str(-F(bad["lp"]["A"]["data"][start]))
            elif name == "mc_offset": bad["lp"]["b"][-3] = "123"
            elif name == "lambda_box": bad["lp"]["lower"][-2] = "-1"
            elif name == "w_box": bad["lp"]["upper"][-1] = "1"
            elif name == "missing_mc": bad["lp"]["A"]["shape"][0] -= 1
            elif name == "m3_sign": bad["lp"]["b"][-1] = "123"
            elif name == "certificate_id": bad["certificate"]["lp_sha256"] = "0"*64
            elif name == "missing_dual": bad["certificate"]["y"].pop()
            elif name == "positive_dual": bad["certificate"]["y"][-1] = "1"
            elif name == "lower": bad["certificate"]["lower"] = "123"
            if name in ("missing_cell", "reordered_cell"):
                records = [r["record"] for r in obj["records"]["M3"]]
                records = records[:-1] if name == "missing_cell" else records[::-1]
                result["faults"][name] = rejected(lambda: check.aggregate(raw, pv, records, mode="M3", **args), reason)
            else:
                result["faults"][name] = rejected(lambda: check.check(raw, pv, bad, index=test_index, mode="M3", **args), reason)
        expired = DevelopmentWork(min(deadline, time.monotonic()+.05), policy=POLICY)
        expired.continue_from(work.counts); time.sleep(.06)
        result["expired_reception"] = rejected(lambda: check.check(raw, pv, original, index=0, mode="M3",
                                                                  **dict(args, work=expired)), "deadline")
        require(expired.counts == work.counts, "expired product reception no traversal")
        result["status"] = "PASS"; result["worker_reception"] = receive(result, work)
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    result.update(work_counts=dict(work.counts), elapsed_before_worker_export=time.monotonic()-start)
    return result
