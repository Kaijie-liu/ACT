"""Finite A2 comparison on one unchanged synthetic shared HZ, not a model run.

Only run() imports native libraries. Reception is portable standard library.
No new source tuning, optimizer, checker, or supervision policy is introduced.
"""
import hashlib
import math
import copy
import time
from fractions import Fraction as F
from pathlib import Path

from scoped_proof.io import load
from scoped_source.hz_request_store import compact, decode, digest, require
from scoped_source.development_work import DevelopmentWork, POLICY
from scoped_source import check_margin_hz_source as source
from scoped_source import check_native_margin_support as scalar
from scoped_source import check_margin_slices as slices
from scoped_source import check_margin_product as product
from margin_native_cases import fixtures, PROPERTIES, rejected
from analyze_margin_slice_transfer_20261021 import transfer

SCHEMA = "MARGIN_A2_FINITE_COMPARISON_V1"
ARMS = ("zero", "finite", "transfer", "endpoint_2", "N1_2", "M3_2",
        "endpoint_4", "N1_4", "M3_4")
SLOTS = dict(zero=2, finite=4, transfer=0, endpoint_2=4, N1_2=10,
             M3_2=10, endpoint_4=8, N1_4=20, M3_4=20)
TOTAL_CALLS = 78
GLOBAL_CELLS = 4
QUERY_SECONDS = 1.0
EFFECT = F(1, 1024)
COST_RATIO = .8
PROPERTY = PROPERTIES[0]
SCOPE = {k: hashlib.sha256(("A2_STRONG_REFERENCE_20261021_"+k).encode()).hexdigest()
         for k in ("request", "domain", "guard", "model", "input")}
ARM_LIMIT = 2*2**20
FAULTS = {"negative_duration": "valid duration", "nonfinite_duration": "valid duration",
          "wrong_native_lp": "original source/objective/cell/native event binding",
          "missing_proposal_group": "all native proposal groups"}


def duration(start, finish, deadline, label):
    require(all(type(v) in (int, float) and math.isfinite(v) for v in (start, finish, deadline))
            and 0 <= finish-start <= 300 and finish <= deadline, "valid duration: "+label)
    return finish-start


def cuts_for(count):
    require(count in (2, 4), "frozen two/four-cell grids")
    lo, hi = F(-5, 16), F(21, 16)
    return tuple(lo+(hi-lo)*F(i, count) for i in range(1, count))


def subset(record, goals):
    wanted = [[kind, str(value)] for kind, value in goals]
    by_goal = {(s["goal"][0], s["goal"][1]): s for s in record["supports"]}
    return dict(record, supports=[by_goal[tuple(g)] for g in wanted])


def check_arm(raw, view, common, name, artifact, work, finite=None):
    require(artifact["name"] == name, "fixed A2 arm identity")
    kw = dict(expected_source=common["source_sha256"], expected_property=common["property_sha256"], work=work)
    if name in ("zero", "finite"):
        mode = "zero" if name == "zero" else "median"
        checked = scalar.check(raw, view, artifact["record"], expected_mode=mode,
                               expected_cells=GLOBAL_CELLS, **kw)
        if name == "finite":
            matched = scalar.check(raw, view, artifact["matched_zero"], expected_mode="zero",
                                   expected_cells=GLOBAL_CELLS, **kw)
            require([s for s in artifact["record"]["supports"] if F(s["nu"]) == 0]
                    == artifact["matched_zero"]["supports"], "same common zero certificates")
            require(F(checked["lower"]) >= F(matched["lower"]), "same-candidate finite monotonicity")
            checked = dict(checked, matched_zero_lower=matched["lower"])
        return checked
    if name == "transfer":
        require(finite is not None and artifact["origin_sha256"] == digest(compact(finite)),
                "priced transfer originating artifact")
        parsed = source.inspect_property(raw, view, **kw)
        value = transfer(parsed, finite["record"], work)
        require(value == artifact["transfer"], "complete transferred-reference reconstruction")
        return dict(lower=value["transferred_lower"], grade=value["grade"], complete_request=False,
                    originating_calls=4, additional_calls=0)
    mode, text_count = name.rsplit("_", 1); count = int(text_count); cuts = cuts_for(count)
    if mode == "endpoint":
        return slices.aggregate(raw, view, artifact["records"], cuts=cuts, differences=False, **kw)
    require(mode in ("N1", "M3") and len(artifact["cells"]) == count, "all product cells")
    parsed, plan, _ = slices.prepare(raw, view, cuts=cuts, **kw)
    cell_results = []
    for index, (cell, saved) in enumerate(zip(plan["cells"], artifact["cells"])):
        require(saved["index"] == index and type(saved["index"]) is int, "ordered A2 cell")
        endpoints = slices.endpoint_goals(cell, work)
        ep = slices.check(raw, view, saved["endpoints"], cuts=cuts, expected_index=index,
                          expected_goals=endpoints, **kw)
        endpoint_lower = min(F(v["lower"]) for v in ep["results"])
        # Same-cell signed ranges must still be independently checked even
        # when the ensuing product solve was replaced by a valid endpoint.
        _, _, _, (lo, hi), _ = product.premises(raw, view, saved["ranges"], cuts=cuts,
                                              index=index, mode=mode, **kw)
        gl, gh = map(F, cell["gate"])
        selected = gl if lo >= 0 or gl == gh else gh if hi <= 0 else None
        if mode == "N1" and selected is not None:
            require(saved["kind"] == "ENDPOINT_REUSE" and saved["product"] is None,
                    "only proven N1 endpoint reuse")
            lower = next(F(r["lower"]) for r in ep["results"] if F(r["goal"][1]) == selected)
        else:
            require(saved["kind"] == "PRODUCT", "required product obligation")
            p = product.check(raw, view, saved["product"], cuts=cuts, index=index, mode=mode, **kw)
            lower = F(p["lower"])
        cell_results.append(dict(index=index, product_or_reused_lower=str(lower),
                                 endpoint_lower=str(endpoint_lower), lower=str(max(lower, endpoint_lower))))
    lower = min(F(r["lower"]) for r in cell_results)
    return dict(lower=str(lower), cells=cell_results, complete_request=False,
                grade="CHECKED_GIVEN_HZ_PRODUCT_WITH_PAID_ENDPOINT_FALLBACK")


def check_events(value):
    events = value["events"]
    require(len(events) % 2 == 0 and len(events) <= 2*TOTAL_CALLS, "bounded complete native events")
    counts = {name: 0 for name in ARMS}
    for j in range(0, len(events), 2):
        intent, returned = events[j:j+2]
        require(intent["event"] == "INTENT" and returned["event"] == "RETURN"
                and intent["call"] == returned["call"] == j//2+1
                and intent["arm"] == returned["arm"] in counts
                and intent["lp_sha256"] == returned["lp_sha256"]
                and intent["objective_index"] == returned["objective_index"]
                and intent["provider"] == returned["provider"] == "HIGHS_RAW_MARGINALS_V1"
                and intent["deadline"] == value["deadline"]
                and type(returned["status"]) is int and returned["status"] == 0
                and returned["success"] is True, "complete successful numerical proposals")
        seconds = duration(returned["started_at"], returned["finished_at"], value["deadline"], "native")
        require(returned["seconds"] == seconds and 0 < returned["time_limit"] <= QUERY_SECONDS,
                "native allowance and duration")
        counts[intent["arm"]] += 1
    for name in ARMS:
        require(counts[name] <= SLOTS[name], "no hidden extra native query")
        if not name.startswith("N1_"):
            require(counts[name] == SLOTS[name], "complete mandatory objective roster")
        else:
            # Only the independently checked endpoint reuse can eliminate one
            # product query per cell; the full±d and endpoint queries remain.
            require(counts[name] >= 4*int(name[-1]), "full paid range and endpoint roster")
    require(sum(counts.values()) == value["native_calls_returned"] == value["native_calls_intended"]
            and value["pending_call"] is None, "exact native totals, no pending work")
    return counts


def check_queries(raw, view, common, name, artifact, events, work):
    """Bind the actually recorded native objectives to the checked source/cells.

    This is execution-roster auditing, not another bound acceptance rule.
    Product LP identity has already been reconstructed by product.check().
    """
    parsed = source.inspect_property(raw, view, expected_source=common["source_sha256"],
                                     expected_property=common["property_sha256"], work=work)
    hz = parsed["joint"]; n = hz["nc"]+hz["nb"]
    def matrix(ck, bk):
        ids, data, ptr = [], [], [0]
        for cr, br in zip(hz[ck], hz[bk]):
            work.visit("row_visits")
            row = dict(cr); row.update({n0+hz["nc"]: v for n0, v in br.items()})
            for j, v in sorted(row.items()):
                work.visit("coefficient_visits"); ids.append(j); data.append(str(v))
            ptr.append(len(ids))
        return dict(shape=[len(ptr)-1, n], data=data, indices=ids, indptr=ptr)
    base = dict(matrix_format="csr_v1", A=matrix("Auc", "Aub"), E=matrix("Ac", "Ab"),
                b=list(map(str, hz["ub"])), h=list(map(str, hz["b"])), lower=[-1]*n, upper=[1]*n)
    def identities(goals, cell=None):
        lpbase = dict(base)
        if cell is not None:
            a = {k: list(v) for k, v in base["A"].items()}
            for sign in (1, -1):
                work.visit("row_visits")
                for j, v in sorted(parsed["rows"]["m"][1].items()):
                    work.visit("coefficient_visits")
                    a["indices"].append(j); a["data"].append(str(work.exact(sign*v)))
                a["indptr"].append(len(a["data"]))
            a["shape"][0] += 2
            lpbase.update(A=a, b=base["b"]+cell["slab_rhs"])
        answer = []
        for spec in goals:
            center, row = (source.objective(parsed, *spec, work) if cell is None
                           else slices.objective(parsed, spec, work))
            work.visit("coefficient_visits", n)
            lp = dict(lpbase, c=[str(row.get(j, F(0))) for j in range(n)], offset=str(center))
            answer.append(digest(compact(lp)))
            work.tick()
        return answer
    expected = []
    if name in ("zero", "finite"):
        expected.append(identities(tuple((F(v["t"]), F(v["nu"])) for v in artifact["record"]["supports"])))
    elif name != "transfer":
        mode, count = name.rsplit("_", 1)
        plan, _ = slices.partition(parsed, cuts_for(int(count)), work)
        for index, cell in enumerate(plan["cells"]):
            goals = slices.endpoint_goals(cell, work)
            if mode != "endpoint": goals = product.RANGE_GOALS+goals
            expected.append(identities(goals, cell))
            if mode != "endpoint" and artifact["cells"][index]["kind"] == "PRODUCT":
                expected.append([artifact["cells"][index]["product"]["certificate"]["lp_sha256"]])
    groups = artifact["proposal_groups"]
    require(len(groups) == len(expected), "all native proposal groups")
    returns = [e for e in events if e["arm"] == name and e["event"] == "RETURN"]
    position = 0
    for group, hashes in zip(groups, expected):
        report = group["native_report"]
        require(report["provider"] == "HIGHS_RAW_MARGINALS_V1"
                and report["native_candidate_calls"] == report["objective_slots"]
                == report["status0_returns"] == len(hashes)
                and len(report["outcomes"]) == len(group["selected_candidates"]) == len(hashes),
                "strong native complete group")
        for index, (identity, outcome, selected) in enumerate(zip(hashes, report["outcomes"], group["selected_candidates"])):
            event = returns[position]; position += 1
            require(identity == outcome["lp_sha256"] == event["lp_sha256"]
                    and event["objective_index"] == index
                    and all(event[k] == v for k, v in outcome.items())
                    and selected in ("NATIVE_DUAL", "CHECKED_ZERO_DUAL_STRONGER"),
                    "original source/objective/cell/native event binding")
    require(position == len(returns), "no unbound native return")


def receive(root, value, work):
    root = Path(root)
    require(value["schema"] == SCHEMA and value["status"] == "PASS" and value["policy"] == POLICY
            and value["deadline"] == work.deadline, "complete A2 worker and continued deadline")
    duration(value["worker_entry"], value["worker_finished_before_export"], value["deadline"], "worker")
    require(value["elapsed_before_worker_export"] == value["worker_finished_before_export"]-value["worker_entry"],
            "worker timestamp subtraction")
    require(sorted(value["arms"]) == sorted(ARMS), "fixed complete A2 arm roster")
    require(set(value["faults"]) == set(FAULTS) and all(
        value["faults"][k]["type"] == "ValueError" and text in value["faults"][k]["reason"]
        for k, text in FAULTS.items()), "new A2 accounting rejection controls")
    counts = check_events(value)
    ref = value["common"]
    require(ref["file"] == "common.json", "common file name")
    common = load(root/ref["file"], ref["sha256"], 2**20)
    require((root/ref["file"]).stat().st_size == ref["bytes"], "common file bytes")
    raw, view = compact(common["source"]), compact(common["view"])
    require(digest(raw) == common["source_sha256"] and digest(view) == common["property_sha256"], "bound source/view")
    base = common["source"]["base"]
    require(common["source"]["scope"] == SCOPE and base["context"]["pair"] == [0, 2]
            and all(base["sources"][k] == v for k, v in fixtures().items())
            and base["router_eq_rows"] == base["router_le_rows"] == [1], "unchanged synthetic source")
    require(common["view"]["q"] == PROPERTY["q"] and common["view"]["constant"] == PROPERTY["constant"],
            "unchanged ten-class property")
    artifacts, receipts, timing = {}, {}, {}
    for name in ARMS:
        begun = time.monotonic(); item = value["arms"][name]
        require(item["file"] == "arm_"+name+".json", "fixed arm file")
        duration(item["started_at"], item["finished_at"], value["deadline"], "arm "+name)
        require(item["seconds"] == item["finished_at"]-item["started_at"], "arm timestamp subtraction")
        artifact = load(root/item["file"], item["sha256"], ARM_LIMIT)
        require((root/item["file"]).stat().st_size == item["bytes"], "arm bytes")
        receipt = check_arm(raw, view, common, name, artifact, work, artifacts.get("finite"))
        require(receipt == artifact["receipt"], "independent full arm receipt")
        check_queries(raw, view, common, name, artifact, value["events"], work)
        expected = SLOTS[name]
        if name.startswith("N1_"):
            expected -= sum(c["kind"] == "ENDPOINT_REUSE" for c in artifact["cells"])
        require(counts[name] == expected, "actual calls agree with checked reuse")
        artifacts[name], receipts[name] = artifact, receipt
        timing[name] = duration(begun, time.monotonic(), work.deadline, "reception "+name)
    work.tick()
    return dict(receipts=receipts, arm_seconds=timing, native_calls=counts,
                work_counts=dict(work.counts), complete_request=False)


def decision(receipts, costs):
    zero = F(receipts["zero"]["lower"]); finite = F(receipts["finite"]["lower"])
    matched = F(receipts["finite"]["matched_zero_lower"])
    references = tuple(n for n in ARMS if n not in ("zero", "finite"))
    eligible = [n for n in references if F(receipts[n]["lower"]) >= finite-EFFECT]
    dominant = [n for n in references if F(receipts[n]["lower"]) >= finite and costs[n] <= costs["finite"]]
    gain = min(finite-zero, finite-matched)
    if gain < EFFECT or dominant:
        action = "STOP_CURRENT_FIXED_TILT_MECHANISM"
    elif eligible and costs["finite"] <= COST_RATIO*min(costs[n] for n in eligible):
        action = "LIMITED_MECHANISM_SIGNAL_NOT_G1_OR_REAL_REQUEST_SUCCESS"
    else:
        action = "GRAY_NO_AUTOMATIC_EXPANSION"
    return dict(action=action, lower_gain=str(gain), minimum_gain=str(EFFECT), cost_ratio=COST_RATIO,
                target=str(finite-EFFECT), eligible_references=eligible, dominating_references=dominant,
                population_inference=False, complete_request=False)


def run(deadline, publish):
    worker_entry = time.monotonic(); work = DevelopmentWork(deadline, policy=POLICY)
    value = dict(schema=SCHEMA, status="FAIL", policy=POLICY, deadline=deadline, worker_entry=worker_entry,
                 arms={}, events=[], faults={}, native_calls_intended=0, native_calls_returned=0,
                 pending_call=None, complete_request=False)
    try:
        value["faults"]["negative_duration"] = rejected(lambda: duration(1., 0., deadline, "fault"), FAULTS["negative_duration"])
        value["faults"]["nonfinite_duration"] = rejected(lambda: duration(0., float("nan"), deadline, "fault"), FAULTS["nonfinite_duration"])
        import importlib.metadata
        import numpy as np
        import scipy.sparse as sp
        import scipy.optimize
        import torch
        from act.back_end.solver.solver_hz import SparseHZono
        from act.back_end.moe.margin_hz_source import bind_source, propose_supports
        from act.back_end.moe.native_margin_support import build
        from act.back_end.moe import margin_slice_support as slice_producer
        from act.back_end.moe import margin_product_support as product_producer
        torch.set_num_threads(1)
        value["versions"] = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch")}
        def native(obj):
            args = dict(obj)
            for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                m = args[key]
                args[key] = sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=m["shape"], dtype=np.float64)
            return SparseHZono(**args)
        live = {k: native(v) for k, v in fixtures().items()}
        bound = bind_source(entry=live["entry"], expert_a=live["a"], expert_b=live["b"], router=live["router"],
                            pair=(0, 2), scope=SCOPE, router_eq_rows=(1,), router_le_rows=(1,), work=work)
        view, pid = source.make_property(bound.payload, expected_source=bound.sha256,
                                        q=PROPERTY["q"], constant=PROPERTY["constant"], work=work)
        common = dict(source=decode(bound.payload), source_sha256=bound.sha256, view=decode(view), property_sha256=pid)
        value["common"] = dict(file="common.json", **publish("common.json", common, 2**20))
        value["common_prelude_seconds"] = duration(worker_entry, time.monotonic(), deadline, "prelude")
        artifacts = {}
        for name in ARMS:
            arm_started_at = time.monotonic()
            def observed(item):
                if item["event"] == "INTENT":
                    require(value["pending_call"] is None and value["native_calls_intended"] < TOTAL_CALLS, "finite call intent")
                    value["native_calls_intended"] += 1
                    value["pending_call"] = value["native_calls_intended"]
                require(value["pending_call"] is not None, "bound call return")
                event = dict(item, arm=name, call=value["pending_call"])
                value["events"].append(event)
                publish("event_%03d.json" % len(value["events"]), event, 4096)
                if item["event"] != "INTENT":
                    value["native_calls_returned"] += item["event"] == "RETURN"
                    value["pending_call"] = None
            common_kwargs = dict(expected_property=pid, work=work, provider="highs",
                                 query_seconds=QUERY_SECONDS, observe=observed)
            artifact = dict(name=name, proposal_groups=[])
            def keep(generated):
                artifact["proposal_groups"].append({k: generated[k] for k in ("native_report", "selected_candidates")})
                return generated
            if name in ("zero", "finite"):
                mode = "zero" if name == "zero" else "median"
                plan, _, goals, _ = scalar.prepare(bound.payload, view, expected_source=bound.sha256,
                    expected_property=pid, mode=mode, cells=GLOBAL_CELLS, work=work)
                generated = keep(propose_supports(bound, view, goals=goals, **common_kwargs))
                artifact["record"] = build(plan, generated["candidates"], work)
                if name == "finite":
                    zero_plan, _, _, _ = scalar.prepare(bound.payload, view, expected_source=bound.sha256,
                        expected_property=pid, mode="zero", cells=GLOBAL_CELLS, work=work)
                    artifact["matched_zero"] = build(zero_plan, [s for s in generated["candidates"] if F(s["nu"]) == 0], work)
            elif name == "transfer":
                parsed = source.inspect_property(bound.payload, view, expected_source=bound.sha256,
                                                  expected_property=pid, work=work)
                artifact.update(origin_sha256=digest(compact(artifacts["finite"])),
                                transfer=transfer(parsed, artifacts["finite"]["record"], work))
            else:
                mode, count_text = name.rsplit("_", 1); count = int(count_text); cuts = cuts_for(count)
                parsed, plan, _ = slices.prepare(bound.payload, view, expected_source=bound.sha256,
                                                 expected_property=pid, cuts=cuts, work=work)
                if mode == "endpoint":
                    records = []
                    for index, cell in enumerate(plan["cells"]):
                        generated = keep(slice_producer.propose(bound, view, cuts=cuts, index=index,
                            goals=slices.endpoint_goals(cell, work), **common_kwargs))
                        records.append(generated["record"])
                    artifact["records"] = records
                else:
                    cells = []
                    for index, cell in enumerate(plan["cells"]):
                        endpoints = slices.endpoint_goals(cell, work)
                        combined = keep(slice_producer.propose(bound, view, cuts=cuts, index=index,
                            goals=product.RANGE_GOALS+endpoints, **common_kwargs))["record"]
                        ranges, ep = subset(combined, product.RANGE_GOALS), subset(combined, endpoints)
                        _, _, _, (lo, hi), _ = product.premises(bound.payload, view, ranges,
                            expected_source=bound.sha256, expected_property=pid, cuts=cuts, index=index, mode=mode, work=work)
                        gl, gh = map(F, cell["gate"])
                        reuse = mode == "N1" and (lo >= 0 or hi <= 0 or gl == gh)
                        record = None if reuse else keep(product_producer.propose(bound, view, ranges,
                            cuts=cuts, index=index, mode=mode, **common_kwargs))["record"]
                        cells.append(dict(index=index, ranges=ranges, endpoints=ep,
                                          kind="ENDPOINT_REUSE" if reuse else "PRODUCT", product=record))
                    artifact["cells"] = cells
            artifact["receipt"] = check_arm(bound.payload, view, common, name, artifact, work, artifacts.get("finite"))
            check_queries(bound.payload, view, common, name, artifact, value["events"], work)
            ref = publish("arm_"+name+".json", artifact, ARM_LIMIT)
            finished = time.monotonic()
            value["arms"][name] = dict(file="arm_"+name+".json", **ref, started_at=arm_started_at,
                                      finished_at=finished, seconds=duration(arm_started_at, finished, deadline, name))
            artifacts[name] = artifact
            publish("progress_%02d.json" % len(value["arms"]),
                    dict(arms=value["arms"], native_calls=value["native_calls_returned"], work_counts=dict(work.counts)), 65536)
        check_events(value)
        wrong = copy.deepcopy(value["events"])
        wrong[0]["lp_sha256"] = wrong[1]["lp_sha256"] = "0"*64
        value["faults"]["wrong_native_lp"] = rejected(lambda: check_queries(
            bound.payload, view, common, "zero", artifacts["zero"], wrong, work), FAULTS["wrong_native_lp"])
        missing = dict(artifacts["zero"], proposal_groups=[])
        value["faults"]["missing_proposal_group"] = rejected(lambda: check_queries(
            bound.payload, view, common, "zero", missing, value["events"], work), FAULTS["missing_proposal_group"])
        value["status"] = "PASS"
    except BaseException as exc:
        value.update(status="FAIL", error=repr(exc))
    worker_finished = time.monotonic()
    value.update(work_counts=dict(work.counts), worker_finished_before_export=worker_finished,
                 elapsed_before_worker_export=duration(worker_entry, worker_finished, deadline, "worker final"))
    return value
