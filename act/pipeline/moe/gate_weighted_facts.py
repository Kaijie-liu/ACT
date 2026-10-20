"""Opt-in, request-local gate-weighted interval facts. Standard library only.

Exact combination of GIVEN numerical-policy intervals, not a network proof.
Signed facts never denote SAFE. No support queries, optimizer or model loading.
"""
import copy
from fractions import Fraction
import math
import time

from scoped_source.hz_request_store import Work, POLICY, compact, identity, require


KIND = "scoped_gate_weighted_interval_v1"
GRADE = "HZ_NUMERICAL_POLICY_CONDITIONAL_ON_INTERVAL_AND_GATE_SOURCES"
RULE = "TOP2_SET_IMPLIES_EACH_MEMBER_TOP2_MEMBERSHIP"


def validate_config(config):
    enabled = config.get("f0", {}).get("gate_weighted_facts", False)
    require(type(enabled) is bool, "gate_weighted_facts must be boolean")
    if enabled:
        require(config.get("scoped_proof_reuse") is True, "gate facts require scoped reuse")
        schedule = config.get("route_complexity_schedule") or {}
        require(schedule.get("total_seconds") == 300
                and schedule.get("multi_pair_tier1_fraction") == .25,
                "gate facts require unchanged 300-second/.25 schedule")
        require(config.get("comparison_method", "staged") in {"staged", "monolithic_f0"},
                "gate facts support scheduled staged/matched paths")
    return enabled


def _number(value, work):
    work.visit("coefficient_visits")
    require(type(value) in (int, float) and math.isfinite(value), "finite numerical source")
    result = Fraction(value)
    require(max(result.numerator.bit_length(), result.denominator.bit_length())
            <= POLICY["stored_bits"], "stored rational capacity")
    return result


def _saved(value, work):
    from scoped_source.hz_request_store import rational
    work.visit("coefficient_visits")
    return rational(value)


def _copy(value, work):
    work.visit("row_visits")
    raw = compact(value)
    require(len(raw) <= POLICY["member_bytes"], "bounded fact source")
    work.visit("coefficient_visits", len(raw))
    result = copy.deepcopy(value)
    require(compact(result) == raw, "source changed during capture")
    work.tick()
    return result


def _scope(scope, work):
    work.tick()
    require(type(scope) is dict and set(scope) == {
        "request_id", "model_state", "lower", "upper", "property", "frame_id",
        "numerical_policy", "gate", "tie_policy"}, "complete fact scope")
    require(scope["gate"] == "selected_softmax_top2"
            and scope["tie_policy"] == "ANY_LEGAL_TOPK"
            and isinstance(scope["frame_id"], str) and bool(scope["frame_id"]),
            "gate/tie/frame scope")
    for field in ("request_id",):
        require(isinstance(scope[field], str) and len(scope[field]) == 64, "request identity")
    for field in ("model_state", "lower", "upper"):
        require(isinstance(scope[field], dict)
                and isinstance(scope[field].get("sha256"), str)
                and len(scope[field]["sha256"]) == 64, "source identity")
    policy = scope["numerical_policy"]
    require(policy["safe_positive_margin"] == 1e-7
            and policy["outward_absolute"] == 1e-9
            and policy["outward_relative"] == 1e-9, "unchanged numerical gates")
    classes = scope["property"]["classes"]
    require(type(classes) is int and 2 <= classes <= 4096, "property dimension")
    return classes


def _property(scope, index, q, constant, work):
    classes = _scope(scope, work)
    require(type(index) is int and index >= 0 and len(q) == classes, "property identity/shape")
    weights = [_number(v, work) for v in q]
    offset = _number(constant, work)
    return weights, offset


def _interval(source, classes, work):
    require(type(source) is dict and set(source) == {"lower", "upper"}
            and len(source["lower"]) == len(source["upper"]) == classes,
            "complete interval source")
    lower = [_number(v, work) for v in source["lower"]]
    upper = [_number(v, work) for v in source["upper"]]
    require(all(a <= b for a, b in zip(lower, upper)), "inverted expert interval")
    return lower, upper


def signed_fact(scope, expert, index, q, constant, source, work):
    """Capture a signed lower bound, never an expert-safety verdict."""
    work.tick()
    require(type(expert) is int and expert >= 0, "expert identity")
    weights, offset = _property(scope, index, q, constant, work)
    lower, upper = _interval(source, len(weights), work)
    terms = [work.exact(w * (lo if w >= 0 else hi))
             for w, lo, hi in zip(weights, lower, upper)]
    exact = offset
    for value in terms:
        exact = work.exact(exact + value)
    policy = scope["numerical_policy"]
    scale = max([abs(float(offset))] + [abs(float(v)) for v in terms])
    padded = math.nextafter(float(exact) - (
        policy["outward_absolute"] + policy["outward_relative"] * scale), -math.inf)
    require(math.isfinite(padded) and Fraction(padded) <= exact, "finite outward fact")
    result = dict(schema="SIGNED_GUARDED_INTERVAL_FACT_V1", scope=_copy(scope, work),
                  expert=expert, property_index=index, q=list(q), constant=constant,
                  guard_kind="TOP2_MEMBERSHIP", source=_copy(source, work),
                  lower_bound=padded, guarantee=GRADE)
    work.tick()
    return result


def _check_fact(fact, scope, expert, index, q, constant, source, work):
    """Independent center/radius evaluation, not the producer's vertex sum."""
    require(set(fact) == {"schema", "scope", "expert", "property_index", "q", "constant",
                         "guard_kind", "source", "lower_bound", "guarantee"}
            and fact["schema"] == "SIGNED_GUARDED_INTERVAL_FACT_V1"
            and fact["scope"] == scope and fact["expert"] == expert
            and fact["property_index"] == index and fact["q"] == list(q)
            and fact["constant"] == constant and fact["source"] == source
            and fact["guard_kind"] == "TOP2_MEMBERSHIP" and fact["guarantee"] == GRADE,
            "signed fact source/scope/property binding")
    weights, offset = _property(scope, index, q, constant, work)
    lower, upper = _interval(source, len(q), work)
    bound = offset
    selected = []
    for w, lo, hi in zip(weights, lower, upper):
        center = work.exact((lo + hi) / 2)
        radius = work.exact((hi - lo) / 2)
        bound = work.exact(bound + work.exact(w * center) - work.exact(abs(w) * radius))
        selected.append(work.exact(w * (lo if w >= 0 else hi)))
    scale = max([abs(float(offset))] + [abs(float(v)) for v in selected])
    policy = scope["numerical_policy"]
    expected = math.nextafter(float(bound) - (
        policy["outward_absolute"] + policy["outward_relative"] * scale), -math.inf)
    value = _number(fact["lower_bound"], work)
    require(fact["lower_bound"] == expected and value <= bound, "signed fact lower bound")
    return value


def supported_gate_source(support):
    return (support.get("complete_exact") is True
            and support.get("lower_status") in (["milp_optimal"], ["constant_exact"])
            and support.get("upper_status") in (["milp_optimal"], ["constant_exact"])
            and support.get("gaps") in ([0], [0.0]))


def gate_fact(scope, pair, weight_expert, bounds, margin_bounds, support, frame_id, work):
    """Normalize direction, but do not certify numerical sigmoid/propagation."""
    _scope(scope, work)
    require(len(pair) == 2 and all(type(v) is int and v >= 0 for v in pair)
            and pair[0] < pair[1] and type(weight_expert) is int and weight_expert in pair,
            "canonical gate pair/direction")
    require(str(frame_id) == scope["frame_id"], "conditioned router frame mismatch")
    lo, hi = [_number(v, work) for v in bounds]
    ml, mh = [_number(v, work) for v in margin_bounds]
    require(0 <= lo <= hi <= 1 and ml <= mh, "gate/margin interval")
    require(supported_gate_source(support),
            "gate support must satisfy original optimal-status policy")
    if weight_expert == pair[1]:
        lo, hi = work.exact(1 - hi), work.exact(1 - lo)
    return dict(schema="SCOPED_GATE_INTERVAL_V1", scope_sha256=identity(scope),
                pair=list(pair), weight_expert=weight_expert,
                raw_lambda_bounds=list(bounds), margin_bounds=list(margin_bounds),
                lambda_bounds=[str(lo), str(hi)], frame_id=str(frame_id),
                support=_copy(support, work), guarantee=GRADE,
                domain="TOP2_SET", containment_rule=RULE)


def check_row(row, *, scope, pair, index, q, constant, sources, gate, work):
    """Check from separately supplied sources; hashes alone are insufficient."""
    work.tick()
    expected_keys = {"property_index", "status", "reason", "solver_status", "solver_bound_kind",
                     "accepted_minimum", "full_model_witness_valid", "proof_sources", "gate_source",
                     "checked_lower_bound", "endpoint_lower_bounds", "guarantee", "containment_rule"}
    require(set(row) == expected_keys and row["property_index"] == index
            and row["status"] == "SAFE" and row["reason"] == "SAFE_GATE_WEIGHTED_INTERVAL"
            and row["solver_status"] is None and row["solver_bound_kind"] == KIND
            and row["full_model_witness_valid"] is False
            and row["guarantee"] == GRADE and row["containment_rule"] == RULE,
            "gate row type/status/guarantee")
    expected_gate = gate_fact(scope, pair, gate["weight_expert"], gate["raw_lambda_bounds"],
                              gate["margin_bounds"], gate["support"], gate["frame_id"], work)
    require(gate == expected_gate and row["gate_source"] == gate, "gate source binding")
    require(len(row["proof_sources"]) == 2 and len(sources) == 2, "two complete expert sources")
    a, b = [_check_fact(f, scope, e, index, q, constant, s, work)
            for f, e, s in zip(row["proof_sources"], pair, sources)]
    lo, hi = [_saved(v, work) for v in gate["lambda_bounds"]]
    # Deliberately different algebra from the proposer, preserving exact arithmetic.
    endpoints = [work.exact(b + work.exact(t * work.exact(a - b))) for t in (lo, hi)]
    exact = min(endpoints)
    require(row["endpoint_lower_bounds"] == list(map(str, endpoints))
            and _saved(row["checked_lower_bound"], work) == exact, "weighted endpoint arithmetic")
    accepted = _number(row["accepted_minimum"], work)
    require(accepted <= exact and accepted > _number(scope["numerical_policy"]["safe_positive_margin"], work),
            "strict original positive margin")
    work.tick()


def propose_row(scope, pair, index, q, constant, sources, gate, work):
    if any(source is None for source in sources):
        return None
    facts = [signed_fact(scope, e, index, q, constant, s, work)
             for e, s in zip(pair, sources)]
    require(len(facts) == 2, "two signed expert facts required")
    a, b = [Fraction(f["lower_bound"]) for f in facts]
    endpoints = []
    for text in gate["lambda_bounds"]:
        t = _saved(text, work)
        endpoints.append(work.exact(work.exact(t * a) + work.exact((1 - t) * b)))
    exact = min(endpoints)
    accepted = math.nextafter(float(exact), -math.inf)
    if not math.isfinite(accepted) or accepted <= scope["numerical_policy"]["safe_positive_margin"]:
        work.tick()
        return None
    row = dict(property_index=index, status="SAFE", reason="SAFE_GATE_WEIGHTED_INTERVAL",
               solver_status=None, solver_bound_kind=KIND, accepted_minimum=accepted,
               full_model_witness_valid=False, proof_sources=facts, gate_source=_copy(gate, work),
               checked_lower_bound=str(exact), endpoint_lower_bounds=list(map(str, endpoints)),
               guarantee=GRADE, containment_rule=RULE)
    check_row(row, scope=scope, pair=pair, index=index, q=q, constant=constant,
              sources=sources, gate=gate, work=work)
    return row


class GateFacts:
    """One cumulative request ledger. Never creates a new budget per pair."""
    def __init__(self, branches, scope, budget, *, work=None):
        require(budget is not None and budget.seconds == 300, "shared 300-second request budget")
        self.budget = budget
        if work is None:
            from act.pipeline.moe.gate_native_contract import GateWork
            work = GateWork(budget.deadline)
        require(work.deadline == budget.deadline, "gate Work/request deadline mismatch")
        self.work = work
        self.scope = _copy(scope, self.work)
        self.sources = {}
        for branch in branches:
            expert = branch["candidate"]
            require(type(expert) is int and expert >= 0 and expert not in self.sources,
                    "duplicate/invalid expert source")
            source = branch.get("proof_output_bounds")
            self.sources[expert] = None if source is None else _copy(source, self.work)
        self.gates = {}
        self.stats = dict(attempted_pair_properties=0, discharged_pair_properties=0,
                          fallback_pair_properties=0, check_seconds=0.0,
                          f0_solve_attempted=0, f0_solve_returned=0, f0_status0_returned=0)

    def record_gate(self, pair, gate_range, support, conditioned_router):
        self.budget.check("gate_fact_source")
        require(tuple(gate_range.pair) == tuple(pair), "gate object pair mismatch")
        require(gate_range.conditioned_router is conditioned_router
                and gate_range.router_frame_id == conditioned_router.frame_id
                and gate_range.router_output_width == conditioned_router.n_out,
                "gate conditioned-router identity/width mismatch")
        key = tuple(pair)
        require(key not in self.gates, "duplicate gate query/source")
        if not supported_gate_source(support):
            # The original F0 may still use its valid fast fallback. M1 does
            # not promote that range or issue another support query.
            self.gates[key] = None
        else:
            self.gates[key] = gate_fact(self.scope, pair, pair[0], list(gate_range.lambda_bounds),
                                       list(gate_range.margin_bounds), support,
                                       gate_range.router_frame_id, self.work)
        self.budget.check("gate_fact_source_complete")

    def attempt(self, pair, index, q, constant):
        started = time.monotonic()
        self.budget.check("gate_fact_begin")
        self.stats["attempted_pair_properties"] += 1
        try:
            gate = self.gates[tuple(pair)]
            row = (None if gate is None else propose_row(
                self.scope, pair, index, [float(v) for v in q], float(constant),
                [self.sources.get(e) for e in pair], gate, self.work))
            self.budget.check("gate_fact_accept")
            self.stats["discharged_pair_properties" if row is not None else
                       "fallback_pair_properties"] += 1
            return row
        finally:
            self.stats["check_seconds"] += time.monotonic() - started

    def record(self):
        return dict(enabled=True, schema="GATE_WEIGHTED_REQUEST_V1", guarantee=GRADE,
                    gates=[self.gates[k] for k in sorted(self.gates) if self.gates[k] is not None],
                    unavailable_gates=[list(k) for k in sorted(self.gates) if self.gates[k] is None],
                    **self.stats,
                    work_counts=dict(self.work.counts),
                    work_scope="M1 capture/combination/check only, not upstream HZ Work",
                    discharged_scope="generated checked facts, not necessarily saved solver calls",
                    check_seconds_scope="attempts only; capture is in total request time")


def audit_row(row, pair, evidence, work):
    require(evidence["registered_budgets"]["f0"].get("gate_weighted_facts") is True
            and evidence["proof_reuse"]["enabled"] is True, "unregistered gate facts")
    reuse = evidence["proof_reuse"]
    identity_record = evidence["identity"]
    scope = {k: identity_record[k] for k in ("model_state", "lower", "upper", "property")}
    scope.update(request_id=evidence["request_id"], frame_id=reuse["frame_id"],
                 numerical_policy=evidence["numerical_safety"],
                 gate="selected_softmax_top2", tie_policy="ANY_LEGAL_TOPK")
    require(identity(identity_record) == evidence["request_id"], "request binding")
    pair = list(pair)
    require(pair in evidence["route_coverage"]["feasible_route_sets"], "uncovered pair")
    prop = identity_record["property"]
    require(prop["kind"] == "TOP1_ROBUST", "unsupported request property")
    index = row["property_index"]
    competitors = [i for i in range(prop["classes"]) if i != prop["clean_prediction"]]
    require(type(index) is int and 0 <= index < len(competitors), "property index")
    q = [0.] * prop["classes"]
    q[prop["clean_prediction"]], q[competitors[index]] = 1., -1.
    branches = evidence["tier1"]["branches"]
    sources = []
    for expert in pair:
        found = [b["proof_output_bounds"] for b in branches
                 if b["candidate"] == expert and b.get("proof_output_bounds") is not None]
        require(len(found) == 1, "unique expert source required")
        sources.append(found[0])
    record = evidence["gate_weighted_facts"]
    require(record["enabled"] is True and record["guarantee"] == GRADE, "gate record grade")
    gates = [g for g in record["gates"] if g["pair"] == pair]
    require(len(gates) == 1, "unique request gate source required")
    check_row(row, scope=scope, pair=pair, index=index, q=q, constant=0.,
              sources=sources, gate=gates[0], work=work)


def audit_evidence(evidence, *, work=None):
    """Check new rows even when the complete request is UNKNOWN or UNSAFE."""
    rows = []
    tier2 = evidence.get("tier2", {})
    for pair in tier2.get("pairs", []):
        rows.extend((r, pair["pair"]) for r in pair.get("property_rows", [])
                    if r.get("solver_bound_kind") == KIND)
    for row in tier2.get("property_rows", []):
        rows.extend((v["proof"], v["pair"]) for v in row.get("coverage_partition", {}).get("reused", [])
                    if v["proof"].get("solver_bound_kind") == KIND)
    record = evidence.get("gate_weighted_facts")
    if not rows and record is None:
        return 0
    require(record is not None, "missing request gate-fact record")
    ledger = work if work is not None else Work(time.monotonic() + 300)
    names = ("attempted_pair_properties", "discharged_pair_properties", "fallback_pair_properties",
             "f0_solve_attempted", "f0_solve_returned", "f0_status0_returned")
    require(all(type(record.get(n)) is int and record[n] >= 0 for n in names), "invalid M1 counters")
    require(record["f0_status0_returned"] <= record["f0_solve_returned"] <= record["f0_solve_attempted"]
            and record["attempted_pair_properties"] >=
            record["discharged_pair_properties"] + record["fallback_pair_properties"]
            and record["discharged_pair_properties"] >= len(rows), "inconsistent M1 counters")
    seen = set()
    for row, pair in rows:
        key = tuple(pair), row["property_index"]
        require(key not in seen, "duplicate gate obligation")
        seen.add(key)
        audit_row(row, pair, evidence, ledger)
    return len(rows)
