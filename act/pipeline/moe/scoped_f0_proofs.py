"""Request-local property facts from guarded Tier-1 output intervals.

Only the exact implication pair membership => each member's top-2 membership
is supported. No arbitrary subset query, cross-request cache, solver status
promotion or abstract-variable reuse is allowed.
"""
import math

from act.back_end.solver.lp_certificate import check, identity, rational
from act.back_end.solver.solver_hz import hz_outward_slack


def make_scope(request_id, request_identity, frame_id, numerical_policy):
    if frame_id is None:
        raise ValueError("reuse requires an identified router frame")
    return {"request_id": request_id, "model_state": request_identity["model_state"],
            "lower": request_identity["lower"], "upper": request_identity["upper"],
            "property": request_identity["property"], "frame_id": str(frame_id),
            "numerical_policy": numerical_policy,
            "gate": "selected_softmax_top2", "tie_policy": "ANY_LEGAL_TOPK"}


def facts_from_branches(branches, scope):
    facts = {}
    classes = scope["property"]["classes"]
    y = scope["property"]["clean_prediction"]
    tolerance = scope["numerical_policy"]["safe_positive_margin"]
    seen = set()
    for branch in branches:
        expert = branch["candidate"]
        if expert in seen:
            raise ValueError("duplicate expert source")
        seen.add(expert)
        source = branch.get("proof_output_bounds")
        if source is None:
            continue
        lo, hi = source["lower"], source["upper"]
        if len(lo) != classes or len(hi) != classes or any(
            not math.isfinite(v) for v in lo + hi
        ) or any(a > b for a, b in zip(lo, hi)):
            raise ValueError("invalid guarded interval source")
        for index, competitor in enumerate(j for j in range(classes) if j != y):
            # Existing endpoints are the trusted HZ propagation boundary.
            # Arithmetic from those endpoints is independently exact-checked.
            exact = rational(lo[y]) - rational(hi[competitor])
            value = math.nextafter(float(exact) - hz_outward_slack(lo[y], hi[competitor]), -math.inf)
            if not math.isfinite(value) or value <= tolerance:
                continue
            lp = {"c": [1, -1], "lower": [lo[y], lo[competitor]],
                  "upper": [hi[y], hi[competitor]]}
            certificate = {"lp_sha256": identity(lp), "inequality_dual": [],
                           "equality_dual": [], "claimed_lower_bound": value}
            check(lp, certificate)
            facts[(expert, index)] = {
                "scope": scope, "expert": expert, "property_index": index,
                "competitor": competitor, "guard_kind": "TOP2_MEMBERSHIP",
                "lower_bound": value, "source_interval": source,
                "interval_lp": lp, "interval_certificate": certificate,
                "source_kind": "TIER1_GUARDED_OUTPUT_INTERVAL",
            }
    return facts


def reuse_property(facts, scope, pair, index):
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("two distinct experts required")
    sources = []
    for expert in pair:
        fact = facts.get((expert, index))
        if fact is None:
            return None
        if (fact["scope"] != scope or fact["expert"] != expert or
                fact["property_index"] != index or fact["guard_kind"] != "TOP2_MEMBERSHIP"):
            raise ValueError("proof scope/property/guard mismatch")
        check(fact["interval_lp"], fact["interval_certificate"])
        if rational(fact["lower_bound"]) != rational(fact["interval_certificate"]["claimed_lower_bound"]):
            raise ValueError("fact bound mismatch")
        sources.append(fact)
    bound = min(s["lower_bound"] for s in sources)
    if bound <= scope["numerical_policy"]["safe_positive_margin"]:
        raise ValueError("nonpositive reuse bound")
    return {"property_index": index, "status": "SAFE", "reason": "SAFE_REUSED_TIER1_INTERVAL",
            "accepted_minimum": bound, "solver_status": None,
            "solver_bound_kind": "scoped_tier1_interval",
            "full_model_witness_valid": False, "solver_seconds": 0.0,
            "proof_sources": sources,
            "containment_rule": "TOP2_SET_IMPLIES_EACH_MEMBER_TOP2_MEMBERSHIP"}


def audit_reused_property(row, pair, evidence):
    """Reconstruct sources from Tier 1, not merely trust the row's references."""
    reuse = evidence["proof_reuse"]
    if not reuse["enabled"]:
        raise ValueError("unregistered reuse")
    scope = make_scope(evidence["request_id"], evidence["identity"],
                       reuse["frame_id"], evidence["numerical_safety"])
    facts = facts_from_branches(evidence["tier1"]["branches"], scope)
    expected = reuse_property(facts, scope, pair, row["property_index"])
    if expected is None or row != expected:
        raise ValueError("reused row differs from source reconstruction")
