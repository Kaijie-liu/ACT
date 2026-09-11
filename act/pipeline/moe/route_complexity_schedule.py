"""Opt-in route-complexity scheduling with a common, charged fact prelude.

No candidate fallback: incomplete feasibility still stops the request. Facts
are computed by the same cheap guarded propagation for both comparison arms.
"""
import math
import time

from act.back_end.moe import analyze_candidates, analyze_topk_sets, build_act_moe_program
from act.back_end.solver.solver_hz import SparseHZono
from act.config.config import HybridZConfig
from act.pipeline.moe.experiment1 import _propagate_component, _solve_output, _forward_validate
from act.pipeline.moe.request_budget import BudgetExhausted
from act.pipeline.moe.scoped_f0_proofs import facts_from_branches, make_scope
from act.front_end.specs import OutputSpec, OutKind


def validate_schedule(config):
    schedule = config.get("route_complexity_schedule")
    if schedule is None:
        return
    if set(schedule) != {"version", "total_seconds", "multi_pair_tier1_fraction"}:
        raise ValueError("unknown or missing route-complexity schedule fields")
    if schedule["version"] != 1:
        raise ValueError("unsupported route-complexity schedule")
    seconds, fraction = float(schedule["total_seconds"]), float(schedule["multi_pair_tier1_fraction"])
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("invalid schedule budget/fraction")
    if config.get("scoped_proof_reuse") is not True:
        raise ValueError("scheduled comparison requires scoped reuse on both arms")
    if config.get("comparison_method", "staged") not in {"staged", "monolithic_f0"}:
        raise ValueError("schedule supports staged and matched monolithic only")


def prepare(*, model, center, lower, upper, clean_prediction, config, budget,
            request_id, request_identity):
    """Return Tier-1-compatible record, exact context and immutable fact scope."""
    record = {"status": "UNKNOWN", "reason": "SCHEDULE_PENDING", "branches": [],
              "full_model_witness_valid": False, "candidate_experts": [],
              "feasible_route_sets": [], "schedule": {
                  "policy": "SINGLE_PAIR_DIRECT_MULTI_PAIR_STAGED_V1",
                  "selected_path": "INCOMPLETE_ROUTE_ANALYSIS",
                  "common_fact_prelude_complete": False}}
    internal = reuse = witness = None
    start = time.monotonic()
    try:
        budget.check("route_build")
        output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[clean_prediction])
        program = build_act_moe_program(model, center=center, lower=lower, upper=upper, output_spec=output_spec)
        router = _propagate_component(program.router)
        budget.check("router_propagation")
        exact = isinstance(router.output_hz, SparseHZono) and router.output_hz.exact
        query = lambda: budget.limit("candidate_or_set_query", config["candidate_query_timeout"])
        candidates = analyze_candidates(router.output_hz, model.spec.top_k,
                                        input_hz=router.input_hz, time_limit_per_expert=query,
                                        router_exact=exact)
        routes = analyze_topk_sets(router.output_hz, model.spec.top_k,
                                   time_limit_per_set=query, router_exact=exact)
        record.update(candidate_seconds=time.monotonic()-start,
                      candidate_experts=list(candidates.candidates),
                      feasible_route_sets=[list(p) for p in routes.feasible])
        budget.check("route_analysis_complete")
        if not candidates.minimal or not routes.exact or not routes.feasible:
            record.update(reason="UNKNOWN_SOLVER_LIMIT", phase="candidate_feasibility")
            return record, None, None, None
        internal = {"program": program, "router": router, "candidates": candidates,
                    "route_sets": routes, "lower": lower, "upper": upper, "output_spec": output_spec}
        path = ("MONOLITHIC_MATCHED" if config.get("comparison_method") == "monolithic_f0"
                else "SINGLE_PAIR_DIRECT" if len(routes.feasible) == 1 else "MULTI_PAIR_STAGED")
        record["schedule"].update(selected_path=path, exact_pair_count=len(routes.feasible))
        scope = make_scope(request_id, request_identity, router.output_hz.frame_id, config["numerical_safety"])
        cheap = HybridZConfig(max_input_dim=1024, guarded_support_enabled=False,
                              expert_property_solver_backend=config["tier1"]["solver"].get("backend", "scipy"))
        propagations = {}
        prelude = time.monotonic()
        branches = {b.expert: b for b in candidates.branches}
        for expert in candidates.candidates:
            budget.check("common_fact_propagation")
            p = _propagate_component(program.experts[expert], entry_hz=branches[expert].guarded_input,
                                     hybridz_config=cheap)
            budget.check("common_fact_propagation_complete")
            propagations[expert] = p
            record["branches"].append({"candidate": expert, "unknown_reason": "NOT_SOLVED",
                "proof_output_bounds": {"lower": p.output_bounds.lb.detach().cpu().reshape(-1).tolist(),
                                        "upper": p.output_bounds.ub.detach().cpu().reshape(-1).tolist()},
                "source_policy": "COMMON_GUARDED_INTERVAL_NO_SUPPORT_SOLVES"})
        reuse = {"scope": scope, "facts": facts_from_branches(record["branches"], scope)}
        budget.check("common_fact_extraction_complete")
        record["schedule"].update(common_fact_prelude_complete=True,
                                  common_fact_seconds=time.monotonic()-prelude,
                                  common_fact_count=len(reuse["facts"]))
        if path != "MULTI_PAIR_STAGED":
            record.update(reason="SCHEDULE_WEIGHTED_READY")
            return record, internal, reuse, None
        # Only the multi-pair arm spends this slice on expert solving. Neither
        # arm receives extra facts from these solves; common interval facts stay fixed.
        until = time.monotonic() + budget.remaining()*config["route_complexity_schedule"]["multi_pair_tier1_fraction"]
        record["schedule"]["tier1_slice_seconds"] = max(0., until-time.monotonic())
        count = program.output_width-1
        for pos, row in enumerate(record["branches"]):
            expert = row["candidate"]
            if all((expert, k) in reuse["facts"] for k in range(count)):
                row.update(unknown_reason="SAFE_PROVED", solver_status="interval_fact_only")
                continue
            try:
                cap = budget.limit("tier1_expert", obligations=len(record["branches"])-pos, until=until)
            except BudgetExhausted:
                break
            result = _solve_output(propagations[expert], output_spec, input_shape=tuple(center.shape), time_limit=cap)
            replay = _forward_validate(model, result.counterexample, lower=lower, upper=upper,
                                       clean_prediction=clean_prediction)
            row.update(solver_status=result.status.value, solver_metadata=dict(result.metadata),
                       full_model_witness_valid=bool(replay["valid"]))
            if replay["valid"]:
                witness = result.counterexample.detach().cpu()
                record.update(status="UNSAFE", reason="UNSAFE_FULL_FORWARD", full_model_witness_valid=True)
                return record, internal, reuse, witness
            budget.check("tier1_expert_complete")
            if result.status.value == "certified":
                row["unknown_reason"] = "SAFE_PROVED"
        if record["branches"] and all(b["unknown_reason"] == "SAFE_PROVED" for b in record["branches"]):
            record.update(status="SAFE", reason="SAFE_PROVED")
        else:
            record.update(reason="SCHEDULE_WEIGHTED_READY")
    except BudgetExhausted as exc:
        record.update(status="TIMEOUT", reason="REQUEST_BUDGET_EXHAUSTED", stopped_at=str(exc))
    return record, internal, reuse, witness
