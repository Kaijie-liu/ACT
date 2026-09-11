"""Monolithic comparison on a direct request's already charged router context."""
from __future__ import annotations

from dataclasses import asdict
import time
import torch

from act.back_end.moe import (
    build_weighted_top2_f0, compute_weighted_top2_gate_range,
    condition_topk_set, guarded_input_topk_set, linear_safety_rows,
    solve_monolithic_weighted_top2_f0,
)
from act.config.config import HybridZConfig
from act.pipeline.moe.experiment1 import shared_input_pair_propagation, _forward_validate
from act.pipeline.moe.request_budget import BudgetExhausted
from act.pipeline.moe.scoped_f0_proofs import reuse_property


def run_monolithic(*, model, center, clean_prediction, internal, config, reuse=None, budget=None):
    started = time.monotonic()
    try:
        return _run_monolithic(model=model, center=center, clean_prediction=clean_prediction,
                               internal=internal, config=config, reuse=reuse, budget=budget)
    except BudgetExhausted as exc:
        # No partial property list may establish SAFE. Source facts remain in
        # the caller's Tier-1 record; this terminal explicitly marks censoring.
        return {"invoked": True, "status": "TIMEOUT", "reason": "REQUEST_BUDGET_EXHAUSTED",
                "stopped_at": str(exc), "property_rows": [], "partial_rows_censored": True,
                "feasible_route_sets": [list(p) for p in internal["route_sets"].feasible],
                "full_model_witness_valid": False, "elapsed_seconds": time.monotonic()-started}, None


def _run_monolithic(*, model, center, clean_prediction, internal, config, reuse=None, budget=None):
    started = time.monotonic()
    support, solver = config["support"], config["solver"]
    hz_config = HybridZConfig(
        max_input_dim=1024, guarded_support_enabled=True,
        guarded_support_lp_neurons=int(support["lp_neurons"]),
        guarded_support_milp_neurons=int(support["milp_neurons"]),
        guarded_support_lp_time_limit=float(support["lp_time_limit"]),
        guarded_support_milp_time_limit=float(support["milp_time_limit"]),
        guarded_support_solver_backend=str(support.get("solver_backend", "scipy")),
        expert_property_solver_backend=str(solver.get("backend", "scipy")),
    )
    router, program = internal["router"], internal["program"]
    properties = linear_safety_rows(internal["output_spec"], program.output_width)
    all_pairs = [tuple(sorted(int(v) for v in p)) for p in internal["route_sets"].feasible]
    reused = {}
    if reuse is not None:
        for pair in all_pairs:
            for index in range(len(properties)):
                proof = reuse_property(reuse["facts"], reuse["scope"], pair, index)
                if proof is not None:
                    reused[pair, index] = proof
    pairs = []
    for pair in all_pairs:
        if all((pair, i) in reused for i in range(len(properties))):
            continue
        if budget:
            budget.check("monolithic_pair_propagation")
            hz_config.guarded_support_lp_time_limit = budget.limit("monolithic_lp_support", float(support["lp_time_limit"]))
            hz_config.guarded_support_milp_time_limit = budget.limit("monolithic_mip_support", float(support["milp_time_limit"]))
        conditioned = condition_topk_set(router.output_hz, pair).hz
        entry = guarded_input_topk_set(router.input_hz, router.output_hz, pair).hz
        propagated = shared_input_pair_propagation(
            program.experts[pair[0]], program.experts[pair[1]],
            entry_hz=entry, hybridz_config=hz_config,
        )
        gates = compute_weighted_top2_gate_range(
            conditioned, pair, time_limit=(budget.limit("monolithic_margin", float(solver["margin_support_seconds"]))
                                          if budget else float(solver["margin_support_seconds"]))
        )
        if budget:
            budget.check("monolithic_pair_complete")
        pairs.append((pair, conditioned, propagated, gates))
    rows, witness = [], None
    for index, (q, constant) in enumerate(properties):
        if budget:
            budget.check("monolithic_property")
        proofs = [{"pair": list(p), "proof": reused[p, index]} for p in all_pairs if (p, index) in reused]
        residual = [p for p in all_pairs if (p, index) not in reused]
        if not residual:
            rows.append({"property_index": index, "status": "SAFE", "reason": "SAFE_REUSED_TIER1_INTERVAL",
                         "minimum": min(v["proof"]["accepted_minimum"] for v in proofs),
                         "pair_count": 0, "solver_status": None, "solver_bound_kind": "scoped_pair_partition",
                         "full_model_witness_valid": False,
                         "coverage_partition": {"reused": proofs, "solved_pairs": []}})
            continue
        encodings = []
        for pair, conditioned, propagated, gates in pairs:
            if pair not in residual:
                continue
            encodings.append(build_weighted_top2_f0(
                propagated.joint, conditioned, pair, q, constant,
                difference_time_limit=(budget.limit("monolithic_difference", float(solver["difference_support_seconds"]))
                                       if budget else float(solver["difference_support_seconds"])), gate_range=gates))
        decision = solve_monolithic_weighted_top2_f0(
            encodings, input_shape=tuple(center.shape),
            time_limit=(budget.limit("monolithic_property_solve", obligations=sum(
                any((p, i) not in reused for p in all_pairs) for i in range(index, len(properties))))
                        if budget else float(solver["property_seconds"])),
            tolerance=float(solver["safety_tolerance"]),
        )
        replay = _forward_validate(
            model, decision.candidate_input, lower=internal["lower"],
            upper=internal["upper"], clean_prediction=clean_prediction,
        )
        row = asdict(decision)
        row.pop("candidate_input", None)
        row.update(property_index=index, full_model_witness_valid=bool(replay["valid"]))
        if replay["valid"]:
            witness = decision.candidate_input.detach().cpu()
            row.update(status="UNSAFE", reason="UNSAFE_FULL_FORWARD_FALLBACK")
        elif budget:
            budget.check("monolithic_property_complete")
        if reuse is not None:
            row["coverage_partition"] = {"reused": proofs, "solved_pairs": [list(p) for p in residual]}
            if proofs and row.get("minimum") is not None:
                row["minimum"] = min(row["minimum"], *(v["proof"]["accepted_minimum"] for v in proofs))
        rows.append(row)
        if witness is not None:
            break
    if witness is not None:
        status, reason = "UNSAFE", "UNSAFE_FULL_FORWARD_FALLBACK"
    elif len(rows) == len(properties) and all_pairs and all(r["status"] == "SAFE" for r in rows):
        status, reason = "SAFE", "SAFE_MONOLITHIC_WEIGHTED_RANGE"
    else:
        status, reason = "UNKNOWN", next(
            (r["reason"] for r in rows if r["status"] != "SAFE"),
            "UNKNOWN_MONOLITHIC_NUMERICAL",
        )
    return {
        "invoked": True, "status": status, "reason": reason,
        "feasible_route_sets": [list(p) for p in all_pairs],
        "property_rows": rows, "full_model_witness_valid": witness is not None,
        "elapsed_seconds": time.monotonic() - started,
    }, witness
