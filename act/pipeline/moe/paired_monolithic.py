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


def run_monolithic(*, model, center, clean_prediction, internal, config):
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
    pairs = []
    for values in internal["route_sets"].feasible:
        pair = tuple(sorted(int(v) for v in values))
        conditioned = condition_topk_set(router.output_hz, pair).hz
        entry = guarded_input_topk_set(router.input_hz, router.output_hz, pair).hz
        propagated = shared_input_pair_propagation(
            program.experts[pair[0]], program.experts[pair[1]],
            entry_hz=entry, hybridz_config=hz_config,
        )
        gates = compute_weighted_top2_gate_range(
            conditioned, pair, time_limit=float(solver["margin_support_seconds"])
        )
        pairs.append((pair, conditioned, propagated, gates))
    rows, witness = [], None
    for index, (q, constant) in enumerate(
        linear_safety_rows(internal["output_spec"], program.output_width)
    ):
        encodings = [build_weighted_top2_f0(
            propagated.joint, conditioned, pair, q, constant,
            difference_time_limit=float(solver["difference_support_seconds"]),
            gate_range=gates,
        ) for pair, conditioned, propagated, gates in pairs]
        decision = solve_monolithic_weighted_top2_f0(
            encodings, input_shape=tuple(center.shape),
            time_limit=float(solver["property_seconds"]),
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
        rows.append(row)
        if witness is not None:
            break
    if witness is not None:
        status, reason = "UNSAFE", "UNSAFE_FULL_FORWARD_FALLBACK"
    elif rows and pairs and all(r["status"] == "SAFE" for r in rows):
        status, reason = "SAFE", "SAFE_MONOLITHIC_WEIGHTED_RANGE"
    else:
        status, reason = "UNKNOWN", next(
            (r["reason"] for r in rows if r["status"] != "SAFE"),
            "UNKNOWN_MONOLITHIC_NUMERICAL",
        )
    return {
        "invoked": True, "status": status, "reason": reason,
        "feasible_route_sets": [list(p[0]) for p in pairs],
        "property_rows": rows, "full_model_witness_valid": witness is not None,
        "elapsed_seconds": time.monotonic() - started,
    }, witness
