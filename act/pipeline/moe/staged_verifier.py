"""Production-style staged verifier for selected-softmax weighted top-2 MoEs.

Unlike the experiment runners, this entry point accepts a direct L-infinity
verification request.  It does not search for a route boundary, execute the
matched no-support ablation, or propagate an unguarded expert for accounting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from act.back_end.moe import (
    GateKind,
    SAFE_WEIGHTED_RANGE,
    UNKNOWN_WEIGHTED_NUMERICAL,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    build_weighted_top2_f0,
    compute_weighted_top2_gate_range,
    condition_topk_set,
    guarded_input_topk_set,
    linear_safety_rows,
    load_output_moe_checkpoint,
    solve_weighted_top2_f0,
)
from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.config.config import HybridZConfig
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _git_value,
    _inside,
    _sha256,
    _write_json,
    shared_input_pair_propagation,
)
from act.pipeline.moe.experiment1c import diagnose_radius
from act.pipeline.moe.experiment1f0 import _support_record, _support_status
from act.pipeline.moe.train import _load_dataset


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/staged_verifier_v1.json"
SEMANTIC_REASONS = {
    "UNKNOWN_GATE_SUFFICIENCY",
    "UNKNOWN_EXPERT_WITNESS_NOT_LIFTED",
}


@dataclass(frozen=True)
class StagedVerificationReport:
    """Serializable evidence plus an optional replayed concrete witness."""

    status: str
    reason: str
    evidence: dict[str, Any]
    witness: torch.Tensor | None = None
    request_tensors: Mapping[str, torch.Tensor] | None = None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": digest.hexdigest(),
    }


def _model_state_identity(model: torch.nn.Module) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensors: list[dict[str, Any]] = []
    for name, value in sorted(model.state_dict().items()):
        identity = _tensor_identity(value)
        digest.update(name.encode("utf-8"))
        digest.update(identity["sha256"].encode("ascii"))
        tensors.append({"name": name, **identity})
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(tensors),
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }


def _require_cpu_float64_model(model: torch.nn.Module) -> None:
    """Pin the exact HybridZ/HiGHS execution contract at the public API."""
    training_modules = [
        name or "<root>" for name, module in model.named_modules() if module.training
    ]
    if training_modules:
        raise ValueError(
            "staged verifier v1 requires eval semantics; training modules: "
            + ", ".join(training_modules[:5])
        )
    tensors = list(model.parameters()) + list(model.buffers())
    devices = {value.device.type for value in tensors}
    floating_dtypes = {
        value.dtype for value in tensors if value.is_floating_point()
    }
    if devices and devices != {"cpu"}:
        raise ValueError(
            "staged verifier v1 requires a CPU model; move the verification "
            "copy to CPU before calling verify_staged_linf"
        )
    if floating_dtypes and floating_dtypes != {torch.float64}:
        raise ValueError(
            "staged verifier v1 requires float64 model parameters and buffers"
        )


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {"candidate_query_timeout", "tier1", "f0", "numerical_safety"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"staged verifier config lacks {sorted(missing)}")
    if config["numerical_safety"] != hz_numerical_policy_manifest():
        raise ValueError("staged verifier numerical policy differs from implementation")
    if float(config["f0"]["solver"]["safety_tolerance"]) != float(
        config["numerical_safety"]["safe_positive_margin"]
    ):
        raise ValueError(
            "F0 acceptance tolerance differs from the frozen numerical policy"
        )
    positive = [
        config["candidate_query_timeout"],
        config["tier1"]["solver"]["low_budget_per_branch"],
        config["tier1"]["solver"]["escalation_budget_per_branch"],
        config["f0"]["solver"]["margin_support_seconds"],
        config["f0"]["solver"]["difference_support_seconds"],
        config["f0"]["solver"]["property_seconds"],
    ]
    if any(float(value) <= 0 for value in positive):
        raise ValueError("all staged verifier budgets must be positive")


def _f0_reason(pair_rows: Sequence[dict[str, Any]]) -> tuple[str, str]:
    if any(row["status"] == "UNSAFE" for row in pair_rows):
        return "UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK
    if pair_rows and all(row["status"] == "SAFE" for row in pair_rows):
        return "SAFE", SAFE_WEIGHTED_RANGE
    reasons = {row["reason"] for row in pair_rows}
    for reason in (
        UNKNOWN_WEIGHTED_SOLVER_LIMIT,
        UNKNOWN_WEIGHTED_NUMERICAL,
        UNKNOWN_WEIGHTED_RELAXATION,
    ):
        if reason in reasons:
            return "UNKNOWN", reason
    return "UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL


def _pair_reason(properties: Sequence[dict[str, Any]]) -> tuple[str, str]:
    if any(row["full_model_witness_valid"] for row in properties):
        return "UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK
    if properties and all(row["status"] == "SAFE" for row in properties):
        return "SAFE", SAFE_WEIGHTED_RANGE
    reasons = {row["reason"] for row in properties}
    for reason in (
        UNKNOWN_WEIGHTED_SOLVER_LIMIT,
        UNKNOWN_WEIGHTED_NUMERICAL,
        UNKNOWN_WEIGHTED_RELAXATION,
    ):
        if reason in reasons:
            return "UNKNOWN", reason
    return "UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL


def _run_f0(
    *,
    model,
    center: torch.Tensor,
    clean_prediction: int,
    internal: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor | None]:
    started = time.monotonic()
    program = internal["program"]
    router = internal["router"]
    route_sets = internal["route_sets"]
    lower = internal["lower"]
    upper = internal["upper"]
    properties = linear_safety_rows(internal["output_spec"], program.output_width)
    support = config["support"]
    solver = config["solver"]
    support_config = HybridZConfig(
        max_input_dim=1024,
        guarded_support_enabled=True,
        guarded_support_lp_neurons=int(support["lp_neurons"]),
        guarded_support_milp_neurons=int(support["milp_neurons"]),
        guarded_support_lp_time_limit=float(support["lp_time_limit"]),
        guarded_support_milp_time_limit=float(support["milp_time_limit"]),
        guarded_support_solver_backend=str(support.get("solver_backend", "scipy")),
        expert_property_solver_backend=str(solver.get("backend", "scipy")),
    )
    pair_rows: list[dict[str, Any]] = []
    witness: torch.Tensor | None = None
    total_tightening = total_solve = 0.0
    for pair_values in route_sets.feasible:
        pair = tuple(sorted(int(value) for value in pair_values))
        pair_started = time.monotonic()
        property_rows: list[dict[str, Any]] = []
        try:
            conditioned_router = condition_topk_set(router.output_hz, pair).hz
            guarded_entry = guarded_input_topk_set(
                router.input_hz, router.output_hz, pair
            ).hz
            propagated = shared_input_pair_propagation(
                program.experts[pair[0]],
                program.experts[pair[1]],
                entry_hz=guarded_entry,
                hybridz_config=support_config,
            )
            tightening_seconds = (
                propagated.expert_a.elapsed + propagated.expert_b.elapsed
            )
            total_tightening += tightening_seconds
            margin_started = time.monotonic()
            gate_range = compute_weighted_top2_gate_range(
                conditioned_router,
                pair,
                time_limit=float(solver["margin_support_seconds"]),
            )
            margin_seconds = time.monotonic() - margin_started
            total_solve += margin_seconds
            for property_index, (q, constant) in enumerate(properties):
                property_started = time.monotonic()
                encoding = build_weighted_top2_f0(
                    propagated.joint,
                    conditioned_router,
                    pair,
                    q,
                    constant,
                    difference_time_limit=float(
                        solver["difference_support_seconds"]
                    ),
                    gate_range=gate_range,
                )
                decision = solve_weighted_top2_f0(
                    encoding,
                    input_shape=tuple(center.shape),
                    time_limit=float(solver["property_seconds"]),
                    tolerance=float(solver["safety_tolerance"]),
                )
                property_seconds = time.monotonic() - property_started
                total_solve += property_seconds
                replay = _forward_validate(
                    model,
                    decision.candidate_input,
                    lower=lower,
                    upper=upper,
                    clean_prediction=clean_prediction,
                )
                valid = bool(replay["valid"])
                if valid:
                    witness = decision.candidate_input.detach().cpu()
                property_rows.append(
                    {
                        "property_index": property_index,
                        "status": "UNSAFE" if valid else decision.status,
                        "reason": (
                            UNSAFE_FULL_FORWARD_FALLBACK
                            if valid
                            else decision.reason
                        ),
                        "solver_status": decision.solver_status,
                        "solver_gap": decision.solver_gap,
                        "accepted_minimum": decision.minimum,
                        "candidate_objective": decision.candidate_objective,
                        "certified_lower_bound": (
                            decision.solver_certified_lower_bound
                        ),
                        "solver_bound_kind": decision.solver_bound_kind,
                        "solver_primal_objective": (
                            decision.solver_primal_objective
                        ),
                        "solver_dual_objective": decision.solver_dual_objective,
                        "margin_bounds": list(encoding.margin_bounds),
                        "lambda_bounds": [
                            encoding.bounds.lambda_lower,
                            encoding.bounds.lambda_upper,
                        ],
                        "difference_bounds": [
                            encoding.bounds.difference_lower,
                            encoding.bounds.difference_upper,
                        ],
                        "full_model_witness_valid": valid,
                        "counterexample_prediction": replay["prediction"],
                        "counterexample_topk_set": replay["topk_set"],
                        "candidate_linf_distance": (
                            float((witness.unsqueeze(0) - center).abs().max().item())
                            if valid and witness is not None
                            else None
                        ),
                        "margin_support": _support_status(encoding.margin_support),
                        "difference_support": _support_status(
                            encoding.difference_support
                        ),
                        "elapsed_seconds": property_seconds,
                    }
                )
                if valid:
                    break
            pair_status, pair_reason = _pair_reason(property_rows)
            pair_rows.append(
                {
                    "pair": list(pair),
                    "status": pair_status,
                    "reason": pair_reason,
                    "expert_a_support": _support_record(propagated.expert_a),
                    "expert_b_support": _support_record(propagated.expert_b),
                    "margin_support": _support_status(gate_range.margin_support),
                    "property_rows": property_rows,
                    "elapsed_seconds": time.monotonic() - pair_started,
                }
            )
        except Exception as exc:
            pair_rows.append(
                {
                    "pair": list(pair),
                    "status": "UNKNOWN",
                    "reason": UNKNOWN_WEIGHTED_NUMERICAL,
                    "error": f"{type(exc).__name__}: {exc}",
                    "property_rows": property_rows,
                    "elapsed_seconds": time.monotonic() - pair_started,
                }
            )
        if witness is not None:
            break
    status, reason = _f0_reason(pair_rows)
    return (
        {
            "invoked": True,
            "status": status,
            "reason": reason,
            "feasible_route_sets": [list(values) for values in route_sets.feasible],
            "pairs": pair_rows,
            "tightening_seconds": total_tightening,
            "solve_seconds": total_solve,
            "elapsed_seconds": time.monotonic() - started,
            "full_model_witness_valid": witness is not None,
        },
        witness,
    )


def verify_staged_linf(
    model,
    center: torch.Tensor,
    epsilon: float,
    config: Mapping[str, Any],
    *,
    expected_clean_prediction: int | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
) -> StagedVerificationReport:
    """Verify top-1 prediction robustness without experiment-only controls."""
    _validate_config(config)
    _require_cpu_float64_model(model)
    if model.spec.gate != GateKind.SELECTED_SOFTMAX or model.spec.top_k != 2:
        raise NotImplementedError(
            "staged verifier v1 supports selected-softmax weighted top-2"
        )
    if float(epsilon) < 0:
        raise ValueError("epsilon must be non-negative")
    started = time.monotonic()
    center = center.detach().cpu().double()
    if center.dim() < 2 or center.shape[0] != 1:
        raise ValueError("center must be a one-lane batched tensor")
    lower = (center - float(epsilon)).clamp(0, 1)
    upper = (center + float(epsilon)).clamp(0, 1)
    with torch.no_grad():
        clean_output, clean_route = model.forward_with_routing(center)
    clean_prediction = int(clean_output.argmax(dim=1).item())
    if (
        expected_clean_prediction is not None
        and clean_prediction != int(expected_clean_prediction)
    ):
        raise ValueError("clean prediction differs from the request")
    clean_set = sorted(int(value) for value in clean_route.indices[0].tolist())
    request_identity = {
        "model_state": _model_state_identity(model),
        "checkpoint": dict(checkpoint_identity or {}),
        "center": _tensor_identity(center),
        "lower": _tensor_identity(lower),
        "upper": _tensor_identity(upper),
        "property": {
            "kind": "TOP1_ROBUST",
            "clean_prediction": clean_prediction,
            "classes": int(clean_output.shape[1]),
        },
        "epsilon": float(epsilon),
        "config_sha256": _canonical_sha256(config),
    }
    transitions = [
        {
            "stage": "REQUEST_ACCEPTED",
            "elapsed_seconds": time.monotonic() - started,
        }
    ]
    tier1_config = {
        "candidate_query_timeout": config["candidate_query_timeout"],
        "support": config["tier1"]["support"],
        "solver": config["tier1"]["solver"],
        "matched_no_support_solve": False,
        "collect_diagnostics": False,
        "return_witness_tensor": True,
        "return_internal_context": True,
    }
    tier1_started = time.monotonic()
    tier1 = diagnose_radius(
        model=model,
        x=center,
        label=clean_prediction,
        clean_prediction=clean_prediction,
        clean_set=clean_set,
        epsilon=float(epsilon),
        epsilon_multiplier=1.0,
        bracket={"kind": "DIRECT_REQUEST_NO_BOUNDARY_SEARCH"},
        config=tier1_config,
    )
    tier1_elapsed = time.monotonic() - tier1_started
    internal = tier1.pop("_internal_context", None)
    witness = tier1.pop("_counterexample_input", None)
    transitions.append(
        {
            "stage": "TIER1_COMPLETE",
            "status": tier1["status"],
            "reason": tier1["reason"],
            "elapsed_seconds": tier1_elapsed,
        }
    )
    tier2: dict[str, Any] = {
        "invoked": False,
        "status": None,
        "reason": "TIER1_DID_NOT_REQUIRE_F0",
        "elapsed_seconds": 0.0,
    }
    status, reason = tier1["status"], tier1["reason"]
    decision_tier = "TIER1_GATE_ELIMINATION"
    if status == "SAFE":
        reason = "SAFE_GATE_ELIMINATION"
    elif status == "UNSAFE":
        reason = "UNSAFE_FULL_FORWARD"
    elif reason in SEMANTIC_REASONS:
        if internal is None:
            raise RuntimeError("Tier 1 semantic incompleteness lacks reusable context")
        tier2, witness = _run_f0(
            model=model,
            center=center,
            clean_prediction=clean_prediction,
            internal=internal,
            config=config["f0"],
        )
        status, reason = tier2["status"], tier2["reason"]
        decision_tier = "TIER2_F0"
        transitions.append(
            {
                "stage": "TIER2_F0_COMPLETE",
                "status": status,
                "reason": reason,
                "elapsed_seconds": tier2["elapsed_seconds"],
            }
        )
    elapsed = time.monotonic() - started
    evidence = {
        "schema_version": 1,
        "verifier": "ACT_HYBRIDZ_STAGED_WEIGHTED_TOP2",
        "request_id": _canonical_sha256(request_identity),
        "identity": request_identity,
        "request": {
            "epsilon": float(epsilon),
            "clean_prediction": clean_prediction,
            "clean_topk_set": clean_set,
        },
        "algorithm": {
            "boundary_search_executed": False,
            "matched_no_support_ablation_executed": False,
            "unguarded_accounting_propagation_executed": False,
            "tier1": "guarded expert-wise gate elimination",
            "tier2": "property-directed weighted top-2 F0",
        },
        "registered_budgets": {
            "candidate_query_timeout": config["candidate_query_timeout"],
            "tier1": config["tier1"],
            "f0": config["f0"],
        },
        "numerical_safety": config["numerical_safety"],
        "transitions": transitions,
        "route_coverage": {
            "candidate_experts": tier1.get("candidate_experts"),
            "feasible_route_sets": tier1.get("feasible_route_sets"),
            "coverage_complete": internal is not None,
            "candidate_set_minimal": (
                bool(internal["candidates"].minimal) if internal is not None else False
            ),
            "route_sets_exact": (
                bool(internal["route_sets"].exact) if internal is not None else False
            ),
        },
        "tier1": tier1,
        "tier2": tier2,
        "verdict": {
            "status": status,
            "reason": reason,
            "decision_tier": decision_tier,
            "certificate_complete": status == "SAFE",
            "full_model_witness_valid": (
                bool(tier2.get("full_model_witness_valid"))
                if tier2["invoked"]
                else bool(tier1.get("full_model_witness_valid"))
            ),
        },
        "timing": {
            "tier1_seconds": tier1_elapsed,
            "tier2_seconds": float(tier2["elapsed_seconds"]),
            "total_seconds": elapsed,
            "all_invoked_stage_times_observed": True,
        },
    }
    return StagedVerificationReport(
        status=status,
        reason=reason,
        evidence=evidence,
        witness=witness,
        request_tensors={
            "center": center.detach().cpu(),
            "lower": lower.detach().cpu(),
            "upper": upper.detach().cpu(),
        },
    )


def write_evidence_package(
    report: StagedVerificationReport,
    output_dir: Path,
) -> dict[str, Any]:
    """Persist one immutable evidence directory and optional replayed witness."""
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence package {output_dir}")
    output_dir.mkdir(parents=True)
    evidence = dict(report.evidence)
    if report.request_tensors is None:
        raise ValueError("evidence package requires represented request tensors")
    request_path = output_dir / "request.pt"
    torch.save(
        {
            **{
                name: value.detach().cpu()
                for name, value in report.request_tensors.items()
            },
            "request_id": evidence["request_id"],
        },
        request_path,
    )
    request_record = {
        "path": str(request_path),
        "sha256": _sha256(request_path),
    }
    witness_record = None
    if report.witness is not None:
        witness_path = output_dir / "witness.pt"
        torch.save(
            {
                "input": report.witness,
                "request_id": evidence["request_id"],
                "verdict": evidence["verdict"],
            },
            witness_path,
        )
        witness_record = {
            "path": str(witness_path),
            "sha256": _sha256(witness_path),
        }
    evidence["witness_artifact"] = witness_record
    evidence_path = output_dir / "evidence.json"
    _write_json(evidence_path, evidence)
    manifest = {
        "schema_version": 1,
        "request_id": evidence["request_id"],
        "status": report.status,
        "reason": report.reason,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "request": request_record,
        "witness": witness_record,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = _inside(args.config, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = _inside(args.checkpoint, WRITE_ROOT)
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.cpu().double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    image, _ = dataset[int(args.dataset_index)]
    report = verify_staged_linf(
        model,
        image.unsqueeze(0),
        float(args.epsilon),
        config,
        checkpoint_identity={
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
    )
    report.evidence["execution"] = {
        "git_head": _git_value("rev-parse", "HEAD"),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "dataset_index": int(args.dataset_index),
    }
    manifest = write_evidence_package(report, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
