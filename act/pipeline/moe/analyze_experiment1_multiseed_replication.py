"""Aggregate the independently audited multi-seed Route A replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


WORKSPACE = Path("/data1/Kane/MOE")
PROJECT_ROOT = WORKSPACE / "ACT"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/experiment1_multiseed_replication_20260906_r1.json"
)
DEFAULT_SELECTION = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/experiment1_multiseed_selection_r1.json"
)
DEFAULT_RUNS = (
    PROJECT_ROOT / "data/moe/results/experiment1_multiseed_replication_seed1_r1",
    PROJECT_ROOT / "data/moe/results/experiment1_multiseed_replication_seed2_r1",
)


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(WORKSPACE)
    return resolved


def _load(path: Path) -> dict[str, Any]:
    return json.loads(_inside(path).read_text(encoding="utf-8"))


def _model_result(run_dir: Path) -> dict[str, Any]:
    run_dir = _inside(run_dir)
    runtime_config_path = run_dir / "config.json"
    audit_path = run_dir / "independent_audit.json"
    census_path = run_dir / "census/summary.json"
    boundary_path = run_dir / "boundary/summary.json"
    runtime_config = _load(runtime_config_path)
    audit = _load(audit_path)
    census = _load(census_path)
    boundary = _load(boundary_path)

    if audit.get("issue_count") != 0 or audit.get("issues") != []:
        raise RuntimeError(f"independent audit did not pass: {audit_path}")
    if not audit.get("soundness_audit_passed"):
        raise RuntimeError(f"soundness audit did not pass: {audit_path}")
    source_config_path = _inside(Path(runtime_config["source_config"]))
    checkpoint_path = _inside(Path(runtime_config["config"]["checkpoint"]))
    if runtime_config.get("source_config_sha256") != _sha256(source_config_path):
        raise RuntimeError(f"source config identity mismatch: {run_dir}")
    if runtime_config.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise RuntimeError(f"checkpoint identity mismatch: {run_dir}")
    if census.get("rows") != 160 or census.get("samples") != 40:
        raise RuntimeError(f"incomplete census: {run_dir}")
    if boundary.get("rows") != 40 or boundary.get("samples") != 40:
        raise RuntimeError(f"incomplete boundary run: {run_dir}")

    candidate = audit["candidate_reduction"]
    width = audit["width"]["route_unstable"]
    endpoint = audit["end_to_end"]
    guard = audit["guard_support"]
    go = audit["go_conditions"]
    no_boundary = int(
        endpoint["reason_counts"].get("NO_ROUTE_BOUNDARY_WITHIN_SEARCH", 0)
    )
    applicable = 40 - no_boundary
    solved = int(endpoint["solved_rows"])
    registered_endpoint_pass = all(bool(value) for value in go.values())
    model_id = str(runtime_config["config"]["model_id"])

    return {
        "model_id": model_id,
        "execution_head": runtime_config["git_head"],
        "checkpoint_sha256": runtime_config["checkpoint_sha256"],
        "candidate_reduction": candidate,
        "route_unstable_width": width,
        "guard_support": {
            **guard,
            "binary_reduction_rate": (
                guard["accounting"]["binary_eliminated"]
                / guard["accounting"]["binaries_before"]
            ),
        },
        "boundary_endpoint": {
            **endpoint,
            "no_boundary_within_4_over_255": no_boundary,
            "applicable_route_boundary_samples": applicable,
            "conditional_solved_rate": solved / applicable,
        },
        "registered_go_conditions": go,
        "registered_endpoint_pass": registered_endpoint_pass,
        "raw_artifacts": {
            "runtime_config": str(runtime_config_path),
            "runtime_config_sha256": _sha256(runtime_config_path),
            "census_summary": str(census_path),
            "census_summary_sha256": _sha256(census_path),
            "boundary_summary": str(boundary_path),
            "boundary_summary_sha256": _sha256(boundary_path),
            "independent_audit": str(audit_path),
            "independent_audit_sha256": _sha256(audit_path),
        },
    }


def analyze(selection_path: Path, run_dirs: tuple[Path, ...]) -> dict[str, Any]:
    selection_path = _inside(selection_path)
    selection = _load(selection_path)
    if selection.get("status") != "FROZEN_BEFORE_FORMAL_ENDPOINT":
        raise RuntimeError("selection manifest is not the frozen R1 manifest")
    models = [_model_result(path) for path in run_dirs]
    if [row["model_id"] for row in models] != ["seed1", "seed2"]:
        raise RuntimeError("registered model order differs")
    heads = {row["execution_head"] for row in models}
    if len(heads) != 1:
        raise RuntimeError("models were executed at different implementation heads")
    if any(
        row["checkpoint_sha256"]
        != selection["models"][row["model_id"]]["checkpoint_sha256"]
        for row in models
    ):
        raise RuntimeError("selection and execution checkpoint identities differ")

    def passed(key: str) -> int:
        return sum(bool(row["registered_go_conditions"][key]) for row in models)

    total_safe = sum(
        int(row["boundary_endpoint"]["unique_safe_samples"]) for row in models
    )
    total_unsafe = sum(
        int(row["boundary_endpoint"]["unsafe_rows"]) for row in models
    )
    total_solved = sum(
        int(row["boundary_endpoint"]["solved_rows"]) for row in models
    )
    total_applicable = sum(
        int(row["boundary_endpoint"]["applicable_route_boundary_samples"])
        for row in models
    )
    complete_passes = sum(row["registered_endpoint_pass"] for row in models)
    return {
        "experiment": "experiment1_multiseed_replication_r1",
        "status": "COMPLETED_AUDITED_MIXED_ENDPOINT",
        "execution_head": next(iter(heads)),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "selection": {
            "samples_per_model": 40,
            "same_images_for_both_models": True,
            "prior_seed0_cohorts_excluded": True,
            "selection_used_verification_outcomes": False,
        },
        "models": models,
        "cross_model_decision": {
            "registered_complete_endpoint_passes": complete_passes,
            "registered_models": len(models),
            "seed_robust_full_bundle_supported": complete_passes == len(models),
            "mechanism_pass_counts": {
                "candidate_reduction_vs_ibp": passed("candidate_reduction_ibp"),
                "candidate_reduction_vs_zonotope": passed(
                    "candidate_reduction_zonotope"
                ),
                "route_unstable_width_median": passed(
                    "route_unstable_width_median"
                ),
                "route_unstable_width_p90": passed("route_unstable_width_p90"),
                "route_changing_unique_safe": passed("unique_safe_count"),
                "all_unsafe_full_model_replay": passed("all_unsafe_replayed"),
                "end_to_end_solved_rate": passed("end_to_end_solved_rate"),
                "f0_semantic_resolution": passed("f0_semantic_resolution"),
                "independent_audit_zero_issues": passed(
                    "independent_audit_zero_issues"
                ),
            },
            "descriptive_totals_not_primary_endpoints": {
                "unique_safe_model_sample_pairs": total_safe,
                "unsafe_model_sample_pairs": total_unsafe,
                "applicable_route_boundary_model_sample_pairs": total_applicable,
                "solved_applicable_model_sample_pairs": total_solved,
                "conditional_solved_rate": total_solved / total_applicable,
            },
        },
        "scientific_decision": {
            "full_registered_bundle": (
                "NOT_REPLICATED_2_OF_2; neither model passed every registered "
                "condition (seed1 missed zonotope candidate reduction; seed2 "
                "missed overall solved-rate)"
            ),
            "route_changing_certificates": (
                "REPLICATED_2_OF_2; both retained models produced full-denominator "
                "route-changing unique SAFE certificates"
            ),
            "binary_width_separation": (
                "REPLICATED_2_OF_2 at the registered median and p90 thresholds"
            ),
            "candidate_reduction": (
                "MODEL_DEPENDENT; IBP threshold passed 2/2, ordinary-zonotope "
                "threshold passed 1/2"
            ),
            "solver_coverage": (
                "MODEL_DEPENDENT; overall registered solved-rate passed 1/2; "
                "no-boundary rows are reported separately from applicable rows"
            ),
        },
        "claim_boundary": (
            "Models are primary units and are reported separately. Descriptive pooled "
            "model-sample counts cannot replace the preregistered 2/2 criterion. The "
            "result supports multi-seed route-changing certificates and conditional "
            "width separation, but not seed-robust wording for the complete endpoint "
            "bundle or for zonotope candidate reduction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_dirs = tuple(args.run_dir) if args.run_dir else DEFAULT_RUNS
    result = analyze(args.selection, run_dirs)
    output = _inside(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
