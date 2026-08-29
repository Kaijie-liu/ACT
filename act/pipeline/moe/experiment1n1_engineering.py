# ===- experiment1n1_engineering.py - N1 paired engineering rerun -----====#
"""Paired N1 engineering rerun on the frozen 20-row Experiment 1D cohort.

This run never overwrites or reclassifies the confirmatory endpoint. It uses
the same frozen inputs, radii, route sets, support policy, and wall deadline,
but replaces each unresolved range-only F0 property with the retained-margin
segmented F0 implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import torch

from act.back_end.moe import (
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    build_segmented_weighted_top2_f0,
    condition_topk_set,
    guarded_input_topk_set,
    load_output_moe_checkpoint,
    solve_segmented_weighted_top2_f0,
)
from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _git_value,
    _inside,
    _sha256,
    _write_json,
    shared_input_pair_propagation,
)
from act.pipeline.moe.experiment1d import (
    RowRecorder,
    _assert_support_identity,
    _attempt_budgets,
    _load_frozen_selection,
    _row_context,
    _save_witness,
    _support_config,
)
from act.pipeline.moe.experiment1f0 import (
    _pair_reason,
    _row_reason,
    _support_record,
    _support_status,
    _validate_candidate,
)
from act.pipeline.moe.train import _load_dataset
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/experiment1n1_bal010_engineering.json"
)
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
CSV_FIELDS = (
    "sample_rank",
    "dataset_index",
    "epsilon",
    "baseline_status",
    "baseline_reason",
    "status",
    "reason",
    "paired_transition",
    "reused_property_rows",
    "rerun_property_rows",
    "segmented_property_rows",
    "active_segments",
    "full_model_witness_valid",
    "baseline_seconds",
    "total_seconds",
    "error",
)


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_baseline(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = _inside(Path(config["baseline_results_jsonl"]), WRITE_ROOT)
    if _sha256(path) != config["baseline_results_sha256"]:
        raise RuntimeError("paired Experiment 1D baseline artifact changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_rank = {int(row["sample_rank"]): row for row in rows}
    if len(rows) != int(config["expected_rows"]) or len(by_rank) != len(rows):
        raise RuntimeError("paired baseline row count or rank identity changed")
    return by_rank


def _decision_snapshot(decision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "elapsed": decision.elapsed,
        "branches": [
            {
                "status": branch.status,
                "reason": branch.reason,
                "minimum": branch.minimum,
                "solver_status": branch.solver_status,
                "solver_gap": branch.solver_gap,
                "solver_certified_lower_bound": branch.solver_certified_lower_bound,
                "solver_bound_kind": branch.solver_bound_kind,
                "solver_primal_objective": branch.solver_primal_objective,
                "solver_dual_objective": branch.solver_dual_objective,
                "elapsed": branch.elapsed,
            }
            for branch in decision.branch_decisions
        ],
    }


def _branch_snapshot(branch) -> dict[str, Any]:
    return {
        "status": branch.status,
        "reason": branch.reason,
        "minimum": branch.minimum,
        "solver_status": branch.solver_status,
        "solver_gap": branch.solver_gap,
        "solver_certified_lower_bound": branch.solver_certified_lower_bound,
        "solver_bound_kind": branch.solver_bound_kind,
        "solver_primal_objective": branch.solver_primal_objective,
        "solver_dual_objective": branch.solver_dual_objective,
        "elapsed": branch.elapsed,
    }


def _validate_segment_candidates(model, decision, context) -> dict[str, Any]:
    candidates = []
    for candidate in decision.candidate_inputs:
        checked = _validate_candidate(
            model,
            candidate,
            clean=context["x"],
            lower=context["lower"],
            upper=context["upper"],
            clean_prediction=context["prediction"],
        )
        candidates.append(checked)
        if checked["valid"]:
            return {"valid": checked, "checks": candidates, "candidate": candidate}
    return {"valid": None, "checks": candidates, "candidate": None}


def _run_n1_f0(selection, context, model, work_dir: Path, config, recorder):
    parent_f0 = (selection["parent"].get("f0") or {})
    parent_pairs = {tuple(row["pair"]): row for row in parent_f0.get("pairs", [])}
    unresolved_total = sum(
        prop.get("status") not in {"SAFE", "UNSAFE"}
        for pair in parent_pairs.values()
        for prop in pair.get("property_rows", [])
    )
    if not parent_pairs:
        unresolved_total = len(context["route_sets"].feasible) * len(context["properties"])
    remaining_items = max(1, unresolved_total)
    support_config = _support_config(config)
    pair_rows: list[dict[str, Any]] = []
    selected_witness = None
    reused_properties = rerun_properties = segmented_properties = active_segments = 0

    for pair_values in context["route_sets"].feasible:
        pair = tuple(sorted(int(value) for value in pair_values))
        parent_pair = parent_pairs.get(pair)
        if parent_pair is not None and parent_pair.get("status") == "SAFE":
            rows = parent_pair.get("property_rows", [])
            reused_properties += len(rows)
            pair_rows.append({**parent_pair, "reused_parent": True})
            continue
        conditioned_router = condition_topk_set(context["router"].output_hz, pair).hz
        guarded_entry = guarded_input_topk_set(
            context["router"].input_hz,
            context["router"].output_hz,
            pair,
        ).hz
        propagated = shared_input_pair_propagation(
            context["program"].experts[pair[0]],
            context["program"].experts[pair[1]],
            entry_hz=guarded_entry,
            hybridz_config=support_config,
        )
        support_a = _support_record(propagated.expert_a)
        support_b = _support_record(propagated.expert_b)
        identity_a = (
            _assert_support_identity(support_a, parent_pair["expert_a_support"])
            if parent_pair is not None
            else None
        )
        identity_b = (
            _assert_support_identity(support_b, parent_pair["expert_b_support"])
            if parent_pair is not None
            else None
        )
        parent_props = {
            int(row["property_index"]): row
            for row in (parent_pair or {}).get("property_rows", [])
        }
        property_rows: list[dict[str, Any]] = []
        for property_index, (q, constant) in enumerate(context["properties"]):
            parent_prop = parent_props.get(property_index)
            if parent_prop is not None and parent_prop.get("status") in {"SAFE", "UNSAFE"}:
                property_rows.append({**parent_prop, "reused_parent": True})
                reused_properties += 1
                continue
            rerun_properties += 1
            encoding = build_segmented_weighted_top2_f0(
                propagated.joint,
                conditioned_router,
                pair,
                q,
                constant,
                cut_points=tuple(float(value) for value in config["n1"]["margin_cuts"]),
                margin_time_limit=float(config["f0"]["margin_support_seconds"]),
                feasibility_time_limit=float(config["n1"]["segment_feasibility_seconds"]),
                difference_time_limit=float(config["f0"]["difference_support_seconds"]),
            )
            segmented_properties += 1
            active_segments += len(encoding.branches)
            first_total, second_total = _attempt_budgets(
                recorder,
                remaining_items,
                float(config["instance_timeout_seconds"]),
            )
            branch_count = max(1, len(encoding.branches))
            attempts = []
            decision = solve_segmented_weighted_top2_f0(
                encoding,
                input_shape=tuple(context["x"].shape),
                time_limit_per_segment=first_total / branch_count,
                tolerance=float(config["f0"]["safety_tolerance"]),
            )
            attempts.append({"total_budget": first_total, **_decision_snapshot(decision)})
            recorder.progress(
                active={"tier": "N1_F0", "pair": list(pair),
                        "property_index": property_index, "attempt": 1},
                solver={"status": decision.status, "reason": decision.reason,
                        "segments": len(encoding.branches)},
            )
            validated = _validate_segment_candidates(model, decision, context)
            if (
                validated["valid"] is None
                and decision.status == "UNKNOWN"
                and decision.reason == UNKNOWN_WEIGHTED_SOLVER_LIMIT
                and second_total > 0.01
            ):
                decision = solve_segmented_weighted_top2_f0(
                    encoding,
                    input_shape=tuple(context["x"].shape),
                    time_limit_per_segment=second_total / branch_count,
                    tolerance=float(config["f0"]["safety_tolerance"]),
                )
                attempts.append({"total_budget": second_total, **_decision_snapshot(decision)})
                recorder.progress(
                    active={"tier": "N1_F0", "pair": list(pair),
                            "property_index": property_index, "attempt": 2},
                    solver={"status": decision.status, "reason": decision.reason,
                            "segments": len(encoding.branches)},
                )
                validated = _validate_segment_candidates(model, decision, context)
            remaining_items -= 1
            witness_path = witness_hash = None
            checked = validated["valid"]
            if checked is not None:
                witness_path, witness_hash = _save_witness(
                    work_dir,
                    context["rank"],
                    "n1",
                    f"{pair[0]}_{pair[1]}_{property_index}",
                    validated["candidate"],
                    {"pair": list(pair), "property_index": property_index,
                     "epsilon": context["epsilon"],
                     "clean_prediction": context["prediction"]},
                )
                selected_witness = {**checked, "path": witness_path, "sha256": witness_hash}
            status = "UNSAFE" if checked is not None else decision.status
            reason = UNSAFE_FULL_FORWARD_FALLBACK if checked is not None else decision.reason
            segments = []
            for branch, branch_decision in zip(
                encoding.branches, decision.branch_decisions, strict=True
            ):
                bounds = branch.encoding.bounds
                segments.append(
                    {
                        "index": branch.segment.index,
                        "margin_interval": [branch.segment.lower, branch.segment.upper],
                        "margin_bounds": list(branch.encoding.margin_bounds),
                        "lambda_bounds": [bounds.lambda_lower, bounds.lambda_upper],
                        "difference_bounds": [bounds.difference_lower, bounds.difference_upper],
                        "product_bounds": [bounds.product_lower, bounds.product_upper],
                        "support_fallback_reason": branch.support.fallback_reason,
                        "decision": _branch_snapshot(branch_decision),
                    }
                )
            property_rows.append(
                {
                    "property_index": property_index,
                    "status": status,
                    "reason": reason,
                    "segments": segments,
                    "conditioned_support_telemetry": encoding.conditioned_support.telemetry.__dict__,
                    "unconditional_difference_bounds": list(
                        encoding.conditioned_support.unconditional_bounds
                    ),
                    "attempts": attempts,
                    "candidate_checks": validated["checks"],
                    "full_model_witness_valid": checked is not None,
                    "counterexample_prediction": checked["prediction"] if checked else None,
                    "counterexample_topk_set": checked["topk_set"] if checked else None,
                    "witness_path": witness_path,
                    "witness_sha256": witness_hash,
                    "reused_parent": False,
                }
            )
            if checked is not None:
                break
        pair_status, pair_reason = _pair_reason(property_rows)
        pair_rows.append(
            {
                "pair": list(pair),
                "status": pair_status,
                "reason": pair_reason,
                "expert_a_support": support_a,
                "expert_b_support": support_b,
                "expert_a_support_identity": identity_a,
                "expert_b_support_identity": identity_b,
                "property_rows": property_rows,
                "reused_parent": False,
            }
        )
        if selected_witness is not None:
            break
    status, reason = _row_reason(pair_rows)
    return {
        "status": status,
        "reason": reason,
        "feasible_pairs": [list(pair) for pair in context["route_sets"].feasible],
        "pairs": pair_rows,
        "reused_property_rows": reused_properties,
        "rerun_property_rows": rerun_properties,
        "segmented_property_rows": segmented_properties,
        "active_segments": active_segments,
        "full_model_witness_valid": selected_witness is not None,
        "counterexample_prediction": selected_witness["prediction"] if selected_witness else None,
        "counterexample_topk_set": selected_witness["topk_set"] if selected_witness else None,
        "witness_path": selected_witness["path"] if selected_witness else None,
        "witness_sha256": selected_witness["sha256"] if selected_witness else None,
    }


def _run_row(selection, baseline, model, dataset, work_dir: Path, config):
    started = time.monotonic()
    recorder = RowRecorder(work_dir, float(config["checkpoint_seconds"]))
    context = _row_context(selection, model, dataset, config)
    n1 = _run_n1_f0(selection, context, model, work_dir, config, recorder)
    recorder.finish_checkpoint(n1["status"])
    return {
        "sample_rank": context["rank"],
        "dataset_index": context["index"],
        "epsilon": context["epsilon"],
        "baseline_status": baseline["status"],
        "baseline_reason": baseline["reason"],
        "baseline_seconds": baseline["total_seconds"],
        "status": n1["status"],
        "reason": n1["reason"],
        "paired_transition": f"{baseline['status']}->{n1['status']}",
        "reused_property_rows": n1["reused_property_rows"],
        "rerun_property_rows": n1["rerun_property_rows"],
        "segmented_property_rows": n1["segmented_property_rows"],
        "active_segments": n1["active_segments"],
        "full_model_witness_valid": n1["full_model_witness_valid"],
        "witness_path": n1["witness_path"],
        "witness_sha256": n1["witness_sha256"],
        "n1": n1,
        "total_seconds": time.monotonic() - started,
    }


def _child(result_path, selection, baseline, work_dir, config):
    try:
        model, payload = load_output_moe_checkpoint(
            Path(config["checkpoint"]), map_location="cpu"
        )
        model.double().eval()
        dataset = _load_dataset(payload["dataset"], False, download=False)
        row = _run_row(selection, baseline, model, dataset, work_dir, config)
    except Exception as exc:
        row = {
            "sample_rank": selection["sample_rank"],
            "dataset_index": selection["dataset_index"],
            "baseline_status": baseline["status"],
            "baseline_reason": baseline["reason"],
            "status": "ERROR",
            "reason": "EXPLICIT_NUMERICAL_OR_RUNNER_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "total_seconds": 0.0,
        }
    _write_json(result_path, row)


def _run_with_deadline(selection, baseline, output_dir: Path, config):
    rank = int(selection["sample_rank"])
    work_dir = output_dir / "row_work" / f"rank{rank}"
    work_dir.mkdir(parents=True, exist_ok=False)
    result_path = work_dir / "row.json"
    process = mp.get_context("spawn").Process(
        target=_child,
        args=(result_path, selection, baseline, work_dir, config),
        daemon=False,
    )
    started = time.monotonic()
    process.start()
    process.join(timeout=float(config["instance_timeout_seconds"]))
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "sample_rank": rank,
            "dataset_index": selection["dataset_index"],
            "epsilon": selection["epsilon"],
            "baseline_status": baseline["status"],
            "baseline_reason": baseline["reason"],
            "baseline_seconds": baseline["total_seconds"],
            "status": "TIMEOUT",
            "reason": "INSTANCE_HARD_DEADLINE",
            "paired_transition": f"{baseline['status']}->TIMEOUT",
            "full_model_witness_valid": False,
            "deadline_seconds": float(config["instance_timeout_seconds"]),
            "total_seconds": time.monotonic() - started,
        }
    if process.exitcode != 0 or not result_path.exists():
        raise RuntimeError(f"rank {rank} child failed with exit code {process.exitcode}")
    row = json.loads(result_path.read_text(encoding="utf-8"))
    if row.get("witness_path"):
        row["witness_path"] = str(Path("row_work") / f"rank{rank}" / row["witness_path"])
    row["deadline_seconds"] = float(config["instance_timeout_seconds"])
    row["total_seconds"] = time.monotonic() - started
    return row


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter(row.get("paired_transition") for row in rows)
    baseline_solved = sum(row["baseline_status"] in {"SAFE", "UNSAFE"} for row in rows)
    n1_solved = sum(row["status"] in {"SAFE", "UNSAFE"} for row in rows)
    paired_times = [
        float(row["total_seconds"]) - float(row["baseline_seconds"])
        for row in rows
        if row.get("baseline_seconds") is not None
    ]
    semantic_conflicts = [
        int(row["sample_rank"])
        for row in rows
        if (row["baseline_status"], row["status"])
        in {("SAFE", "UNSAFE"), ("UNSAFE", "SAFE")}
    ]
    unreplayed_unsafe = [
        int(row["sample_rank"])
        for row in rows
        if row["status"] == "UNSAFE" and not row.get("full_model_witness_valid")
    ]
    return {
        "scope": "engineering_performance_rerun_not_confirmatory_overwrite",
        "rows": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "paired_transitions": dict(sorted(transitions.items())),
        "baseline_solved_rows": baseline_solved,
        "n1_solved_rows": n1_solved,
        "net_solved_change": n1_solved - baseline_solved,
        "segmented_property_rows": sum(int(row.get("segmented_property_rows", 0)) for row in rows),
        "active_segments": sum(int(row.get("active_segments", 0)) for row in rows),
        "all_unsafe_full_forward_validated": all(
            row["status"] != "UNSAFE" or row.get("full_model_witness_valid")
            for row in rows
        ),
        "semantic_conflict_sample_ranks": semantic_conflicts,
        "unreplayed_unsafe_sample_ranks": unreplayed_unsafe,
        "audit_issues_pre_independent_audit": len(semantic_conflicts)
        + len(unreplayed_unsafe)
        + sum(row["status"] == "ERROR" for row in rows),
        "paired_runtime_difference_median": statistics.median(paired_times) if paired_times else None,
        "paired_runtime_difference_values": paired_times,
        "original_confirmatory_overall_solved_rate_immutable": 0.56,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("N1 engineering rerun requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("N1 engineering rerun requires a clean worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("N1 engineering rerun requires act-py312")
    if not Path(get_torchvision_data_root()).resolve().is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision data root escapes /data1/Kane/MOE")
    if config["numerical_safety"] != hz_numerical_policy_manifest():
        raise RuntimeError("frozen numerical policy differs from implementation")
    selected = _load_frozen_selection(config)
    baseline = _load_baseline(config)
    if output_dir.exists():
        raise RuntimeError(f"N1 engineering rerun refuses to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "config.json",
        {
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "git_head": _git_value("rev-parse", "HEAD"),
            "checkpoint_sha256": _sha256(Path(config["checkpoint"])),
            "config": config,
        },
    )
    rows = []
    json_path, csv_path = output_dir / "results.jsonl", output_dir / "results.csv"
    with json_path.open("a", encoding="utf-8") as json_handle, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in selected:
            row = _run_with_deadline(item, baseline[int(item["sample_rank"])], output_dir, config)
            rows.append(row)
            _append_json(json_handle, row)
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in CSV_FIELDS
                }
            )
            csv_handle.flush()
            os.fsync(csv_handle.fileno())
    summary = _summary(rows)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
