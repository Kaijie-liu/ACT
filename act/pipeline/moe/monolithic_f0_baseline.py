# ===- act/pipeline/moe/monolithic_f0_baseline.py -------------------====#
"""True single-formulation weighted-top2 F0 baseline on the frozen D0 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from act.back_end.moe import (
    UNKNOWN_MONOLITHIC_NUMERICAL,
    UNKNOWN_MONOLITHIC_RELAXATION,
    UNKNOWN_MONOLITHIC_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    build_weighted_top2_f0,
    compute_weighted_top2_gate_range,
    condition_topk_set,
    guarded_input_topk_set,
    load_output_moe_checkpoint,
    solve_monolithic_weighted_top2_f0,
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
    _load_frozen_selection,
    _row_context,
    _support_config,
)
from act.pipeline.moe.experiment1f0 import _validate_candidate
from act.pipeline.moe.train import _load_dataset
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/monolithic_f0_baseline_r1.json"
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")


def _append_json(handle, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _save_witness(
    work_dir: Path,
    rank: int,
    property_index: int,
    candidate: torch.Tensor,
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
    relative = Path("witnesses") / f"rank{rank}_property{property_index}.pt"
    path = work_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input": candidate.detach().cpu(), "metadata": dict(metadata)}, path)
    return str(relative), _sha256(path)


def _row_status(properties: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if any(row.get("full_model_witness_valid") for row in properties):
        return "UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK
    if properties and all(row["status"] == "SAFE" for row in properties):
        return "SAFE", "SAFE_MONOLITHIC_WEIGHTED_RANGE"
    reasons = {str(row["reason"]) for row in properties}
    if UNKNOWN_MONOLITHIC_SOLVER_LIMIT in reasons:
        return "TIMEOUT", UNKNOWN_MONOLITHIC_SOLVER_LIMIT
    if UNKNOWN_MONOLITHIC_NUMERICAL in reasons:
        return "UNKNOWN", UNKNOWN_MONOLITHIC_NUMERICAL
    return "UNKNOWN", UNKNOWN_MONOLITHIC_RELAXATION


def _run_row(selection, model, dataset, config, work_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + float(config["instance_timeout_seconds"])
    context = _row_context(selection, model, dataset, config["selection_source"])
    support_config = _support_config(config["selection_source"])
    pair_data: list[dict[str, Any]] = []
    for pair_values in context["route_sets"].feasible:
        pair = tuple(sorted(int(value) for value in pair_values))
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
        gate_range = compute_weighted_top2_gate_range(
            conditioned_router,
            pair,
            time_limit=float(config["f0"]["margin_support_seconds"]),
        )
        pair_data.append(
            {
                "pair": pair,
                "conditioned_router": conditioned_router,
                "propagated": propagated,
                "gate_range": gate_range,
            }
        )
    property_rows: list[dict[str, Any]] = []
    witness = None
    total_properties = len(context["properties"])
    for property_index, (q, constant) in enumerate(context["properties"]):
        encodings = []
        for item in pair_data:
            encodings.append(
                build_weighted_top2_f0(
                    item["propagated"].joint,
                    item["conditioned_router"],
                    item["pair"],
                    q,
                    constant,
                    difference_time_limit=float(
                        config["f0"]["difference_support_seconds"]
                    ),
                    gate_range=item["gate_range"],
                )
            )
        remaining_properties = total_properties - property_index
        remaining_seconds = max(0.01, deadline - time.monotonic() - 2.0)
        budget = max(0.01, remaining_seconds / max(1, remaining_properties))
        decision = solve_monolithic_weighted_top2_f0(
            encodings,
            input_shape=tuple(context["x"].shape),
            time_limit=budget,
            tolerance=float(config["f0"]["safety_tolerance"]),
        )
        checked = _validate_candidate(
            model,
            decision.candidate_input,
            clean=context["x"],
            lower=context["lower"],
            upper=context["upper"],
            clean_prediction=context["prediction"],
        )
        status = "UNSAFE" if checked["valid"] else decision.status
        reason = UNSAFE_FULL_FORWARD_FALLBACK if checked["valid"] else decision.reason
        witness_path = witness_sha256 = None
        if checked["valid"]:
            witness_path, witness_sha256 = _save_witness(
                work_dir,
                context["rank"],
                property_index,
                decision.candidate_input,
                {
                    "active_relaxation_pair": list(decision.active_pair or ()),
                    "forward_route": checked["topk_set"],
                    "epsilon": context["epsilon"],
                    "clean_prediction": context["prediction"],
                },
            )
            witness = {
                "path": witness_path,
                "sha256": witness_sha256,
                **checked,
            }
        property_row = {
            "property_index": property_index,
            "status": status,
            "reason": reason,
            "minimum": decision.minimum,
            "candidate_objective": decision.candidate_objective,
            "active_relaxation_pair": list(decision.active_pair or ()),
            "solver_status": decision.solver_status,
            "solver_gap": decision.solver_gap,
            "solver_nodes": decision.solver_nodes,
            "solver_certified_lower_bound": decision.solver_certified_lower_bound,
            "solver_bound_kind": decision.solver_bound_kind,
            "pair_count": decision.pair_count,
            "variables": decision.variables,
            "binary_variables": decision.binary_variables,
            "constraint_rows": decision.constraint_rows,
            "budget_seconds": budget,
            "solve_seconds": decision.elapsed,
            "full_model_witness_valid": bool(checked["valid"]),
            "counterexample_prediction": checked["prediction"],
            "counterexample_topk_set": checked["topk_set"],
            "witness_path": witness_path,
            "witness_sha256": witness_sha256,
        }
        property_rows.append(property_row)
        _replace_json(
            work_dir / "progress.json",
            {
                "sample_rank": context["rank"],
                "completed_properties": len(property_rows),
                "active_property": property_index,
                "latest": property_row,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        if checked["valid"]:
            break
    status, reason = _row_status(property_rows)
    return {
        "sample_rank": context["rank"],
        "dataset_index": context["index"],
        "epsilon": context["epsilon"],
        "clean_prediction": context["prediction"],
        "clean_topk_set": context["clean_set"],
        "feasible_pairs": [list(item["pair"]) for item in pair_data],
        "formulation": "single disjunctive MILP over all pair-specific guarded F0 HZs using bounded homogenization",
        "status": status,
        "reason": reason,
        "properties": property_rows,
        "full_model_witness_valid": witness is not None,
        "counterexample_prediction": witness["prediction"] if witness else None,
        "counterexample_topk_set": witness["topk_set"] if witness else None,
        "witness_path": witness["path"] if witness else None,
        "witness_sha256": witness["sha256"] if witness else None,
        "total_seconds": time.monotonic() - started,
    }


def _child(result_path: Path, selection, config, work_dir: Path) -> None:
    try:
        model, payload = load_output_moe_checkpoint(
            Path(config["checkpoint"]), map_location="cpu"
        )
        model.double().eval()
        dataset = _load_dataset(payload["dataset"], False, download=False)
        row = _run_row(selection, model, dataset, config, work_dir)
    except Exception as exc:
        row = {
            "sample_rank": int(selection["sample_rank"]),
            "dataset_index": int(selection["dataset_index"]),
            "epsilon": float(selection["epsilon"]),
            "status": "ERROR",
            "reason": "EXPLICIT_MONOLITHIC_RUNNER_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "full_model_witness_valid": False,
            "total_seconds": 0.0,
        }
    _write_json(result_path, row)


def _run_with_deadline(selection, output_dir: Path, config) -> dict[str, Any]:
    rank = int(selection["sample_rank"])
    work_dir = output_dir / "row_work" / f"rank{rank}"
    work_dir.mkdir(parents=True, exist_ok=False)
    result_path = work_dir / "row.json"
    process = mp.get_context("spawn").Process(
        target=_child,
        args=(result_path, selection, config, work_dir),
        daemon=False,
    )
    started = time.monotonic()
    process.start()
    timeout = float(config["instance_timeout_seconds"])
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        progress_path = work_dir / "progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {}
        )
        return {
            "sample_rank": rank,
            "dataset_index": int(selection["dataset_index"]),
            "epsilon": float(selection["epsilon"]),
            "status": "TIMEOUT",
            "reason": "INSTANCE_HARD_DEADLINE",
            "properties": [],
            "progress_at_deadline": progress,
            "full_model_witness_valid": False,
            "deadline_seconds": timeout,
            "total_seconds": time.monotonic() - started,
        }
    if process.exitcode != 0 or not result_path.exists():
        raise RuntimeError(f"rank {rank} child failed with exit code {process.exitcode}")
    row = json.loads(result_path.read_text(encoding="utf-8"))
    if row.get("witness_path"):
        row["witness_path"] = str(
            Path("row_work") / f"rank{rank}" / row["witness_path"]
        )
    row["deadline_seconds"] = timeout
    row["total_seconds"] = time.monotonic() - started
    return row


def summarize(rows: Sequence[Mapping[str, Any]], reference_rows) -> dict[str, Any]:
    statuses = Counter(str(row["status"]) for row in rows)
    reference = {int(row["sample_rank"]): row for row in reference_rows}
    transitions = Counter()
    for row in rows:
        old = str(reference[int(row["sample_rank"])]["status"])
        transitions[f"route_a_{old}__monolithic_{row['status']}"] += 1
    return {
        "rows": len(rows),
        "status_counts": dict(statuses),
        "solved_rows": statuses["SAFE"] + statuses["UNSAFE"],
        "unsafe_rows": statuses["UNSAFE"],
        "unsafe_full_forward_validated": sum(
            row["status"] == "UNSAFE" and row.get("full_model_witness_valid")
            for row in rows
        ),
        "route_a_reference_status_counts": dict(
            Counter(str(row["status"]) for row in reference_rows)
        ),
        "paired_status_transitions": dict(transitions),
        "total_seconds": sum(float(row["total_seconds"]) for row in rows),
        "claim_scope": (
            "common frozen cohort and 900-second row deadline; runtime is "
            "descriptive because Route A reference artifacts predate this run"
        ),
    }


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = _inside(Path(config["selection_source_config"]), PROJECT_ROOT)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config["selection_source"] = source
    output_dir = _inside(
        Path(config["smoke_output_dir"] if smoke else config["output_dir"]),
        WRITE_ROOT,
    )
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("monolithic baseline requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("monolithic baseline requires a clean worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("monolithic baseline requires act-py312")
    if _sha256(source_path) != config["selection_source_config_sha256"]:
        raise RuntimeError("selection source config changed")
    if config["numerical_safety"] != hz_numerical_policy_manifest():
        raise RuntimeError("numerical policy changed")
    if not Path(get_torchvision_data_root()).resolve().is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision root escapes /data1/Kane/MOE")
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    selected = _load_frozen_selection(source)
    if smoke:
        selected = selected[:1]
        config["instance_timeout_seconds"] = float(config["smoke_timeout_seconds"])
    reference_path = Path(config["route_a_reference_results"])
    if _sha256(reference_path) != config["route_a_reference_results_sha256"]:
        raise RuntimeError("Route A reference results changed")
    reference_rows = [
        json.loads(line)
        for line in reference_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference_by_rank = {int(row["sample_rank"]): row for row in reference_rows}
    reference_selected = [reference_by_rank[int(item["sample_rank"])] for item in selected]
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "config.json",
        {
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "selection_source_config_sha256": _sha256(source_path),
            "route_a_reference_results_sha256": _sha256(reference_path),
            "checkpoint_sha256": _sha256(Path(config["checkpoint"])),
            "git_head": _git_value("rev-parse", "HEAD"),
            "smoke": smoke,
            "config": {key: value for key, value in config.items() if key != "selection_source"},
        },
    )
    rows: list[dict[str, Any]] = []
    with (
        (output_dir / "results.jsonl").open("x", encoding="utf-8") as result_handle,
        (output_dir / "run.log").open("x", encoding="utf-8") as log_handle,
    ):
        for position, selection in enumerate(selected, 1):
            row = _run_with_deadline(selection, output_dir, config)
            rows.append(row)
            _append_json(result_handle, row)
            log_handle.write(
                f"ROW {position}/{len(selected)} rank={row['sample_rank']} "
                f"status={row['status']} reason={row['reason']} "
                f"seconds={row['total_seconds']:.3f}\n"
            )
            log_handle.flush()
            os.fsync(log_handle.fileno())
    summary = summarize(rows, reference_selected)
    _write_json(output_dir / "summary.json", summary)
    return {"output_dir": str(output_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.config), smoke=args.smoke), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
