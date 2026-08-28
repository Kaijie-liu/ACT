# ===- act/pipeline/moe/experiment1f0.py - Weighted Top-2 F0 ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Run F0 only on the frozen Experiment 1C semantic-incompleteness rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from act.back_end.moe import (
    SAFE_WEIGHTED_RANGE,
    UNKNOWN_WEIGHTED_NUMERICAL,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    analyze_topk_sets,
    build_act_moe_program,
    build_weighted_top2_f0,
    condition_topk_set,
    guarded_input_topk_set,
    linear_safety_rows,
    load_output_moe_checkpoint,
    solve_weighted_top2_f0,
)
from act.back_end.solver.solver_hz import SparseHZono
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _git_value,
    _inside,
    _propagate_component,
    _sha256,
    _write_json,
    shared_input_pair_propagation,
)
from act.pipeline.moe.experiment1c import _support_summary
from act.pipeline.moe.train import _load_dataset
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/experiment1f0_bal010.json"
)
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
F0_REASONS = {
    SAFE_WEIGHTED_RANGE,
    UNSAFE_FULL_FORWARD_FALLBACK,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNKNOWN_WEIGHTED_NUMERICAL,
}
CSV_FIELDS = (
    "parent_row_id",
    "parent_line_number",
    "parent_row_sha256",
    "parent_artifact_sha256",
    "sample_rank",
    "dataset_index",
    "epsilon",
    "epsilon_multiplier",
    "parent_reason",
    "status",
    "reason",
    "feasible_pairs",
    "pair_count",
    "full_model_witness_valid",
    "counterexample_prediction",
    "counterexample_topk_set",
    "witness_path",
    "witness_sha256",
    "candidate_seconds",
    "tightening_seconds",
    "solve_seconds",
    "total_seconds",
    "pairs",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _append_json_line(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _log(handle, message: str) -> None:
    handle.write(message.rstrip() + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {field: row.get(field) for field in CSV_FIELDS}
    for field in ("feasible_pairs", "counterexample_topk_set", "pairs"):
        result[field] = json.dumps(
            result[field], sort_keys=True, separators=(",", ":")
        )
    return result


def _load_parent_rows(
    path: Path,
    artifact_hash: str,
    eligible_reasons: Sequence[str],
    expected_rows: int,
) -> list[dict[str, Any]]:
    if _sha256(path) != artifact_hash:
        raise RuntimeError("frozen Experiment 1C artifact hash changed")
    eligible = set(str(value) for value in eligible_reasons)
    selected: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = json.loads(raw)
            if row.get("reason") not in eligible:
                continue
            row_hash = _sha256_bytes(raw)
            identity = _sha256_bytes(
                f"{artifact_hash}:{line_number}:{row_hash}".encode("utf-8")
            )
            selected.append(
                {
                    "parent_row_id": identity,
                    "parent_line_number": line_number,
                    "parent_row_sha256": row_hash,
                    "parent_artifact_sha256": artifact_hash,
                    "parent": row,
                }
            )
    if len(selected) != int(expected_rows):
        raise RuntimeError(
            f"expected {expected_rows} semantic rows, found {len(selected)}"
        )
    return selected


def _support_record(propagation) -> dict[str, Any]:
    summary = _support_summary(propagation.guarded_support)
    summary.update(
        {
            "relu_binaries": propagation.unstable_total,
            "binary_width": propagation.binary_width,
            "propagation_seconds": propagation.elapsed,
        }
    )
    return summary


def _support_status(result) -> dict[str, Any]:
    return {
        "complete_exact": bool(result.exact),
        "lower_status": list(result.lower_status),
        "upper_status": list(result.upper_status),
        "gaps": list(result.solver_gap),
        "solves": int(result.solves),
        "elapsed": float(result.elapsed),
    }


def _validate_candidate(
    model,
    candidate: torch.Tensor | None,
    *,
    clean: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    clean_prediction: int,
) -> dict[str, Any]:
    checked = _forward_validate(
        model,
        candidate,
        lower=lower,
        upper=upper,
        clean_prediction=clean_prediction,
    )
    if candidate is None or candidate.unsqueeze(0).shape != clean.shape:
        checked["linf_distance"] = None
    else:
        checked["linf_distance"] = float(
            (candidate.unsqueeze(0) - clean).abs().max().item()
        )
    return checked


def _save_witness(
    output_dir: Path,
    row_id: str,
    pair: Sequence[int],
    property_index: int,
    candidate: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    relative = Path("witnesses") / (
        f"{row_id}_pair{int(pair[0])}_{int(pair[1])}_property{property_index}.pt"
    )
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite witness {path}")
    torch.save(
        {"input": candidate.detach().cpu(), "metadata": metadata},
        path,
    )
    return str(relative), _sha256(path)


def _pair_reason(property_rows: Sequence[dict[str, Any]]) -> tuple[str, str]:
    if any(row["full_model_witness_valid"] for row in property_rows):
        return "UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK
    if property_rows and all(row["status"] == "SAFE" for row in property_rows):
        return "SAFE", SAFE_WEIGHTED_RANGE
    reasons = {row["reason"] for row in property_rows}
    for reason in (
        UNKNOWN_WEIGHTED_SOLVER_LIMIT,
        UNKNOWN_WEIGHTED_NUMERICAL,
        UNKNOWN_WEIGHTED_RELAXATION,
    ):
        if reason in reasons:
            return "UNKNOWN", reason
    return "UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL


def _row_reason(pair_rows: Sequence[dict[str, Any]]) -> tuple[str, str]:
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


def _run_parent_row(
    *,
    selection: dict[str, Any],
    model,
    dataset,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    parent = selection["parent"]
    rank = int(parent["sample_rank"])
    index = int(parent["dataset_index"])
    epsilon = float(parent["epsilon"])
    image, label = dataset[index]
    x = image.unsqueeze(0).double()
    lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
    with torch.no_grad():
        clean_output, clean_route = model.forward_with_routing(x)
    clean_prediction = int(clean_output.argmax(dim=1).item())
    clean_set = sorted(int(value) for value in clean_route.indices[0].tolist())
    if clean_prediction != int(parent["clean_prediction"]):
        raise RuntimeError(f"clean prediction changed for parent row {rank}")
    if clean_prediction != int(label):
        raise RuntimeError(f"parent row {rank} is no longer clean-correct")
    if clean_set != sorted(int(value) for value in parent["clean_topk_set"]):
        raise RuntimeError(f"clean route set changed for parent row {rank}")

    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[clean_prediction])
    properties = linear_safety_rows(output_spec, int(clean_output.shape[1]))
    program = build_act_moe_program(
        model,
        center=x,
        lower=lower,
        upper=upper,
        output_spec=output_spec,
    )
    candidate_started = time.monotonic()
    router = _propagate_component(program.router)
    if not isinstance(router.output_hz, SparseHZono) or not router.output_hz.exact:
        raise RuntimeError("F0 exact pair enumeration requires an exact sparse router HZ")
    route_sets = analyze_topk_sets(
        router.output_hz,
        model.spec.top_k,
        time_limit_per_set=float(config["candidate_query_timeout"]),
        router_exact=True,
    )
    candidate_seconds = time.monotonic() - candidate_started
    if not route_sets.exact:
        return {
            **{key: value for key, value in selection.items() if key != "parent"},
            "sample_rank": rank,
            "dataset_index": index,
            "epsilon": epsilon,
            "epsilon_multiplier": float(parent["epsilon_multiplier"]),
            "parent_reason": parent["reason"],
            "status": "UNKNOWN",
            "reason": UNKNOWN_WEIGHTED_SOLVER_LIMIT,
            "feasible_pairs": [list(pair) for pair in route_sets.feasible],
            "pair_count": len(route_sets.feasible),
            "route_set_unresolved": [list(pair) for pair in route_sets.unresolved],
            "full_model_witness_valid": False,
            "counterexample_prediction": None,
            "counterexample_topk_set": None,
            "witness_path": None,
            "witness_sha256": None,
            "candidate_seconds": candidate_seconds,
            "tightening_seconds": 0.0,
            "solve_seconds": 0.0,
            "total_seconds": time.monotonic() - started,
            "pairs": [],
        }

    support = config["support"]
    support_config = HybridZConfig(
        max_input_dim=1024,
        guarded_support_enabled=True,
        guarded_support_lp_neurons=int(support["lp_neurons"]),
        guarded_support_milp_neurons=int(support["milp_neurons"]),
        guarded_support_lp_time_limit=float(support["lp_time_limit"]),
        guarded_support_milp_time_limit=float(support["milp_time_limit"]),
    )
    solver = config["solver"]
    pair_rows: list[dict[str, Any]] = []
    total_tightening = 0.0
    total_solve = 0.0
    selected_witness: dict[str, Any] | None = None
    for pair_values in route_sets.feasible:
        pair = tuple(sorted(int(value) for value in pair_values))
        pair_started = time.monotonic()
        property_rows: list[dict[str, Any]] = []
        try:
            conditioned_router = condition_topk_set(
                router.output_hz,
                pair,
            ).hz
            guarded_entry = guarded_input_topk_set(
                router.input_hz,
                router.output_hz,
                pair,
            ).hz
            propagated = shared_input_pair_propagation(
                program.experts[pair[0]],
                program.experts[pair[1]],
                entry_hz=guarded_entry,
                hybridz_config=support_config,
            )
            tightening = (
                propagated.expert_a.elapsed + propagated.expert_b.elapsed
            )
            total_tightening += tightening
            for property_index, (q, constant) in enumerate(properties):
                property_started = time.monotonic()
                encoding = build_weighted_top2_f0(
                    propagated.joint,
                    conditioned_router,
                    pair,
                    q,
                    constant,
                    margin_time_limit=float(solver["margin_support_seconds"]),
                    difference_time_limit=float(
                        solver["difference_support_seconds"]
                    ),
                )
                decision = solve_weighted_top2_f0(
                    encoding,
                    input_shape=tuple(x.shape),
                    time_limit=float(solver["property_seconds"]),
                    tolerance=float(solver["safety_tolerance"]),
                )
                property_elapsed = time.monotonic() - property_started
                total_solve += property_elapsed
                checked = _validate_candidate(
                    model,
                    decision.candidate_input,
                    clean=x,
                    lower=lower,
                    upper=upper,
                    clean_prediction=clean_prediction,
                )
                witness_path = witness_hash = None
                if checked["valid"]:
                    metadata = {
                        "parent_row_id": selection["parent_row_id"],
                        "pair": list(pair),
                        "property_index": property_index,
                        "epsilon": epsilon,
                        "clean_prediction": clean_prediction,
                        "counterexample_prediction": checked["prediction"],
                        "counterexample_topk_set": checked["topk_set"],
                    }
                    witness_path, witness_hash = _save_witness(
                        output_dir,
                        selection["parent_row_id"],
                        pair,
                        property_index,
                        decision.candidate_input,
                        metadata,
                    )
                    selected_witness = {
                        **checked,
                        "path": witness_path,
                        "sha256": witness_hash,
                    }
                competitor = int(
                    next(
                        index
                        for index, value in enumerate(q)
                        if index != clean_prediction and value < 0.0
                    )
                )
                property_status = (
                    "UNSAFE" if checked["valid"] else decision.status
                )
                property_reason = (
                    UNSAFE_FULL_FORWARD_FALLBACK
                    if checked["valid"]
                    else decision.reason
                )
                property_rows.append(
                    {
                        "property_index": property_index,
                        "competitor_class": competitor,
                        "status": property_status,
                        "reason": property_reason,
                        "minimum": decision.minimum,
                        "candidate_objective": decision.candidate_objective,
                        "solver_status": decision.solver_status,
                        "solver_gap": decision.solver_gap,
                        "solver_seconds": decision.elapsed,
                        "property_seconds": property_elapsed,
                        "margin_bounds": list(encoding.margin_bounds),
                        "lambda_bounds": [
                            encoding.bounds.lambda_lower,
                            encoding.bounds.lambda_upper,
                        ],
                        "difference_bounds": [
                            encoding.bounds.difference_lower,
                            encoding.bounds.difference_upper,
                        ],
                        "product_bounds": [
                            encoding.bounds.product_lower,
                            encoding.bounds.product_upper,
                        ],
                        "margin_support": _support_status(
                            encoding.margin_support
                        ),
                        "difference_support": _support_status(
                            encoding.difference_support
                        ),
                        "candidate_recovered": decision.candidate_input is not None,
                        "full_model_witness_valid": bool(checked["valid"]),
                        "counterexample_prediction": checked["prediction"],
                        "counterexample_topk_set": checked["topk_set"],
                        "candidate_linf_distance": checked["linf_distance"],
                        "witness_path": witness_path,
                        "witness_sha256": witness_hash,
                    }
                )
                if checked["valid"]:
                    break
            pair_status, pair_reason = _pair_reason(property_rows)
            pair_rows.append(
                {
                    "pair": list(pair),
                    "status": pair_status,
                    "reason": pair_reason,
                    "shared_continuous": propagated.joint.shared_continuous,
                    "shared_binary": propagated.joint.shared_binary,
                    "expert_a_private_continuous": (
                        propagated.joint.a_private_continuous
                    ),
                    "expert_b_private_continuous": (
                        propagated.joint.b_private_continuous
                    ),
                    "expert_a_private_binary": propagated.joint.a_private_binary,
                    "expert_b_private_binary": propagated.joint.b_private_binary,
                    "expert_a_support": _support_record(propagated.expert_a),
                    "expert_b_support": _support_record(propagated.expert_b),
                    "property_rows": property_rows,
                    "seconds": time.monotonic() - pair_started,
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
                    "seconds": time.monotonic() - pair_started,
                }
            )
        if selected_witness is not None:
            break

    status, reason = _row_reason(pair_rows)
    if reason not in F0_REASONS:
        raise RuntimeError(f"unregistered F0 reason {reason}")
    return {
        **{key: value for key, value in selection.items() if key != "parent"},
        "sample_rank": rank,
        "dataset_index": index,
        "label": int(label),
        "clean_prediction": clean_prediction,
        "clean_topk_set": clean_set,
        "epsilon": epsilon,
        "epsilon_multiplier": float(parent["epsilon_multiplier"]),
        "parent_status": parent["status"],
        "parent_reason": parent["reason"],
        "status": status,
        "reason": reason,
        "feasible_pairs": [list(pair) for pair in route_sets.feasible],
        "pair_count": len(route_sets.feasible),
        "full_model_witness_valid": selected_witness is not None,
        "counterexample_prediction": (
            selected_witness["prediction"] if selected_witness else None
        ),
        "counterexample_topk_set": (
            selected_witness["topk_set"] if selected_witness else None
        ),
        "witness_path": selected_witness["path"] if selected_witness else None,
        "witness_sha256": (
            selected_witness["sha256"] if selected_witness else None
        ),
        "candidate_seconds": candidate_seconds,
        "tightening_seconds": total_tightening,
        "solve_seconds": total_solve,
        "total_seconds": time.monotonic() - started,
        "pairs": pair_rows,
    }


def _summary(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in rows)
    reason_counts = Counter(row["reason"] for row in rows)
    resolved = int(status_counts["SAFE"] + status_counts["UNSAFE"])
    safe_samples = sorted(
        {int(row["sample_rank"]) for row in rows if row["status"] == "SAFE"}
    )
    thresholds = config["preregistered_thresholds"]
    threshold_met = (
        resolved >= int(thresholds["resolved_rows"])
        or len(safe_samples) >= int(thresholds["new_unique_safe_samples"])
    )
    return {
        "rows": len(rows),
        "samples": len({int(row["sample_rank"]) for row in rows}),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "resolved_rows": resolved,
        "new_unique_safe_samples": len(safe_samples),
        "new_unique_safe_sample_ranks": safe_samples,
        "full_forward_unsafe_rows": sum(
            row["status"] == "UNSAFE" and row["full_model_witness_valid"]
            for row in rows
        ),
        "total_candidate_seconds": sum(row["candidate_seconds"] for row in rows),
        "total_tightening_seconds": sum(
            row["tightening_seconds"] for row in rows
        ),
        "total_solve_seconds": sum(row["solve_seconds"] for row in rows),
        "total_seconds": sum(row["total_seconds"] for row in rows),
        "preregistered_threshold_met": threshold_met,
        "next_stage": "freeze_f0_configuration" if threshold_met else "implement_f1",
    }


def run(args) -> dict[str, Any]:
    config_path = _inside(Path(args.config), PROJECT_ROOT)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    parent_path = _inside(Path(config["parent_diagnostics_jsonl"]), WRITE_ROOT)
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("Experiment 1F0 requires feat/moe-route-verification")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("Experiment 1F0 requires a clean feature-branch worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("Experiment 1F0 requires the act-py312 conda environment")
    dataset_root = Path(get_torchvision_data_root()).resolve()
    if not dataset_root.is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("torchvision dataset root escapes /data1/Kane/MOE")

    artifact_hash = str(config["parent_diagnostics_sha256"])
    selected = _load_parent_rows(
        parent_path,
        artifact_hash,
        config["eligible_parent_reasons"],
        int(config["expected_rows"]),
    )
    output_paths = {
        "selection": output_dir / "selection.json",
        "config": output_dir / "config.json",
        "jsonl": output_dir / "results.jsonl",
        "csv": output_dir / "results.csv",
        "summary": output_dir / "summary.json",
        "log": output_dir / "experiment1f0.log",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing or (output_dir / "witnesses").exists():
        raise RuntimeError(f"Experiment 1F0 refuses to overwrite outputs: {existing}")

    selection_record = {
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": artifact_hash,
        "eligible_parent_reasons": config["eligible_parent_reasons"],
        "rows": [
            {key: value for key, value in item.items() if key != "parent"}
            | {
                "sample_rank": item["parent"]["sample_rank"],
                "dataset_index": item["parent"]["dataset_index"],
                "epsilon": item["parent"]["epsilon"],
                "epsilon_multiplier": item["parent"]["epsilon_multiplier"],
                "parent_reason": item["parent"]["reason"],
            }
            for item in selected
        ],
    }
    _write_json(output_paths["selection"], selection_record)
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    if model.spec.top_k != 2 or model.spec.gate.value != "selected_softmax":
        raise RuntimeError("F0 supports only selected-softmax top-2 checkpoints")
    dataset = _load_dataset(payload["dataset"], False, download=False)
    runtime = {
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "git_head": _git_value("rev-parse", "HEAD"),
        "checkpoint_sha256": _sha256(checkpoint),
        "parent_diagnostics_sha256": artifact_hash,
        "torchvision_root": str(dataset_root),
        "config": config,
    }
    _write_json(output_paths["config"], runtime)

    rows: list[dict[str, Any]] = []
    with (
        output_paths["jsonl"].open("x", encoding="utf-8") as jsonl_handle,
        output_paths["csv"].open("x", newline="", encoding="utf-8") as csv_handle,
        output_paths["log"].open("x", encoding="utf-8") as log_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        _log(
            log_handle,
            f"START git_head={runtime['git_head']} rows={len(selected)}",
        )
        for position, item in enumerate(selected, 1):
            row = _run_parent_row(
                selection=item,
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                config=config,
            )
            rows.append(row)
            _append_json_line(jsonl_handle, row)
            writer.writerow(_csv_row(row))
            csv_handle.flush()
            os.fsync(csv_handle.fileno())
            _log(
                log_handle,
                "ROW "
                f"position={position}/{len(selected)} "
                f"parent={row['parent_row_id']} rank={row['sample_rank']} "
                f"multiplier={row['epsilon_multiplier']} "
                f"status={row['status']} reason={row['reason']} "
                f"seconds={row['total_seconds']:.3f}",
            )
        summary = _summary(rows, config)
        _write_json(output_paths["summary"], summary)
        _log(log_handle, f"DONE summary={json.dumps(summary, sort_keys=True)}")
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "manifest": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
