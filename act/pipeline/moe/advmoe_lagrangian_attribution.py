"""Attribute the frozen AdvMoE Lagrangian development null endpoint.

This diagnostic deliberately reuses only the accepted development cohort.  It
extends the frozen multiplier grid on a deterministic closest-to-completion
subset and never changes, relabels, or overwrites the parent experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoePath,
    CrownCompatibleAdvMoeRouter,
    specialize_advmoe_path,
)
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.advmoe_two_path import (
    _cleanup_cuda,
    _load_model,
    aggregate_lagrangian_grid_calls,
    evaluate_lagrangian_guard_grid,
    top1_property_rows,
)
from act.pipeline.moe.crown_adapter_cohort import validate_crown_configuration
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


POSITIVE = "CERTIFIED_MARGIN_FILTER"
ENDPOINT_POSITIVE = "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def lagrangian_row_minimum(row: dict[str, Any]) -> float:
    values = [
        float(branch["minimum_lower_bound"])
        for branch in row["lagrangian_guard_crown"]
    ]
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"malformed Lagrangian branch minima: {row['row_id']}")
    return min(values)


def select_closest_residual_rows(
    parent_rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select a deterministic attribution-only subset from parent results."""

    eligible = [
        row
        for row in parent_rows
        if row["statuses"]["lagrangian_guard_ablation"] != ENDPOINT_POSITIVE
        and not bool(row["attack"]["prediction_flip"])
    ]
    eligible.sort(key=lambda row: (-lagrangian_row_minimum(row), row["row_id"]))
    if len(eligible) < int(count):
        raise RuntimeError("insufficient residual development rows")
    return eligible[: int(count)]


def combine_parent_and_extension(
    parent_branch: dict[str, Any],
    extension_branch: dict[str, Any] | None,
    *,
    parent_multipliers: list[float],
    extension_multipliers: list[float],
    property_rows: int,
    tolerance: float,
) -> dict[str, Any]:
    """Re-aggregate the immutable parent calls with optional new calls."""

    if extension_branch is None:
        return copy.deepcopy(parent_branch)
    calls = list(parent_branch["calls"]) + list(extension_branch["calls"])
    multipliers = list(parent_multipliers) + list(extension_multipliers)
    return aggregate_lagrangian_grid_calls(
        calls,
        multipliers,
        property_rows=property_rows,
        tolerance=tolerance,
    )


def _parent_artifacts(
    config: dict[str, Any], workspace: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    parent = config["parent_development"]
    paths: dict[str, Path] = {}
    values: dict[str, dict[str, Any]] = {}
    for name in ("config", "summary", "audit", "analysis", "rows"):
        entry = parent[name]
        path = _inside(Path(entry["path"]), workspace)
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"parent {name} hash mismatch")
        paths[name] = path
        if name != "rows":
            values[name] = _read_json(path)
    if values["audit"].get("status") != "PASS" or values["audit"].get("issues") != []:
        raise RuntimeError("parent development audit is not accepted")
    if values["analysis"].get("status") != "PASS":
        raise RuntimeError("parent development analysis is not accepted")
    rows = _read_jsonl(paths["rows"])
    if len(rows) != int(config["selection"]["parent_rows"]):
        raise RuntimeError("parent row count mismatch")
    return values["config"], values["summary"], rows, paths["rows"]


def run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    workspace = Path(config["workspace_boundary"])
    config_path = _inside(config_path, workspace)
    repository = _inside(Path(config["act_repository"]), workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("attribution configuration is not frozen")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if _git(repository, "branch", "--show-current") != config["required_branch"]:
        raise RuntimeError("ACT branch gate failed")
    if _git(repository, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree is dirty")

    parent_config, parent_summary, parent_rows, parent_rows_path = _parent_artifacts(
        config, workspace
    )
    selected = select_closest_residual_rows(
        parent_rows, int(config["selection"]["rows"])
    )
    selected_ids = [row["row_id"] for row in selected]
    if selected_ids != config["selection"]["expected_row_ids"]:
        raise RuntimeError("deterministic attribution subset changed")

    parent_multipliers = [
        float(value)
        for value in parent_summary["lagrangian_multiplier_protocol"][
            "resolved_multipliers"
        ]
    ]
    extension_multipliers = [
        float(value) for value in config["multiplier_extension"]["multipliers"]
    ]
    if set(parent_multipliers).intersection(extension_multipliers):
        raise ValueError("extension grid overlaps the frozen parent grid")
    if extension_multipliers != sorted(extension_multipliers) or any(
        not math.isfinite(value) or value <= max(parent_multipliers)
        for value in extension_multipliers
    ):
        raise ValueError("extension multipliers must be sorted and above parent grid")

    crown = config["crown"]
    crown_configuration = validate_crown_configuration(
        method=crown["method"],
        track_gradients=bool(crown["gradient_tracking"]),
        bound_options=crown.get("bound_options"),
    )
    free, total = torch.cuda.mem_get_info(crown["device"])
    if free / 1024**3 < float(crown["minimum_free_gpu_memory_gib"]):
        raise RuntimeError("GPU memory gate failed")

    source = _inside(Path(parent_config["official_source"]["repository"]), workspace)
    if _git(source, "rev-parse", "HEAD") != parent_config["official_source"]["commit"]:
        raise RuntimeError("official source commit mismatch")
    if _git(source, "rev-parse", "HEAD^{tree}") != parent_config["official_source"]["tree"]:
        raise RuntimeError("official source tree mismatch")
    if _git(source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")

    archive = _inside(Path(parent_config["dataset_archive"]), workspace)
    inputs, _labels = load_cifar10_test_archive(archive)
    model, router, moe_type, checkpoint = _load_model(parent_config, workspace)
    specialized = [
        specialize_advmoe_path(model, route, moe_type)[0].eval()
        for route in (0, 1)
    ]
    path_adapters = [CrownCompatibleAdvMoePath(path).eval() for path in specialized]
    router_adapter = CrownCompatibleAdvMoeRouter(router).eval()
    tolerance = float(config["numerical"]["safe_positive_margin"])

    output_dir.mkdir(parents=True)
    rows_path = output_dir / "rows.jsonl"
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with rows_path.open("x", encoding="utf-8") as handle:
        for parent_row in selected:
            center = torch.from_numpy(
                inputs[int(parent_row["dataset_index"]) : int(parent_row["dataset_index"]) + 1]
            )
            epsilon = float(parent_row["epsilon"])
            lower = torch.clamp(center - epsilon, 0.0, 1.0)
            upper = torch.clamp(center + epsilon, 0.0, 1.0)
            properties = top1_property_rows(int(parent_row["clean_prediction"]))
            branch_records: list[dict[str, Any]] = []
            for route in (0, 1):
                parent_branch = parent_row["lagrangian_guard_crown"][route]
                extension = None
                if parent_branch["status"] != POSITIVE:
                    extension = evaluate_lagrangian_guard_grid(
                        router_adapter,
                        path_adapters[route],
                        route,
                        center,
                        lower,
                        upper,
                        property_rows=properties,
                        multipliers=extension_multipliers,
                        device=crown["device"],
                        tolerance=tolerance,
                        method=crown["method"],
                        track_gradients=bool(crown["gradient_tracking"]),
                        bound_options=crown.get("bound_options"),
                    )
                combined = combine_parent_and_extension(
                    parent_branch,
                    extension,
                    parent_multipliers=parent_multipliers,
                    extension_multipliers=extension_multipliers,
                    property_rows=len(properties),
                    tolerance=tolerance,
                )
                old = np.asarray(parent_branch["lower_bounds"], dtype=np.float64)
                new = np.asarray(combined["lower_bounds"], dtype=np.float64)
                delta = new - old
                branch_records.append(
                    {
                        "route": route,
                        "parent": {
                            "status": parent_branch["status"],
                            "minimum_lower_bound": parent_branch["minimum_lower_bound"],
                            "lower_bounds": parent_branch["lower_bounds"],
                            "selected_multipliers": parent_branch["selected_multipliers"],
                        },
                        "extension": extension,
                        "combined": combined,
                        "strict_property_improvements": int(np.sum(delta > tolerance)),
                        "strict_property_regressions": int(np.sum(delta < -tolerance)),
                        "maximum_property_improvement": float(delta.max()),
                    }
                )
            combined_positive = all(
                branch["combined"]["status"] == POSITIVE for branch in branch_records
            )
            parent_positive = (
                parent_row["statuses"]["lagrangian_guard_ablation"]
                == ENDPOINT_POSITIVE
            )
            record = {
                "row_id": parent_row["row_id"],
                "sample_slot": int(parent_row["sample_slot"]),
                "dataset_index": int(parent_row["dataset_index"]),
                "epsilon_over_255": float(parent_row["epsilon_over_255"]),
                "clean_prediction": int(parent_row["clean_prediction"]),
                "clean_route": int(parent_row["clean_route"]),
                "parent_prediction_flip_witness": bool(
                    parent_row["attack"]["prediction_flip"]
                ),
                "parent_endpoint_positive": parent_positive,
                "combined_endpoint_positive": combined_positive,
                "branches": branch_records,
            }
            records.append(record)
            _append_json(handle, record)

    complete_gains = sum(
        row["combined_endpoint_positive"] and not row["parent_endpoint_positive"]
        for row in records
    )
    property_improvements = sum(
        branch["strict_property_improvements"]
        for row in records
        for branch in row["branches"]
    )
    if complete_gains:
        outcome = "FINITE_GRID_TRUNCATION_CONFIRMED_AT_ENDPOINT"
        next_stage = "STOP_ATTRIBUTION_ENDPOINT_EXPLAINED"
    elif property_improvements:
        outcome = "FINITE_GRID_CONTRIBUTES_WITHOUT_ENDPOINT_GAIN"
        next_stage = "FIXED_MULTIPLIER_FAMILY_DIAGNOSTIC_REQUIRED"
    else:
        outcome = "FINITE_GRID_NOT_PRIMARY_ON_CLOSEST_RESIDUALS"
        next_stage = "FIXED_MULTIPLIER_FAMILY_DIAGNOSTIC_REQUIRED"
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "parent": {
            "rows": {"path": str(parent_rows_path), "sha256": _sha256(parent_rows_path)},
            "selected_row_ids": selected_ids,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path), "count": len(records)},
        "multiplier_grid": {
            "parent": parent_multipliers,
            "extension": extension_multipliers,
            "combined": parent_multipliers + extension_multipliers,
        },
        "outcome": {
            "classification": outcome,
            "complete_endpoint_gains": complete_gains,
            "strict_property_improvements": property_improvements,
            "next_stage": next_stage,
            "claim_scope": "attribution-only development subset; no prevalence claim",
        },
        "backend_configuration": crown_configuration,
        "runtime_seconds": time.monotonic() - started,
        "gpu": {"free_gib_before": free / 1024**3, "total_gib": total / 1024**3},
        "official_source_clean_after": not bool(_git(source, "status", "--porcelain=v1")),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    print(json.dumps(result["outcome"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
