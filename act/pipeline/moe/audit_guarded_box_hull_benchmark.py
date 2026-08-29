"""Independent artifact audit for the paired guarded-box-hull benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.benchmark_guarded_box_hull import (
    _backend_order,
    _bounds_sha256,
    _max_abs_bound_difference,
)
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json
from act.pipeline.moe.experiment1d import _load_frozen_selection, _row_context
from act.pipeline.moe.train import _load_dataset


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact_issues(
    branch: dict[str, Any], result_dir: Path
) -> tuple[list[str], float | None]:
    issues: list[str] = []
    relative = branch.get("bounds_artifact")
    expected_hash = branch.get("bounds_artifact_sha256")
    if not relative or not expected_hash:
        return ["missing bound artifact identity"], None
    path = _inside(result_dir / relative, result_dir)
    if not path.exists():
        return ["bound artifact is missing"], None
    if _sha256(path) != expected_hash:
        issues.append("bound artifact hash mismatch")
    try:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {
                "highspy": (
                    np.asarray(payload["highspy_lower"], dtype=np.float64),
                    np.asarray(payload["highspy_upper"], dtype=np.float64),
                ),
                "scipy": (
                    np.asarray(payload["scipy_lower"], dtype=np.float64),
                    np.asarray(payload["scipy_upper"], dtype=np.float64),
                ),
            }
    except Exception as error:
        return [*issues, f"bound artifact cannot be loaded: {type(error).__name__}"], None
    for backend, (lower, upper) in arrays.items():
        if lower.shape != upper.shape or lower.ndim != 1:
            issues.append(f"{backend} bound shape is invalid")
            continue
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            issues.append(f"{backend} bounds are non-finite")
        if np.any(lower > upper):
            issues.append(f"{backend} lower bound exceeds upper bound")
        if _bounds_sha256(lower, upper) != branch[backend].get("bounds_sha256"):
            issues.append(f"{backend} recorded bound hash mismatch")
    difference = _max_abs_bound_difference(arrays["highspy"], arrays["scipy"])
    recorded = branch.get("bound_max_abs_diff")
    if difference is None or recorded is None or not np.isclose(
        difference, float(recorded), rtol=0.0, atol=0.0
    ):
        issues.append("recorded maximum bound difference mismatch")
    return issues, difference


def audit(config_path: Path, result_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    result_dir = _inside(result_dir, WRITE_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    rows = _jsonl(result_dir / "rows.jsonl")
    branches = _jsonl(result_dir / "branches.jsonl")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    selection = _load_frozen_selection(config)
    issues: list[dict[str, Any]] = []

    def add(scope: str, detail: str, identity: str | int | None = None) -> None:
        issues.append({"scope": scope, "detail": detail, "identity": identity})

    if runtime.get("source_config_sha256") != _sha256(config_path):
        add("provenance", "source config hash mismatch")
    if runtime.get("config") != config:
        add("provenance", "embedded runtime config differs")
    checkpoint = Path(config["checkpoint"])
    if runtime.get("checkpoint_sha256") != _sha256(checkpoint):
        add("provenance", "checkpoint hash mismatch")
    expected_ranks = [int(item["sample_rank"]) for item in selection]
    actual_ranks = [int(row["sample_rank"]) for row in rows]
    if actual_ranks != expected_ranks:
        add("rows", "row order differs from frozen selection")
    if len(set(actual_ranks)) != len(actual_ranks):
        add("rows", "duplicate row rank")

    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    expected_pairs: dict[int, set[tuple[int, int]]] = {}
    for item in selection:
        context = _row_context(item, model, dataset, config)
        rank = int(context["rank"])
        if not context["route_sets"].exact:
            add("routes", "audited route-set enumeration is not exact", rank)
        expected_pairs[rank] = {
            tuple(sorted(int(value) for value in pair))
            for pair in context["route_sets"].feasible
        }

    branch_ids: set[str] = set()
    actual_pairs: dict[int, set[tuple[int, int]]] = {
        rank: set() for rank in expected_ranks
    }
    recomputed_differences: list[float] = []
    for branch in branches:
        branch_id = str(branch.get("branch_id"))
        rank = int(branch["sample_rank"])
        pair = tuple(sorted(int(value) for value in branch["route_pair"]))
        if branch_id in branch_ids:
            add("branches", "duplicate branch identity", branch_id)
        branch_ids.add(branch_id)
        actual_pairs.setdefault(rank, set()).add(pair)
        if tuple(branch.get("backend_order", ())) != _backend_order(rank):
            add("order", "backend order differs from frozen parity rule", branch_id)
        artifact_issues, difference = _artifact_issues(branch, result_dir)
        for detail in artifact_issues:
            add("bounds", detail, branch_id)
        if difference is not None:
            recomputed_differences.append(difference)
        paired_complete = bool(branch["highspy"].get("complete")) and bool(
            branch["scipy"].get("complete")
        )
        if bool(branch.get("paired_complete")) != paired_complete:
            add("branches", "paired_complete flag mismatch", branch_id)
        for backend in ("highspy", "scipy"):
            record = branch[backend]
            if record.get("error") and record.get("complete"):
                add("backend", f"{backend} is complete despite runner error", branch_id)
    for rank, pairs in expected_pairs.items():
        if actual_pairs.get(rank, set()) != pairs:
            add("routes", "recorded branches differ from exact feasible pairs", rank)

    row_by_rank = {int(row["sample_rank"]): row for row in rows}
    for rank in expected_ranks:
        row = row_by_rank.get(rank)
        if row is None:
            continue
        ids = {
            str(branch["branch_id"])
            for branch in branches
            if int(branch["sample_rank"]) == rank
        }
        if set(row.get("branch_ids", [])) != ids:
            add("rows", "row branch identities mismatch", rank)
        if int(row.get("route_branch_count", -1)) != len(ids):
            add("rows", "row branch count mismatch", rank)

    all_paired_complete = (
        len(rows) == int(config["expected_rows"])
        and not any(row.get("error") for row in rows)
        and len(branches) == sum(len(value) for value in expected_pairs.values())
        and all(bool(branch.get("paired_complete")) for branch in branches)
    )
    tolerance = float(config["bound_comparison_tolerance"])
    disagreements = sum(value > tolerance for value in recomputed_differences)
    speed_eligible = bool(all_paired_complete and not disagreements and not issues)
    if bool(summary.get("speed_conclusion", {}).get("eligible")) != speed_eligible:
        add("summary", "speed conclusion eligibility mismatch")
    if int(summary.get("route_branches", -1)) != len(branches):
        add("summary", "route branch count mismatch")
    expected_order_counts = dict(
        sorted(Counter("->".join(branch["backend_order"]) for branch in branches).items())
    )
    if summary.get("backend_order_counts") != expected_order_counts:
        add("summary", "backend order counts mismatch")
    return {
        "schema_version": 1,
        "scope": "independent_guarded_box_hull_engineering_audit",
        "issues": issues,
        "issue_count": len(issues),
        "rows": len(rows),
        "route_branches": len(branches),
        "bound_artifacts_replayed": len(branches),
        "all_exact_feasible_pairs_covered": all(
            actual_pairs.get(rank, set()) == pairs for rank, pairs in expected_pairs.items()
        ),
        "all_paired_complete": all_paired_complete,
        "bound_disagreements": disagreements,
        "speed_conclusion_eligible": speed_eligible,
        "artifact_sha256": {
            "source_config": _sha256(config_path),
            "runtime_config": _sha256(result_dir / "config.json"),
            "rows": _sha256(result_dir / "rows.jsonl"),
            "branches": _sha256(result_dir / "branches.jsonl"),
            "summary": _sha256(result_dir / "summary.json"),
        },
        "original_confirmatory_overall_solved_rate_immutable": 0.56,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.result_dir)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
