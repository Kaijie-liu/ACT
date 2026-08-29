# ===- act/pipeline/moe/audit_experiment1d.py - Closure Audit ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Independent audit for Experiment 1D applicable-unresolved closure."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1d import DEFAULT_CONFIG, _load_frozen_selection
from act.pipeline.moe.train import _load_dataset


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _audit_safe_f0(row: dict[str, Any], issues: list[str], tolerance: float) -> None:
    rank = row["sample_rank"]
    f0 = row.get("f0") or {}
    feasible = {tuple(pair) for pair in f0.get("feasible_pairs", [])}
    evaluated = {tuple(pair["pair"]) for pair in f0.get("pairs", [])}
    if not feasible or feasible != evaluated:
        issues.append(f"rank {rank}: SAFE F0 lacks complete pair coverage")
    for pair in f0.get("pairs", []):
        properties = pair.get("property_rows", [])
        if pair.get("status") != "SAFE" or len(properties) != 9:
            issues.append(f"rank {rank}: SAFE pair is incomplete")
            continue
        for prop in properties:
            if prop.get("status") != "SAFE":
                issues.append(f"rank {rank}: SAFE pair has unresolved property")
            if prop.get("minimum") is None or float(prop["minimum"]) <= tolerance:
                issues.append(f"rank {rank}: SAFE property lacks positive bound")
            if prop.get("solver_status") != 0:
                issues.append(f"rank {rank}: SAFE property lacks optimal status")
            if prop.get("solver_bound_kind") not in {"mip_dual_bound", "lp_status0_optimum"}:
                issues.append(f"rank {rank}: SAFE property lacks certified bound kind")


def audit(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    audit_path = output_dir / "independent_audit.json"
    if audit_path.exists():
        raise RuntimeError(f"refusing to overwrite {audit_path}")
    with (output_dir / "config.json").open(encoding="utf-8") as handle:
        runtime = json.load(handle)
    results = _load_jsonl(output_dir / "results.jsonl")
    selection = _load_frozen_selection(config)
    by_rank = {int(row["sample_rank"]): row for row in selection}
    issues: list[str] = []
    expected_ranks = [int(row["sample_rank"]) for row in selection]
    actual_ranks = [int(row["sample_rank"]) for row in results]
    if actual_ranks != expected_ranks or len(results) != int(config["expected_rows"]):
        issues.append("closure rows differ from the frozen ordered selection")
    if runtime.get("source_config_sha256") != _sha256(config_path):
        issues.append("runtime config hash mismatch")
    if runtime.get("selection_manifest_sha256") != _sha256(Path(config["selection_manifest"])):
        issues.append("runtime selection manifest hash mismatch")
    if runtime.get("checkpoint_sha256") != _sha256(Path(config["checkpoint"])):
        issues.append("runtime checkpoint hash mismatch")
    if config.get("numerical_safety") != hz_numerical_policy_manifest():
        issues.append("numerical SAFE policy changed")

    model, payload = load_output_moe_checkpoint(Path(config["checkpoint"]), map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    replayed = 0
    for row in results:
        rank = int(row["sample_rank"])
        frozen = by_rank.get(rank)
        if frozen is None:
            continue
        if int(row["dataset_index"]) != int(frozen["dataset_index"]):
            issues.append(f"rank {rank}: dataset index changed")
        if not math.isclose(float(row["epsilon"]), float(frozen["epsilon"]), rel_tol=0.0, abs_tol=1e-15):
            issues.append(f"rank {rank}: frozen radius changed")
        if row.get("category") != frozen["category"] or row.get("reuse_mode") != frozen["reuse_mode"]:
            issues.append(f"rank {rank}: category/reuse mode changed")
        if row.get("parent_status") != frozen["parent"]["status"] or row.get("parent_reason") != frozen["parent"]["reason"]:
            issues.append(f"rank {rank}: parent status/reason changed")
        if row.get("deadline_enforced") is not True:
            issues.append(f"rank {rank}: 900-second deadline not enforced")
        if row.get("status") != "TIMEOUT" and float(row.get("total_seconds", 0.0)) > float(config["instance_timeout_seconds"]):
            issues.append(f"rank {rank}: completed after hard deadline")
        checkpoint = output_dir / "row_work" / f"rank{rank}" / "checkpoint_300.json"
        if not checkpoint.exists() or row.get("checkpoint_recorded") is not True:
            issues.append(f"rank {rank}: 300-second checkpoint missing")
        if row.get("status") == "ERROR" or row.get("error"):
            issues.append(f"rank {rank}: explicit runner error")
        if "NUMERICAL" in str(row.get("reason", "")):
            issues.append(f"rank {rank}: numerical fallback/result")
        if row.get("status") == "SAFE":
            if row.get("reason") == "SAFE_WEIGHTED_RANGE":
                _audit_safe_f0(row, issues, float(config["f0"]["safety_tolerance"]))
            elif row.get("reason") == "SAFE_GATE_ELIMINATION":
                gate = row.get("gate") or {}
                if not gate.get("branches") or any(branch.get("unknown_reason") != "SAFE_PROVED" for branch in gate["branches"]):
                    issues.append(f"rank {rank}: gate SAFE has unresolved expert")
            else:
                issues.append(f"rank {rank}: unregistered SAFE reason")
        if row.get("status") == "UNSAFE":
            if not row.get("full_model_witness_valid"):
                issues.append(f"rank {rank}: UNSAFE lacks full-forward flag")
                continue
            witness = row.get("witness_path")
            path = _inside(output_dir / witness, output_dir) if witness else None
            if path is None or not path.exists() or _sha256(path) != row.get("witness_sha256"):
                issues.append(f"rank {rank}: witness path/hash mismatch")
                continue
            saved = torch.load(path, map_location="cpu", weights_only=False)
            image, _ = dataset[int(row["dataset_index"])]
            x = image.unsqueeze(0).double()
            epsilon = float(row["epsilon"])
            checked = _forward_validate(
                model, saved["input"], lower=(x - epsilon).clamp(0, 1),
                upper=(x + epsilon).clamp(0, 1),
                clean_prediction=int(row["clean_prediction"]),
            )
            if not checked["valid"]:
                issues.append(f"rank {rank}: replayed witness invalid")
            replayed += int(checked["valid"])

    status = Counter(row.get("status") for row in results)
    newly_solved = status["SAFE"] + status["UNSAFE"]
    coverage = (int(config["preregistered_unlock"]["parent_solved_applicable_rows"]) + newly_solved) / int(config["preregistered_unlock"]["parent_applicable_rows"])
    thresholds = config["preregistered_unlock"]
    conditions = {
        "all_20_rows_run": len(results) == int(config["expected_rows"]),
        "independent_audit_zero_issues": len(issues) == 0,
        "all_new_unsafe_replayed": replayed == status["UNSAFE"],
        "at_least_5_newly_solved": newly_solved >= int(thresholds["minimum_newly_solved_applicable_rows"]),
        "applicable_coverage_at_least_80_percent": coverage >= float(thresholds["minimum_applicable_coverage"]),
        "no_silent_numerical_fallback": not any(row.get("status") == "ERROR" or "NUMERICAL" in str(row.get("reason", "")) for row in results),
    }
    result = {
        "issues": issues, "issue_count": len(issues), "rows": len(results),
        "status_counts": dict(status), "newly_solved_applicable_rows": newly_solved,
        "new_unsafe_replayed": replayed,
        "parent_confirmatory_overall_solved_rate_immutable": 0.56,
        "boundary_applicability": "76/100",
        "parent_conditional_coverage": 56 / 76,
        "closure_conditional_coverage": coverage,
        "conditions": conditions,
        "public_baseline_unlocked": all(conditions.values()),
    }
    _write_json(audit_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
