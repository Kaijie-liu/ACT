# ===- act/pipeline/moe/audit_experiment1c.py - Independent 1C Audit -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Independently validate and summarize Experiment 1C result artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WRITE_ROOT = Path("/data1/Kane/MOE")
DEFAULT_RESULT_DIR = Path(
    "/data1/Kane/MOE/ACT/data/moe/results/experiment1c_bal010_r2"
)
MULTIPLIERS = (1.01, 1.05, 1.10)


def _inside(path: Path, root: Path = WRITE_ROOT) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes allowed root {root}: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit output {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return [center - half, center + half]


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    data = [float(value) for value in values]
    if not data:
        return {"median": None, "q1": None, "q3": None, "p90": None}
    quartiles = statistics.quantiles(data, n=4, method="inclusive")
    deciles = statistics.quantiles(data, n=10, method="inclusive")
    return {
        "median": statistics.median(data),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "p90": deciles[8],
    }


def _load_rows(result_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    jsonl = result_dir / "diagnostics.jsonl"
    rows = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (result_dir / "diagnostics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    return rows, csv_rows


def audit(result_dir: Path) -> dict[str, Any]:
    result_dir = _inside(result_dir)
    rows, csv_rows = _load_rows(result_dir)
    runtime = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (result_dir / "selection.json").read_text(encoding="utf-8")
    )["samples"]
    issues: list[str] = []

    keys = [(int(row["sample_rank"]), float(row["epsilon_multiplier"])) for row in rows]
    if len(rows) != 60:
        issues.append(f"expected 60 JSONL rows, found {len(rows)}")
    if len(csv_rows) != len(rows):
        issues.append("CSV and JSONL row counts differ")
    if len(keys) != len(set(keys)):
        issues.append("duplicate sample-rank/multiplier rows")
    selected_ranks = [int(item["sample_rank"]) for item in selection]
    if len(selected_ranks) != 20 or len(set(selected_ranks)) != 20:
        issues.append("selection does not contain 20 distinct ranks")

    clustered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clustered[int(row["sample_rank"])].append(row)
        bracket = row["bracket"]
        epsilon = float(row["epsilon"])
        multiplier = float(row["epsilon_multiplier"])
        if bracket["lower_status"] != "stable" or bracket["upper_status"] != "unstable":
            issues.append(f"rank {row['sample_rank']} has a non-strict bracket")
        if epsilon + 1e-15 < multiplier * float(bracket["upper"]):
            issues.append(f"rank {row['sample_rank']} radius is below bracket upper")
        if row["status"] == "UNSAFE" and not row["full_model_witness_valid"]:
            issues.append(f"rank {row['sample_rank']} has unvalidated UNSAFE")
        if any(
            branch["branch_status"] == "falsified"
            for branch in row.get("branches", [])
        ) and row["status"] == "UNSAFE" and not row["full_model_witness_valid"]:
            issues.append(f"rank {row['sample_rank']} lifted a branch-only witness")

    for rank in selected_ranks:
        actual = sorted(row["epsilon_multiplier"] for row in clustered.get(rank, []))
        if actual != list(MULTIPLIERS):
            issues.append(f"rank {rank} multipliers are {actual}")
    if sorted(clustered) != sorted(selected_ranks):
        issues.append("result sample ranks differ from selection")

    for json_row, csv_row in zip(rows, csv_rows):
        for field in ("sample_rank", "dataset_index", "status", "reason"):
            if str(json_row[field]) != csv_row[field]:
                issues.append(f"CSV/JSONL mismatch for {field}")
                break

    recorded_artifacts = {
        "checkpoint_sha256": Path(runtime["config"]["checkpoint"]),
        "development_results_sha256": Path(
            runtime["config"]["development_results_csv"]
        ),
        "development_sample_indices_sha256": Path(
            runtime["config"]["development_sample_indices"]
        ),
        "source_config_sha256": Path(runtime["source_config"]),
    }
    for key, path in recorded_artifacts.items():
        if _sha256(_inside(path)) != runtime[key]:
            issues.append(f"runtime provenance mismatch for {key}")

    branches = [branch for row in rows for branch in row.get("branches", [])]
    row_status = Counter(row["status"] for row in rows)
    row_reason = Counter(row["reason"] for row in rows)
    branch_reason = Counter(branch["unknown_reason"] for branch in branches)
    closest = [
        row for row in rows if math.isclose(row["epsilon_multiplier"], 1.01)
    ]
    closest_status = Counter(row["status"] for row in closest)
    closest_reason = Counter(row["reason"] for row in closest)

    safe_samples = sorted(
        rank for rank, values in clustered.items() if any(row["status"] == "SAFE" for row in values)
    )
    unsafe_samples = sorted(
        rank for rank, values in clustered.items() if any(row["status"] == "UNSAFE" for row in values)
    )
    unresolved_samples = sorted(
        rank
        for rank, values in clustered.items()
        if any(row["status"] in {"UNKNOWN", "TIMEOUT"} for row in values)
    )
    incomplete_brackets = sorted(
        {
            int(row["sample_rank"])
            for row in rows
            if not row["bracket"]["bisection_complete"]
        }
    )

    fast = sum(branch["support"]["fast_unstable"] for branch in branches)
    lp_eliminated = sum(branch["support"]["lp_eliminated"] for branch in branches)
    milp_eliminated = sum(
        branch["support"]["milp_eliminated"] for branch in branches
    )
    before = sum(branch["expert_relu_binaries_before_guard"] for branch in branches)
    after = sum(branch["expert_relu_binaries_after_guard"] for branch in branches)
    width_ratios = [
        branch["expert_relu_binaries_after_guard"]
        / branch["expert_relu_binaries_before_guard"]
        for branch in branches
        if branch["expert_relu_binaries_before_guard"]
    ]
    reduction_rates = [1.0 - value for value in width_ratios]
    unresolved_rows = row_status["UNKNOWN"] + row_status["TIMEOUT"]
    semantic_incompleteness = (
        row_reason["UNKNOWN_GATE_SUFFICIENCY"]
        + row_reason["UNKNOWN_EXPERT_WITNESS_NOT_LIFTED"]
    )

    result_hashes = {
        path.name: _sha256(path)
        for path in sorted(result_dir.iterdir())
        if path.is_file() and path.name != "independent_audit.json"
    }
    return {
        "independent_of_runner_summary": True,
        "integrity": {
            "issues": issues,
            "jsonl_rows": len(rows),
            "csv_rows": len(csv_rows),
            "samples": len(clustered),
            "duplicate_keys": len(keys) - len(set(keys)),
            "all_unsafe_forward_validated": not any(
                row["status"] == "UNSAFE" and not row["full_model_witness_valid"]
                for row in rows
            ),
            "incomplete_bracket_samples": incomplete_brackets,
        },
        "provenance": {
            "experiment_git_head": runtime["git_head"],
            "checkpoint_sha256": runtime["checkpoint_sha256"],
            "development_results_sha256": runtime["development_results_sha256"],
            "development_sample_indices_sha256": runtime[
                "development_sample_indices_sha256"
            ],
            "artifact_sha256": result_hashes,
        },
        "outcomes": {
            "row_status_counts": dict(row_status),
            "row_reason_counts": dict(row_reason),
            "branch_reason_counts": dict(branch_reason),
            "closest_radius_status_counts": dict(closest_status),
            "closest_radius_reason_counts": dict(closest_reason),
            "unique_safe_samples": safe_samples,
            "unique_safe_rate": len(safe_samples) / len(clustered),
            "unique_safe_wilson_95": _wilson(len(safe_samples), len(clustered)),
            "unsafe_samples": unsafe_samples,
            "unresolved_samples": unresolved_samples,
            "monotonic_inferred_rows": sum(
                row.get("monotonic_inference") is not None for row in rows
            ),
        },
        "guarded_support": {
            "branches": len(branches),
            "fast_unstable": fast,
            "lp_eliminated": lp_eliminated,
            "milp_eliminated": milp_eliminated,
            "corrected_after_milp_unstable": fast - lp_eliminated - milp_eliminated,
            "expert_binaries_before": before,
            "expert_binaries_after": after,
            "expert_binaries_eliminated": before - after,
            "branches_with_elimination": sum(value < 1.0 for value in width_ratios),
            "width_ratio": _quantiles(width_ratios),
            "reduction_rate": _quantiles(reduction_rates),
            "support_seconds": sum(
                branch["support"]["seconds"] for branch in branches
            ),
            "fallback_branches": sum(
                branch["support"]["fallback_sides"] > 0 for branch in branches
            ),
        },
        "decision": {
            "unresolved_rows": unresolved_rows,
            "unresolved_rate": unresolved_rows / len(rows),
            "semantic_incompleteness_rows": semantic_incompleteness,
            "semantic_share_of_unresolved": (
                semantic_incompleteness / unresolved_rows if unresolved_rows else 0.0
            ),
            "weighted_fallback_trigger": bool(
                unresolved_rows
                and semantic_incompleteness / unresolved_rows >= 1.0 / 3.0
            ),
            "confirmatory_holdout_ready": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Experiment 1C artifacts")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--output", default="independent_audit.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_dir = _inside(Path(args.result_dir))
    output = _inside(result_dir / args.output)
    report = audit(result_dir)
    _write_new_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
