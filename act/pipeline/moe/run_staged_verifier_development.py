"""Run the frozen production-path development closure on seed-2 R1 residuals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _git_value,
    _inside,
    _sha256,
    _write_json,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_seed2_unresolved_dev_r1.json"
)
SELECTED_SOURCE_REASONS = {
    "UNKNOWN_WEIGHTED_SOLVER_LIMIT",
    "TIMEOUT_EXPERT_SOLVE",
    "INSTANCE_HARD_DEADLINE",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_selection(config: dict[str, Any]) -> None:
    source_path = _inside(Path(config["source_results"]), WRITE_ROOT)
    index_path = _inside(Path(config["source_sample_indices"]), WRITE_ROOT)
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    staged_config = _inside(Path(config["staged_config"]), PROJECT_ROOT)
    expected_hashes = {
        source_path: config["source_results_sha256"],
        index_path: config["source_sample_indices_sha256"],
        checkpoint: config["checkpoint_sha256"],
        staged_config: config["staged_config_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen artifact hash changed: {path}")

    rows = _load_jsonl(source_path)
    by_rank = {int(row["sample_rank"]): row for row in rows}
    selected_ranks = {
        int(row["sample_rank"])
        for row in rows
        if row["reason"] in SELECTED_SOURCE_REASONS
    }
    frozen = config["selection"]
    if {int(row["sample_rank"]) for row in frozen} != selected_ranks:
        raise RuntimeError("selection is not the complete frozen residual set")
    index_rows = json.loads(index_path.read_text(encoding="utf-8"))["samples"]
    indices = {
        int(row["sample_rank"]): int(row["dataset_index"])
        for row in index_rows
    }
    boundary_root = source_path.parent
    for item in frozen:
        rank = int(item["sample_rank"])
        source = by_rank[rank]
        if int(source["dataset_index"]) != int(item["dataset_index"]):
            raise RuntimeError(f"dataset index changed at rank {rank}")
        if int(indices[rank]) != int(item["dataset_index"]):
            raise RuntimeError(f"sample-index manifest changed at rank {rank}")
        if source["status"] != item["source_status"]:
            raise RuntimeError(f"source status changed at rank {rank}")
        if source["reason"] != item["source_reason"]:
            raise RuntimeError(f"source reason changed at rank {rank}")
        if item["epsilon_source"] == "completed_row":
            observed = source.get("epsilon")
        else:
            progress_path = boundary_root / source["partial_work_dir"] / "progress.json"
            if _sha256(progress_path) != item["progress_sha256"]:
                raise RuntimeError(f"partial progress changed at rank {rank}")
            observed = json.loads(progress_path.read_text(encoding="utf-8"))[
                "epsilon"
            ]
        if float(observed) != float(item["epsilon"]):
            raise RuntimeError(f"epsilon changed at rank {rank}")


def _summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter(row["production_status"] for row in rows)
    reasons = Counter(row["production_reason"] for row in rows)
    newly_solved = [
        row
        for row in rows
        if row["production_status"] in {"SAFE", "UNSAFE"}
        and row["source_status"] not in {"SAFE", "UNSAFE"}
    ]
    completed = [row for row in rows if row["production_status"] != "TIMEOUT"]
    audits = [row for row in rows if row.get("evidence_audit_status") is not None]
    required = int(
        config["decision_rule"]["new_complete_outcomes_required_to_freeze_a_new_cohort"]
    )
    return {
        "schema_version": 1,
        "classification": config["classification"],
        "selected_rows": len(rows),
        "production_status_counts": dict(sorted(statuses.items())),
        "production_reason_counts": dict(sorted(reasons.items())),
        "new_complete_outcomes": len(newly_solved),
        "new_safe": sum(row["production_status"] == "SAFE" for row in newly_solved),
        "new_unsafe": sum(
            row["production_status"] == "UNSAFE" for row in newly_solved
        ),
        "new_complete_sample_ranks": [row["sample_rank"] for row in newly_solved],
        "completed_requests": len(completed),
        "hard_timeouts": sum(row["production_status"] == "TIMEOUT" for row in rows),
        "evidence_packages_audited": len(audits),
        "evidence_audit_issues": sum(
            int(row.get("evidence_audit_issue_count", 0)) for row in audits
        ),
        "primary_signal_met": len(newly_solved) >= required,
        "next_decision": (
            "ELIGIBLE_TO_PREREGISTER_SEPARATE_NEW_HZ_COHORT"
            if len(newly_solved) >= required
            else "NO_NEW_COHORT_KEEP_ENGINEERING_ONLY"
        ),
        "runtime_interpretation": (
            "New wall times and frozen source times are non-interleaved and "
            "descriptive; no speedup claim is permitted."
        ),
        "endpoint_interpretation": (
            "Outcome-selected development closure only; never overwrite or "
            "pool into seed-2 R1."
        ),
    }


def run(config_path: Path) -> Path:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if Path(sys.executable).resolve() != Path(config["python"]).resolve():
        raise RuntimeError("development run requires the frozen act-py312 Python")
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("development run requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("development run requires a clean worktree")
    _validate_selection(config)
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "packages").mkdir()
    (output / "progress").mkdir()
    (output / "logs").mkdir()
    runtime = {
        "schema_version": 1,
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "git_branch": _git_value("branch", "--show-current"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "config": config,
    }
    _write_json(output / "config.json", runtime)

    rows: list[dict[str, Any]] = []
    results_path = output / "results.jsonl"
    environment = os.environ.copy()
    environment["ACT_TORCHVISION_DATA_ROOT"] = str(
        PROJECT_ROOT / "data/torchvision"
    )
    with results_path.open("x", encoding="utf-8") as results:
        for item in config["selection"]:
            rank = int(item["sample_rank"])
            package = output / "packages" / f"rank{rank}"
            progress = output / "progress" / f"rank{rank}.json"
            log_path = output / "logs" / f"rank{rank}.log"
            command = [
                str(Path(config["python"])),
                "-m",
                "act.pipeline.moe.staged_verifier",
                "--checkpoint",
                config["checkpoint"],
                "--dataset-index",
                str(item["dataset_index"]),
                "--epsilon",
                repr(float(item["epsilon"])),
                "--config",
                config["staged_config"],
                "--output-dir",
                str(package),
                "--progress-path",
                str(progress),
            ]
            started = time.monotonic()
            return_code = None
            timed_out = False
            with log_path.open("x", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=float(config["instance_hard_timeout_seconds"]),
                        check=False,
                    )
                    return_code = int(completed.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
            wall = time.monotonic() - started
            progress_value = (
                json.loads(progress.read_text(encoding="utf-8"))
                if progress.is_file()
                else None
            )
            row = {
                **item,
                "production_wall_seconds": wall,
                "production_return_code": return_code,
                "production_hard_timeout": timed_out,
                "last_progress": progress_value,
                "package": str(package) if package.is_dir() else None,
                "package_manifest_sha256": None,
                "production_status": "TIMEOUT" if timed_out else "ERROR",
                "production_reason": (
                    "INSTANCE_HARD_DEADLINE" if timed_out else "PROCESS_ERROR"
                ),
                "evidence_audit_status": None,
                "evidence_audit_issue_count": None,
            }
            if not timed_out and return_code == 0 and package.is_dir():
                manifest_path = package / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                audit = audit_evidence_package(
                    package, replay_unsafe=manifest["status"] == "UNSAFE"
                )
                row.update(
                    {
                        "package_manifest_sha256": _sha256(manifest_path),
                        "production_status": manifest["status"],
                        "production_reason": manifest["reason"],
                        "evidence_audit_status": audit["status"],
                        "evidence_audit_issue_count": audit["issue_count"],
                        "evidence_audit": audit,
                    }
                )
                if audit["status"] != "PASS":
                    raise RuntimeError(f"evidence audit failed at rank {rank}")
            rows.append(row)
            results.write(json.dumps(row, sort_keys=True) + "\n")
            results.flush()
            os.fsync(results.fileno())
    summary = _summary(rows, config)
    _write_json(output / "summary.json", summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output = run(args.config)
    print(output)


if __name__ == "__main__":
    main()
