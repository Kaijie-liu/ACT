"""Run the preregistered fixed-radius production staged-verifier cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping

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
    / "act/pipeline/moe/configs/staged_verifier_seed2_fixed2_confirmatory_r1.json"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_registration(config: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    selection_path = _inside(Path(config["selection_manifest"]), PROJECT_ROOT)
    staged_config = _inside(Path(config["staged_config"]), PROJECT_ROOT)
    for path, expected in (
        (checkpoint, config["checkpoint_sha256"]),
        (selection_path, config["selection_manifest_sha256"]),
        (staged_config, config["staged_config_sha256"]),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen artifact hash changed: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "FROZEN_BEFORE_VERIFICATION_ENDPOINTS":
        raise RuntimeError("selection is not frozen before endpoints")
    if selection["request"]["boundary_search"] is not False:
        raise RuntimeError("confirmatory selection unexpectedly uses boundary search")
    if selection["request"]["route_instability_prefilter"] is not False:
        raise RuntimeError("confirmatory selection prefilters route instability")
    model_id = config.get("model_id")
    if model_id is not None:
        registered_model = selection.get("models", {}).get(str(model_id))
        if registered_model is None:
            raise RuntimeError(f"selection lacks registered model {model_id}")
        if registered_model.get("checkpoint_sha256") != config["checkpoint_sha256"]:
            raise RuntimeError("selection model checkpoint differs from run config")
    samples = selection["samples"]
    if len(samples) != 100:
        raise RuntimeError("confirmatory selection must contain 100 inputs")
    if [int(row["sample_rank"]) for row in samples] != list(range(100)):
        raise RuntimeError("confirmatory ranks are not complete and ordered")
    return selection


def _route_changing(evidence: Mapping[str, Any]) -> bool | None:
    coverage = evidence.get("route_coverage", {})
    if not (
        coverage.get("coverage_complete") is True
        and coverage.get("route_sets_exact") is True
    ):
        return None
    return len(coverage.get("feasible_route_sets") or []) > 1


def summarize_rows(
    rows: list[dict[str, Any]], expected_rows: int
) -> dict[str, Any]:
    statuses = Counter(row["status"] for row in rows)
    reasons = Counter(row["reason"] for row in rows)
    safe = [row for row in rows if row["status"] == "SAFE"]
    unsafe = [row for row in rows if row["status"] == "UNSAFE"]
    route_known = [row for row in rows if row["route_changing"] is not None]
    route_changing = [row for row in route_known if row["route_changing"]]
    route_changing_safe = [
        row for row in safe if row["route_changing"] is True
    ]
    f0_invoked = [row for row in rows if row.get("f0_invoked") is True]
    f0_complete = [
        row for row in f0_invoked if row["status"] in {"SAFE", "UNSAFE"}
    ]
    audited = [row for row in rows if row.get("evidence_audit_status")]
    complete_run = len(rows) == expected_rows
    all_unsafe_replayed = all(
        row.get("full_model_witness_valid") is True
        and row.get("evidence_audit_status") == "PASS"
        for row in unsafe
    )
    audit_issues = sum(int(row.get("evidence_audit_issue_count") or 0) for row in audited)
    signal = (
        complete_run
        and audit_issues == 0
        and all_unsafe_replayed
        and len(route_changing_safe) >= 1
    )
    complete_times = [
        float(row["verifier_total_seconds"])
        for row in rows
        if row.get("verifier_total_seconds") is not None
    ]
    return {
        "schema_version": 1,
        "classification": "PREREGISTERED_NEW_FIXED_RADIUS_HZ_COHORT",
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "run_complete": complete_run,
        "status_counts": dict(sorted(statuses.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "complete_outcomes": len(safe) + len(unsafe),
        "safe": len(safe),
        "unsafe_full_model_replayed": sum(
            row.get("full_model_witness_valid") is True for row in unsafe
        ),
        "route_coverage_known": len(route_known),
        "route_stable": sum(not row["route_changing"] for row in route_known),
        "route_changing": len(route_changing),
        "route_changing_safe": len(route_changing_safe),
        "route_changing_safe_ranks": [row["sample_rank"] for row in route_changing_safe],
        "tier1_safe": sum(
            row["status"] == "SAFE"
            and row.get("decision_tier") == "TIER1_GATE_ELIMINATION"
            for row in rows
        ),
        "f0_invoked": len(f0_invoked),
        "f0_complete_outcomes": len(f0_complete),
        "f0_safe": sum(row["status"] == "SAFE" for row in f0_invoked),
        "outer_hard_timeouts": sum(row["outer_hard_timeout"] for row in rows),
        "solver_reported_timeouts": sum(
            row["status"] == "TIMEOUT" and not row["outer_hard_timeout"]
            for row in rows
        ),
        "evidence_packages_audited": len(audited),
        "evidence_audit_issues": audit_issues,
        "all_unsafe_full_model_replayed": all_unsafe_replayed,
        "preregistered_replication_signal_met": signal,
        "timing": {
            "complete_package_count": len(complete_times),
            "median_verifier_seconds": (
                statistics.median(complete_times) if complete_times else None
            ),
            "incomplete_rows_are_right_censored": True,
            "speedup_claim_permitted": False,
        },
        "claim_boundary": (
            "The denominator is all 100 selected clean-correct inputs. This is "
            "not certified accuracy, not boundary-adaptive, and not a timing "
            "comparison to historical experiment runners."
        ),
    }


def _row_from_package(
    item: Mapping[str, Any],
    package: Path,
    wall_seconds: float,
    return_code: int,
) -> dict[str, Any]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((package / "evidence.json").read_text(encoding="utf-8"))
    audit = audit_evidence_package(
        package, replay_unsafe=manifest["status"] == "UNSAFE"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"evidence audit failed at rank {item['sample_rank']}")
    verdict = evidence["verdict"]
    tier2 = evidence["tier2"]
    return {
        **dict(item),
        "epsilon": float(evidence["request"]["epsilon"]),
        "status": manifest["status"],
        "reason": manifest["reason"],
        "decision_tier": verdict["decision_tier"],
        "route_changing": _route_changing(evidence),
        "candidate_experts": len(
            evidence["route_coverage"].get("candidate_experts") or []
        ),
        "feasible_route_sets": len(
            evidence["route_coverage"].get("feasible_route_sets") or []
        ),
        "f0_invoked": bool(tier2["invoked"]),
        "full_model_witness_valid": bool(verdict["full_model_witness_valid"]),
        "verifier_total_seconds": float(evidence["timing"]["total_seconds"]),
        "runner_wall_seconds": wall_seconds,
        "outer_hard_timeout": False,
        "return_code": return_code,
        "package": str(package),
        "package_manifest_sha256": _sha256(package / "manifest.json"),
        "evidence_audit_status": audit["status"],
        "evidence_audit_issue_count": audit["issue_count"],
    }


def run(config_path: Path, *, resume: bool = False) -> Path:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if Path(sys.executable).resolve() != Path(config["python"]).resolve():
        raise RuntimeError("confirmatory run requires the frozen act-py312 Python")
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("confirmatory run requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("confirmatory run requires a clean worktree")
    selection = _validate_registration(config)
    samples = selection["samples"]
    epsilon = float(selection["request"]["epsilon"])
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    runtime_path = output / "config.json"
    results_path = output / "results.jsonl"
    if output.exists() and not resume:
        raise RuntimeError(f"refusing to overwrite {output}")
    if not output.exists():
        output.mkdir(parents=True)
        for name in ("packages", "progress", "logs"):
            (output / name).mkdir()
        _write_json(
            runtime_path,
            {
                "schema_version": 1,
                "source_config": str(config_path),
                "source_config_sha256": _sha256(config_path),
                "selection_manifest_sha256": config["selection_manifest_sha256"],
                "git_branch": _git_value("branch", "--show-current"),
                "execution_git_head": _git_value("rev-parse", "HEAD"),
                "config": config,
            },
        )
    else:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime["source_config_sha256"] != _sha256(config_path):
            raise RuntimeError("resume config differs from frozen runtime")

    rows = _load_jsonl(results_path)
    if [int(row["sample_rank"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("existing results are not an ordered prefix")
    environment = os.environ.copy()
    environment["ACT_TORCHVISION_DATA_ROOT"] = str(
        PROJECT_ROOT / "data/torchvision"
    )
    with results_path.open("a", encoding="utf-8") as results:
        for item in samples[len(rows) :]:
            rank = int(item["sample_rank"])
            attempt = 1
            while (output / "logs" / f"rank{rank}_attempt{attempt}.log").exists():
                attempt += 1
            package = output / "packages" / f"rank{rank}_attempt{attempt}"
            progress = output / "progress" / f"rank{rank}_attempt{attempt}.json"
            log_path = output / "logs" / f"rank{rank}_attempt{attempt}.log"
            command = [
                str(Path(config["python"])),
                "-m",
                "act.pipeline.moe.staged_verifier",
                "--checkpoint",
                config["checkpoint"],
                "--dataset-index",
                str(item["dataset_index"]),
                "--epsilon",
                repr(epsilon),
                "--config",
                config["staged_config"],
                "--output-dir",
                str(package),
                "--progress-path",
                str(progress),
            ]
            started = time.monotonic()
            return_code: int | None = None
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
            last_progress = (
                json.loads(progress.read_text(encoding="utf-8"))
                if progress.is_file()
                else None
            )
            if not timed_out and return_code == 0 and package.is_dir():
                row = _row_from_package(item, package, wall, return_code)
                row["attempt"] = attempt
            else:
                row = {
                    **dict(item),
                    "attempt": attempt,
                    "epsilon": epsilon,
                    "status": "TIMEOUT" if timed_out else "ERROR",
                    "reason": (
                        "INSTANCE_HARD_DEADLINE" if timed_out else "PROCESS_ERROR"
                    ),
                    "decision_tier": None,
                    "route_changing": None,
                    "candidate_experts": None,
                    "feasible_route_sets": None,
                    "f0_invoked": (
                        last_progress is not None
                        and str(last_progress.get("active_stage", "")).startswith(
                            "TIER2_F0"
                        )
                    ),
                    "full_model_witness_valid": False,
                    "verifier_total_seconds": None,
                    "runner_wall_seconds": wall,
                    "outer_hard_timeout": timed_out,
                    "return_code": return_code,
                    "package": None,
                    "package_manifest_sha256": None,
                    "evidence_audit_status": None,
                    "evidence_audit_issue_count": None,
                    "last_progress": last_progress,
                }
            rows.append(row)
            results.write(json.dumps(row, sort_keys=True) + "\n")
            results.flush()
            os.fsync(results.fileno())
            _write_json(output / "summary.partial.json", summarize_rows(rows, len(samples)))
    _write_json(output / "summary.json", summarize_rows(rows, len(samples)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(args.config, resume=args.resume))


if __name__ == "__main__":
    main()
