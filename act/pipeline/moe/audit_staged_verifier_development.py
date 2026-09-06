"""Independent audit of the seed-2 production-path development closure."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_seed2_unresolved_dev_r1.json"
)


def _issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    issues: list[str] = []
    runtime_path = output / "config.json"
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    for path in (runtime_path, results_path, summary_path):
        _issue(issues, path.is_file(), f"missing result artifact {path.name}")
    if issues:
        return {"status": "FAIL", "issue_count": len(issues), "issues": issues}

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    rows = _rows(results_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _issue(
        issues,
        runtime.get("source_config_sha256") == _sha256(config_path),
        "runtime config hash mismatch",
    )
    _issue(issues, runtime.get("config") == config, "runtime config body mismatch")
    frozen_by_rank = {
        int(row["sample_rank"]): row for row in config["selection"]
    }
    result_by_rank = {int(row["sample_rank"]): row for row in rows}
    _issue(issues, len(rows) == len(frozen_by_rank), "unexpected result row count")
    _issue(issues, len(result_by_rank) == len(rows), "duplicate result sample rank")
    _issue(
        issues,
        set(result_by_rank) == set(frozen_by_rank),
        "result ranks differ from frozen selection",
    )

    independently_audited = 0
    independently_replayed = 0
    for rank, row in result_by_rank.items():
        frozen = frozen_by_rank[rank]
        for field in (
            "dataset_index",
            "epsilon",
            "source_status",
            "source_reason",
            "epsilon_source",
        ):
            _issue(
                issues,
                row.get(field) == frozen.get(field),
                f"rank {rank} changed {field}",
            )
        status = row.get("production_status")
        _issue(
            issues,
            status in {"SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"},
            f"rank {rank} has invalid production status",
        )
        package_value = row.get("package")
        if status == "TIMEOUT":
            _issue(
                issues,
                row.get("production_hard_timeout") is True,
                f"rank {rank} timeout lacks hard-timeout marker",
            )
            _issue(
                issues,
                isinstance(row.get("last_progress"), dict),
                f"rank {rank} timeout lacks right-censor stage",
            )
            continue
        _issue(issues, package_value is not None, f"rank {rank} lacks package")
        if package_value is None:
            continue
        package = _inside(Path(package_value), WRITE_ROOT)
        _issue(
            issues,
            package == output / "packages" / f"rank{rank}",
            f"rank {rank} package path mismatch",
        )
        replay = status == "UNSAFE"
        package_audit = audit_evidence_package(package, replay_unsafe=replay)
        independently_audited += 1
        independently_replayed += int(replay)
        _issue(
            issues,
            package_audit["status"] == "PASS",
            f"rank {rank} package audit failed: {package_audit['issues']}",
        )
        evidence_path = package / "evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        _issue(
            issues,
            evidence["request"]["epsilon"] == frozen["epsilon"],
            f"rank {rank} request epsilon mismatch",
        )
        _issue(
            issues,
            evidence.get("execution", {}).get("dataset_index")
            == frozen["dataset_index"],
            f"rank {rank} dataset index mismatch",
        )
        algorithm = evidence.get("algorithm", {})
        _issue(
            issues,
            algorithm.get("boundary_search_executed") is False,
            f"rank {rank} ran boundary search",
        )
        _issue(
            issues,
            algorithm.get("matched_no_support_ablation_executed") is False,
            f"rank {rank} ran no-support control",
        )
        _issue(
            issues,
            algorithm.get("unguarded_accounting_propagation_executed") is False,
            f"rank {rank} ran accounting propagation",
        )
        _issue(
            issues,
            evidence["identity"]["checkpoint"]["sha256"]
            == config["checkpoint_sha256"],
            f"rank {rank} checkpoint identity mismatch",
        )

    counts = Counter(row.get("production_status") for row in rows)
    newly_solved = sum(
        row.get("production_status") in {"SAFE", "UNSAFE"}
        and row.get("source_status") not in {"SAFE", "UNSAFE"}
        for row in rows
    )
    _issue(
        issues,
        summary.get("selected_rows") == len(rows),
        "summary selected count mismatch",
    )
    _issue(
        issues,
        summary.get("production_status_counts") == dict(sorted(counts.items())),
        "summary status counts mismatch",
    )
    _issue(
        issues,
        summary.get("new_complete_outcomes") == newly_solved,
        "summary new-complete count mismatch",
    )
    return {
        "schema_version": 1,
        "classification": config["classification"],
        "config_sha256": _sha256(config_path),
        "runtime_git_head": runtime.get("git_head"),
        "rows": len(rows),
        "production_status_counts": dict(sorted(counts.items())),
        "new_complete_outcomes": newly_solved,
        "packages_independently_audited": independently_audited,
        "unsafe_witnesses_independently_replayed": independently_replayed,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "claim_boundary": (
            "Outcome-selected development engineering only; no prevalence, "
            "speedup, frozen-endpoint revision, or holdout claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = audit(args.config)
    config = json.loads(_inside(args.config, PROJECT_ROOT).read_text(encoding="utf-8"))
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    _write_json(output / "independent_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
