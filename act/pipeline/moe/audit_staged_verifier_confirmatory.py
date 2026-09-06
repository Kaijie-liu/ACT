"""Independently audit the fixed-radius staged-verifier confirmatory cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _git_value,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.freeze_staged_verifier_confirmatory import (
    _indices_from_artifact,
    select_clean_correct,
)
from act.pipeline.moe.run_staged_verifier_confirmatory import (
    DEFAULT_CONFIG,
    _load_jsonl,
    _route_changing,
    _validate_registration,
    summarize_rows,
)
from act.pipeline.moe.train import _load_dataset


def _issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


@torch.no_grad()
def _reconstruct_selection(config: dict[str, Any]) -> list[dict[str, int]]:
    selection = _validate_registration(config)
    excluded: set[int] = set()
    for record in selection["excluded_cohorts"]:
        path = _inside(Path(record["path"]), PROJECT_ROOT)
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"exclusion artifact hash changed: {path}")
        excluded.update(
            _indices_from_artifact(json.loads(path.read_text(encoding="utf-8")))
        )
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.cpu().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    predictions: list[int] = []
    labels: list[int] = []
    for inputs, batch_labels in DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0
    ):
        logits, _ = model.forward_with_routing(inputs)
        predictions.extend(int(value) for value in logits.argmax(dim=1).tolist())
        labels.extend(int(value) for value in batch_labels.tolist())
    return select_clean_correct(
        predictions,
        labels,
        start_index=2000,
        sample_count=100,
        excluded_indices=excluded,
    )


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = _validate_registration(config)
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    issues: list[str] = []
    runtime_path = output / "config.json"
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    for path in (runtime_path, results_path, summary_path):
        _issue(issues, path.is_file(), f"missing runtime artifact: {path}")
    if issues:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "issue_count": len(issues),
            "issues": issues,
        }

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(results_path)
    runner_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _issue(
        issues,
        runtime.get("source_config_sha256") == _sha256(config_path),
        "runtime config hash differs from registration",
    )
    _issue(
        issues,
        runtime.get("selection_manifest_sha256")
        == config["selection_manifest_sha256"],
        "runtime selection hash differs from registration",
    )
    reconstructed = _reconstruct_selection(config)
    _issue(
        issues,
        reconstructed == selection["samples"],
        "clean-correct ordered selection did not reconstruct",
    )
    expected_ranks = [int(row["sample_rank"]) for row in selection["samples"]]
    _issue(
        issues,
        [int(row["sample_rank"]) for row in rows] == expected_ranks,
        "result ranks differ from the frozen 100-row selection",
    )

    package_audits: list[dict[str, Any]] = []
    unsafe_replays = 0
    for expected, row in zip(selection["samples"], rows):
        rank = int(expected["sample_rank"])
        _issue(
            issues,
            int(row["dataset_index"]) == int(expected["dataset_index"]),
            f"dataset index differs at rank {rank}",
        )
        _issue(
            issues,
            int(row["clean_prediction"]) == int(expected["clean_prediction"]),
            f"clean prediction differs at rank {rank}",
        )
        _issue(
            issues,
            float(row["epsilon"]) == float(selection["request"]["epsilon"]),
            f"epsilon differs at rank {rank}",
        )
        if row["status"] == "ERROR":
            issues.append(f"process error at rank {rank}")
        if row.get("package") is None:
            _issue(
                issues,
                row["status"] == "TIMEOUT" and row["outer_hard_timeout"] is True,
                f"rank {rank} lacks package without an outer timeout",
            )
            continue
        package = _inside(Path(row["package"]), WRITE_ROOT)
        _issue(
            issues,
            package
            == output
            / "packages"
            / f"rank{rank}_attempt{int(row['attempt'])}",
            f"package location differs at rank {rank}",
        )
        evidence_path = package / "evidence.json"
        manifest_path = package / "manifest.json"
        _issue(issues, evidence_path.is_file(), f"missing evidence at rank {rank}")
        _issue(issues, manifest_path.is_file(), f"missing manifest at rank {rank}")
        if not evidence_path.is_file() or not manifest_path.is_file():
            continue
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _issue(
            issues,
            manifest["status"] == row["status"]
            and manifest["reason"] == row["reason"],
            f"row verdict differs from package at rank {rank}",
        )
        _issue(
            issues,
            int(evidence["execution"]["dataset_index"])
            == int(expected["dataset_index"]),
            f"evidence dataset index differs at rank {rank}",
        )
        _issue(
            issues,
            float(evidence["request"]["epsilon"])
            == float(selection["request"]["epsilon"]),
            f"evidence epsilon differs at rank {rank}",
        )
        _issue(
            issues,
            evidence["identity"]["checkpoint"]["sha256"]
            == config["checkpoint_sha256"],
            f"evidence checkpoint differs at rank {rank}",
        )
        _issue(
            issues,
            _route_changing(evidence) == row["route_changing"],
            f"route-changing classification differs at rank {rank}",
        )
        package_audit = audit_evidence_package(
            package, replay_unsafe=row["status"] == "UNSAFE"
        )
        package_audits.append(package_audit)
        if row["status"] == "UNSAFE" and package_audit["status"] == "PASS":
            unsafe_replays += 1
        _issue(
            issues,
            package_audit["status"] == "PASS",
            f"evidence package audit failed at rank {rank}",
        )

    independent_summary = summarize_rows(rows, len(selection["samples"]))
    for field in (
        "observed_rows",
        "status_counts",
        "reason_counts",
        "complete_outcomes",
        "safe",
        "route_stable",
        "route_changing",
        "route_changing_safe",
        "tier1_safe",
        "f0_invoked",
        "f0_complete_outcomes",
        "outer_hard_timeouts",
        "solver_reported_timeouts",
    ):
        _issue(
            issues,
            independent_summary[field] == runner_summary.get(field),
            f"runner summary differs for {field}",
        )
    result = {
        "schema_version": 1,
        "audit": "staged_verifier_seed2_fixed2_confirmatory_independent",
        "classification": config["classification"],
        "execution_git_head": runtime.get("execution_git_head"),
        "auditor_git_head": _git_value("rev-parse", "HEAD"),
        "selection_reconstructed": reconstructed == selection["samples"],
        "rows_audited": len(rows),
        "packages_independently_audited": len(package_audits),
        "unsafe_witnesses_independently_replayed": unsafe_replays,
        "summary": independent_summary,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = _inside(args.config, PROJECT_ROOT)
    result = audit(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    path = output / "independent_audit.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    _write_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
