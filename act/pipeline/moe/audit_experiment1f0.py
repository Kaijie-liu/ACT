# ===- act/pipeline/moe/audit_experiment1f0.py - F0 Audit ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Independently audit the frozen Experiment 1F0 rows and concrete witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from act.back_end.moe import (
    SAFE_WEIGHTED_RANGE,
    UNKNOWN_WEIGHTED_NUMERICAL,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    load_output_moe_checkpoint,
)
from act.pipeline.moe.experiment1 import (
    WRITE_ROOT,
    _forward_validate,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1f0 import DEFAULT_CONFIG
from act.pipeline.moe.train import _load_dataset


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_rows(path: Path) -> dict[int, tuple[dict[str, Any], str]]:
    result: dict[int, tuple[dict[str, Any], str]] = {}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            result[line_number] = (json.loads(raw), _hash_bytes(raw))
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def audit(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    parent_path = _inside(Path(config["parent_diagnostics_jsonl"]), WRITE_ROOT)
    result_path = output_dir / "results.jsonl"
    selection_path = output_dir / "selection.json"
    summary_path = output_dir / "summary.json"
    runtime_path = output_dir / "config.json"
    audit_path = output_dir / "independent_audit.json"
    if audit_path.exists():
        raise RuntimeError(f"refusing to overwrite {audit_path}")

    issues: list[str] = []
    expected_parent_hash = str(config["parent_diagnostics_sha256"])
    actual_parent_hash = _sha256(parent_path)
    _issue(
        issues,
        actual_parent_hash == expected_parent_hash,
        "parent diagnostics SHA-256 mismatch",
    )
    rows = _load_jsonl(result_path)
    source = _source_rows(parent_path)
    selection = json.load(selection_path.open(encoding="utf-8"))
    runner_summary = json.load(summary_path.open(encoding="utf-8"))
    runtime = json.load(runtime_path.open(encoding="utf-8"))
    expected_rows = int(config["expected_rows"])
    _issue(issues, len(rows) == expected_rows, "result row count mismatch")
    _issue(
        issues,
        len(selection["rows"]) == expected_rows,
        "selection row count mismatch",
    )
    ids = [row["parent_row_id"] for row in rows]
    _issue(issues, len(ids) == len(set(ids)), "duplicate parent row IDs")
    selected_ids = {row["parent_row_id"] for row in selection["rows"]}
    _issue(issues, set(ids) == selected_ids, "result/selection parent IDs differ")
    eligible = set(config["eligible_parent_reasons"])
    allowed_reasons = {
        SAFE_WEIGHTED_RANGE,
        UNSAFE_FULL_FORWARD_FALLBACK,
        UNKNOWN_WEIGHTED_RELAXATION,
        UNKNOWN_WEIGHTED_SOLVER_LIMIT,
        UNKNOWN_WEIGHTED_NUMERICAL,
    }

    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    witness_rows = 0
    safe_rows = 0
    for row in rows:
        prefix = row["parent_row_id"][:12]
        line_number = int(row["parent_line_number"])
        _issue(issues, line_number in source, f"{prefix}: missing parent line")
        if line_number not in source:
            continue
        parent, parent_row_hash = source[line_number]
        expected_id = _hash_bytes(
            f"{expected_parent_hash}:{line_number}:{parent_row_hash}".encode()
        )
        _issue(
            issues,
            row["parent_row_id"] == expected_id,
            f"{prefix}: parent ID mismatch",
        )
        _issue(
            issues,
            row["parent_row_sha256"] == parent_row_hash,
            f"{prefix}: parent row hash mismatch",
        )
        _issue(
            issues,
            row["parent_artifact_sha256"] == expected_parent_hash,
            f"{prefix}: parent artifact hash mismatch",
        )
        _issue(
            issues,
            parent["reason"] in eligible and row["parent_reason"] == parent["reason"],
            f"{prefix}: ineligible or changed parent reason",
        )
        for field in ("sample_rank", "dataset_index", "epsilon", "epsilon_multiplier"):
            _issue(
                issues,
                row[field] == parent[field],
                f"{prefix}: parent field {field} changed",
            )
        _issue(
            issues,
            row["reason"] in allowed_reasons,
            f"{prefix}: unregistered F0 reason",
        )

        feasible = {tuple(pair) for pair in row["feasible_pairs"]}
        evaluated = {tuple(pair["pair"]) for pair in row["pairs"]}
        _issue(
            issues,
            evaluated.issubset(feasible),
            f"{prefix}: evaluated pair outside exact feasible set",
        )
        if row["status"] == "SAFE":
            safe_rows += 1
            _issue(
                issues,
                row["reason"] == SAFE_WEIGHTED_RANGE,
                f"{prefix}: SAFE has wrong reason",
            )
            _issue(
                issues,
                evaluated == feasible and bool(feasible),
                f"{prefix}: SAFE did not cover every feasible pair",
            )
            for pair in row["pairs"]:
                _issue(
                    issues,
                    pair["status"] == "SAFE"
                    and pair["reason"] == SAFE_WEIGHTED_RANGE,
                    f"{prefix}: SAFE row contains unresolved pair",
                )
                _issue(
                    issues,
                    len(pair["property_rows"]) == 9
                    and all(
                        item["status"] == "SAFE"
                        and item["reason"] == SAFE_WEIGHTED_RANGE
                        for item in pair["property_rows"]
                    ),
                    f"{prefix}: SAFE pair did not prove all nine margins",
                )
        elif row["status"] == "UNSAFE":
            witness_rows += 1
            _issue(
                issues,
                row["reason"] == UNSAFE_FULL_FORWARD_FALLBACK
                and row["full_model_witness_valid"] is True,
                f"{prefix}: UNSAFE lacks full-forward status",
            )
            reported_witness = row.get("witness_path")
            witness_path = (
                _inside(output_dir / reported_witness, output_dir)
                if isinstance(reported_witness, str) and reported_witness
                else None
            )
            _issue(
                issues,
                witness_path is not None
                and witness_path.exists()
                and _sha256(witness_path) == row["witness_sha256"],
                f"{prefix}: witness file/hash mismatch",
            )
            if witness_path is not None and witness_path.exists():
                payload = torch.load(
                    witness_path, map_location="cpu", weights_only=False
                )
                image, _ = dataset[int(row["dataset_index"])]
                x = image.unsqueeze(0).double()
                epsilon = float(row["epsilon"])
                checked = _forward_validate(
                    model,
                    payload["input"],
                    lower=(x - epsilon).clamp(0, 1),
                    upper=(x + epsilon).clamp(0, 1),
                    clean_prediction=int(row["clean_prediction"]),
                )
                _issue(
                    issues,
                    checked["valid"]
                    and checked["prediction"] == row["counterexample_prediction"]
                    and checked["topk_set"] == row["counterexample_topk_set"],
                    f"{prefix}: independently replayed witness is invalid",
                )
        else:
            _issue(
                issues,
                row["status"] == "UNKNOWN"
                and row["reason"].startswith("UNKNOWN_WEIGHTED_"),
                f"{prefix}: invalid unresolved status semantics",
            )
            _issue(
                issues,
                row["full_model_witness_valid"] is False,
                f"{prefix}: unresolved row hides a validated witness",
            )

    status_counts = dict(Counter(row["status"] for row in rows))
    reason_counts = dict(Counter(row["reason"] for row in rows))
    resolved = status_counts.get("SAFE", 0) + status_counts.get("UNSAFE", 0)
    safe_sample_ranks = sorted(
        {int(row["sample_rank"]) for row in rows if row["status"] == "SAFE"}
    )
    recomputed = {
        "rows": len(rows),
        "samples": len({int(row["sample_rank"]) for row in rows}),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "resolved_rows": resolved,
        "new_unique_safe_samples": len(safe_sample_ranks),
        "new_unique_safe_sample_ranks": safe_sample_ranks,
        "full_forward_unsafe_rows": witness_rows,
    }
    for key, value in recomputed.items():
        _issue(
            issues,
            runner_summary.get(key) == value,
            f"runner summary mismatch for {key}",
        )
    _issue(
        issues,
        runtime["parent_diagnostics_sha256"] == expected_parent_hash,
        "runtime parent hash mismatch",
    )
    report = {
        "audit": "experiment1f0_independent",
        "issues": issues,
        "issue_count": len(issues),
        "soundness_audit_passed": not issues,
        "recomputed_summary": recomputed,
        "safe_rows_checked": safe_rows,
        "witness_rows_replayed": witness_rows,
        "artifact_hashes_before_audit": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path != audit_path
        },
    }
    _write_json(audit_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = audit(Path(args.config).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
