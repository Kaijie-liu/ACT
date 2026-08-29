"""Independent artifact and witness audit for the N1 engineering rerun."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json
from act.pipeline.moe.train import _load_dataset


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _safe_structure_issues(
    row: dict[str, Any], *, expected_property_rows: int | None = None
) -> list[str]:
    issues: list[str] = []
    n1 = row.get("n1") or {}
    if n1.get("status") != "SAFE":
        issues.append("row SAFE but nested N1 status is not SAFE")
    pairs = n1.get("pairs", [])
    feasible_pairs = {
        tuple(sorted(int(value) for value in pair))
        for pair in n1.get("feasible_pairs", [])
    }
    recorded_pairs = {
        tuple(sorted(int(value) for value in pair.get("pair", [])))
        for pair in pairs
    }
    if not feasible_pairs:
        issues.append("SAFE row has no feasible pair")
    if recorded_pairs != feasible_pairs:
        issues.append("SAFE row does not cover every feasible pair exactly once")
    if len(recorded_pairs) != len(pairs):
        issues.append("SAFE row contains a duplicate pair")
    for pair in pairs:
        if pair.get("status") != "SAFE":
            issues.append(f"SAFE row contains non-SAFE pair {pair.get('pair')}")
        properties = pair.get("property_rows", [])
        property_indices = [int(prop.get("property_index", -1)) for prop in properties]
        if len(set(property_indices)) != len(property_indices):
            issues.append(f"SAFE pair {pair.get('pair')} contains duplicate properties")
        if expected_property_rows is not None and set(property_indices) != set(
            range(expected_property_rows)
        ):
            issues.append(
                f"SAFE pair {pair.get('pair')} does not cover every property row"
            )
        for prop in properties:
            if prop.get("status") != "SAFE":
                issues.append(
                    f"SAFE row contains non-SAFE property {pair.get('pair')}/"
                    f"{prop.get('property_index')}"
                )
            if prop.get("reused_parent"):
                continue
            if prop.get("reason") != "SAFE_WEIGHTED_SEGMENTED":
                issues.append("new segmented SAFE property has the wrong reason")
            segments = prop.get("segments", [])
            if not segments:
                issues.append("new segmented SAFE property has no active segments")
            if any(segment.get("decision", {}).get("status") != "SAFE" for segment in segments):
                issues.append("new segmented SAFE property contains a non-SAFE segment")
    return issues


def _replay_unsafe(
    row: dict[str, Any],
    *,
    result_dir: Path,
    model,
    dataset,
) -> list[str]:
    issues: list[str] = []
    if not row.get("full_model_witness_valid"):
        return ["UNSAFE row lacks full_model_witness_valid"]
    relative = row.get("witness_path")
    expected_hash = row.get("witness_sha256")
    if not relative or not expected_hash:
        return ["UNSAFE row lacks witness path or hash"]
    path = result_dir / relative
    if not path.exists():
        return ["UNSAFE witness file is missing"]
    if _sha256(path) != expected_hash:
        return ["UNSAFE witness hash mismatch"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidate = payload.get("input")
    if not isinstance(candidate, torch.Tensor):
        return ["UNSAFE witness payload has no tensor input"]
    image, label = dataset[int(row["dataset_index"])]
    clean = image.unsqueeze(0).double()
    epsilon = float(row["epsilon"])
    lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
    value = candidate.unsqueeze(0).double()
    if value.shape != clean.shape:
        issues.append("UNSAFE witness shape differs from clean input")
        return issues
    tolerance = 1e-7
    if bool((value < lower - tolerance).any()) or bool((value > upper + tolerance).any()):
        issues.append("UNSAFE witness lies outside frozen L-infinity region")
    with torch.no_grad():
        clean_output, _ = model.forward_with_routing(clean)
        candidate_output, candidate_route = model.forward_with_routing(value)
    clean_prediction = int(clean_output.argmax(dim=1).item())
    candidate_prediction = int(candidate_output.argmax(dim=1).item())
    if clean_prediction != int(label):
        issues.append("frozen sample is no longer clean-correct")
    if candidate_prediction == clean_prediction:
        issues.append("UNSAFE witness does not change full-model prediction")
    recorded_prediction = (row.get("n1") or {}).get("counterexample_prediction")
    if recorded_prediction is not None and candidate_prediction != int(recorded_prediction):
        issues.append("UNSAFE replay prediction differs from recorded prediction")
    recorded_route = (row.get("n1") or {}).get("counterexample_topk_set")
    actual_route = sorted(int(value) for value in candidate_route.indices[0].tolist())
    if recorded_route is not None and actual_route != sorted(int(value) for value in recorded_route):
        issues.append("UNSAFE replay route differs from recorded route")
    return issues


def audit(config_path: Path, result_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    result_dir = _inside(result_dir, WRITE_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(result_dir / "results.jsonl")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_path = _inside(Path(config["baseline_results_jsonl"]), WRITE_ROOT)
    baseline_rows = _load_jsonl(baseline_path)
    baseline = {int(row["sample_rank"]): row for row in baseline_rows}
    selection = json.loads(
        _inside(Path(config["selection_manifest"]), PROJECT_ROOT).read_text(encoding="utf-8")
    )
    expected_ranks = [int(row["sample_rank"]) for row in selection["rows"]]
    issues: list[dict[str, Any]] = []

    def add(scope: str, detail: str, rank: int | None = None) -> None:
        issues.append({"scope": scope, "detail": detail, "sample_rank": rank})

    if runtime.get("source_config_sha256") != _sha256(config_path):
        add("provenance", "source config hash mismatch")
    if runtime.get("config") != config:
        add("provenance", "embedded runtime config differs from source config")
    if runtime.get("checkpoint_sha256") != _sha256(Path(config["checkpoint"])):
        add("provenance", "checkpoint hash mismatch")
    if _sha256(baseline_path) != config["baseline_results_sha256"]:
        add("provenance", "baseline artifact hash mismatch")
    if len(rows) != int(config["expected_rows"]):
        add("rows", "result row count differs from frozen expected_rows")
    actual_ranks = [int(row["sample_rank"]) for row in rows]
    if actual_ranks != expected_ranks:
        add("rows", "result rank order differs from frozen selection")
    if len(set(actual_ranks)) != len(actual_ranks):
        add("rows", "duplicate sample rank")

    model, payload = load_output_moe_checkpoint(Path(config["checkpoint"]), map_location="cpu")
    model.double().eval()
    if payload.get("dataset") != "CIFAR10":
        add("provenance", "checkpoint dataset is not frozen CIFAR10")
    dataset = _load_dataset(payload["dataset"], False, download=False)
    num_classes = int(payload["factory_config"]["num_classes"])
    expected_property_rows = num_classes - 1
    for row in rows:
        rank = int(row["sample_rank"])
        parent = baseline.get(rank)
        if parent is None:
            add("pairing", "rank missing from paired baseline", rank)
            continue
        if row.get("baseline_status") != parent.get("status"):
            add("pairing", "baseline status mismatch", rank)
        if row.get("baseline_reason") != parent.get("reason"):
            add("pairing", "baseline reason mismatch", rank)
        if int(row.get("dataset_index", -1)) != int(parent.get("dataset_index", -2)):
            add("pairing", "dataset index mismatch", rank)
        if abs(float(row.get("epsilon", -1.0)) - float(parent.get("epsilon", -2.0))) > 1e-15:
            add("pairing", "epsilon mismatch", rank)
        n1_pairs = {
            tuple(sorted(int(value) for value in pair))
            for pair in (row.get("n1") or {}).get("feasible_pairs", [])
        }
        baseline_pairs = {
            tuple(sorted(int(value) for value in pair))
            for pair in (parent.get("f0") or {}).get("feasible_pairs", [])
        }
        if row.get("status") != "TIMEOUT" and n1_pairs != baseline_pairs:
            add("pairing", "N1 feasible route pairs differ from paired baseline", rank)
        expected_transition = f"{parent['status']}->{row['status']}"
        if row.get("paired_transition") != expected_transition:
            add("pairing", "paired transition mismatch", rank)
        if (parent["status"], row["status"]) in {
            ("SAFE", "UNSAFE"),
            ("UNSAFE", "SAFE"),
        }:
            add("soundness", "paired solved semantics conflict", rank)
        if row["status"] == "SAFE":
            for detail in _safe_structure_issues(
                row, expected_property_rows=expected_property_rows
            ):
                add("safe_structure", detail, rank)
        if row["status"] == "UNSAFE":
            for detail in _replay_unsafe(
                row, result_dir=result_dir, model=model, dataset=dataset
            ):
                add("unsafe_replay", detail, rank)
        if row["status"] == "ERROR" or "NUMERICAL" in str(row.get("reason")):
            add("execution", "explicit error or numerical fallback", rank)

    recomputed_status = dict(Counter(row["status"] for row in rows))
    recomputed_reasons = dict(Counter(row["reason"] for row in rows))
    recomputed_transitions = dict(
        sorted(Counter(row.get("paired_transition") for row in rows).items())
    )
    if recomputed_status != summary.get("status_counts"):
        add("summary", "status counts mismatch")
    if recomputed_reasons != summary.get("reason_counts"):
        add("summary", "reason counts mismatch")
    if recomputed_transitions != summary.get("paired_transitions"):
        add("summary", "paired transition counts mismatch")
    return {
        "schema_version": 1,
        "scope": "independent_n1_engineering_artifact_audit",
        "issues": issues,
        "issue_count": len(issues),
        "rows": len(rows),
        "unique_ranks": len(set(actual_ranks)),
        "unsafe_rows": sum(row["status"] == "UNSAFE" for row in rows),
        "unsafe_replayed": sum(
            row["status"] == "UNSAFE"
            and not any(
                issue["scope"] == "unsafe_replay" and issue["sample_rank"] == row["sample_rank"]
                for issue in issues
            )
            for row in rows
        ),
        "artifact_sha256": {
            "source_config": _sha256(config_path),
            "runtime_config": _sha256(result_dir / "config.json"),
            "results_jsonl": _sha256(result_dir / "results.jsonl"),
            "results_csv": _sha256(result_dir / "results.csv"),
            "summary": _sha256(result_dir / "summary.json"),
            "baseline_results": _sha256(baseline_path),
        },
        "confirmatory_endpoint_unchanged": 0.56,
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
