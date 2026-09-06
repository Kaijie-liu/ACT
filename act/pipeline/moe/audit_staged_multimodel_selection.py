"""Independently reconstruct the common three-model fixed-task selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json
from act.pipeline.moe.freeze_staged_multimodel_bundle import (
    EXCLUSION_SOURCES,
    MODELS,
    OUTPUT,
    SAMPLE_COUNT,
    START_DATASET_INDEX,
    select_common_clean_correct,
)
from act.pipeline.moe.freeze_staged_verifier_confirmatory import _indices_from_artifact
from act.pipeline.moe.train import _load_dataset


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/staged_verifier_multimodel_selection_20260906_r1.json"
)


@torch.no_grad()
def audit(selection_path: Path = OUTPUT) -> dict[str, Any]:
    selection_path = _inside(selection_path, PROJECT_ROOT)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if selection.get("status") != "FROZEN_BEFORE_VERIFICATION_ENDPOINTS":
        issues.append("selection status is not frozen before endpoints")
    if selection.get("request", {}).get("boundary_search") is not False:
        issues.append("boundary search is enabled")
    if selection.get("request", {}).get("route_instability_prefilter") is not False:
        issues.append("route-instability prefilter is enabled")

    excluded_indices: set[int] = set()
    exclusion_records: list[dict[str, Any]] = []
    for source_value in EXCLUSION_SOURCES:
        source = _inside(source_value, PROJECT_ROOT)
        value = json.loads(source.read_text(encoding="utf-8"))
        indices = _indices_from_artifact(value)
        excluded_indices.update(indices)
        exclusion_records.append(
            {"path": str(source), "sha256": _sha256(source), "count": len(indices)}
        )

    predictions: dict[str, list[int]] = {}
    labels: list[int] | None = None
    model_records: dict[str, Any] = {}
    for model_id, frozen in MODELS.items():
        checkpoint = _inside(Path(frozen["checkpoint"]), WRITE_ROOT)
        observed_hash = _sha256(checkpoint)
        if observed_hash != frozen["checkpoint_sha256"]:
            issues.append(f"{model_id} checkpoint hash differs from audit source")
        registered = selection.get("models", {}).get(model_id, {})
        if registered.get("checkpoint_sha256") != observed_hash:
            issues.append(f"{model_id} registered checkpoint hash differs")
        model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
        model.cpu().eval()
        dataset = _load_dataset(payload["dataset"], False, download=False)
        current_predictions: list[int] = []
        current_labels: list[int] = []
        for inputs, batch_labels in DataLoader(
            dataset, batch_size=512, shuffle=False, num_workers=0
        ):
            logits, _ = model.forward_with_routing(inputs)
            current_predictions.extend(
                int(value) for value in logits.argmax(dim=1).tolist()
            )
            current_labels.extend(int(value) for value in batch_labels.tolist())
        if labels is None:
            labels = current_labels
        elif current_labels != labels:
            issues.append(f"{model_id} ordered labels differ")
        predictions[model_id] = current_predictions
        model_records[model_id] = {
            "checkpoint_sha256": observed_hash,
            "clean_accuracy": sum(
                value == target
                for value, target in zip(current_predictions, current_labels)
            )
            / len(current_labels),
        }
    assert labels is not None
    reconstructed = select_common_clean_correct(
        predictions,
        labels,
        start_index=START_DATASET_INDEX,
        sample_count=SAMPLE_COUNT,
        excluded_indices=excluded_indices,
    )
    if reconstructed != selection.get("samples"):
        issues.append("common clean-correct selection does not reconstruct")
    indices = [int(row["dataset_index"]) for row in reconstructed]
    if len(indices) != len(set(indices)):
        issues.append("selection contains duplicate indices")
    if excluded_indices.intersection(indices):
        issues.append("selection overlaps a frozen prior cohort")
    if indices != sorted(indices):
        issues.append("selection is not in ordered dataset-index order")

    return {
        "schema_version": 1,
        "classification": "PRE_ENDPOINT_COMMON_SELECTION_AUDIT",
        "selection": {"path": str(selection_path), "sha256": _sha256(selection_path)},
        "models": model_records,
        "excluded_union_size": len(excluded_indices),
        "exclusion_artifacts": exclusion_records,
        "sample_count": len(reconstructed),
        "dataset_index_minimum": min(indices),
        "dataset_index_maximum": max(indices),
        "selection_reconstructed": reconstructed == selection.get("samples"),
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "claim_boundary": (
            "The audit uses only frozen model predictions, labels, ordering, and "
            "prior-cohort exclusion; it executes no verification endpoint."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path(
        "/data1/Kane/miniconda3/envs/act-py312/bin/python"
    ).resolve():
        raise RuntimeError("selection audit requires act-py312")
    result = audit(args.selection)
    output = _inside(args.output, PROJECT_ROOT)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
