"""Freeze the untouched fixed-radius cohort for staged verifier evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.train import _load_dataset


CHECKPOINT = PROJECT_ROOT / "data/moe/checkpoints/cifar10_top2_e8_seed2_bal010.pt"
CHECKPOINT_SHA256 = "a60517c7964177858d10303bc7b306293c8da3cde6caa93a5d8f442968e97758"
OUTPUT = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_seed2_fixed2_confirmatory_selection_r1.json"
)
START_DATASET_INDEX = 2000
SAMPLE_COUNT = 100
EPSILON = 2.0 / 255.0
EXCLUSION_SOURCES = (
    PROJECT_ROOT / "data/moe/results/experiment1_bal010/sample_indices.json",
    PROJECT_ROOT
    / "data/moe/results/experiment1_confirmatory_bal010_r1/sample_indices.json",
    PROJECT_ROOT
    / "act/pipeline/moe/configs/experiment1_multiseed_selection_r1.json",
)


def _indices_from_artifact(value: dict[str, Any]) -> list[int]:
    if "indices" in value:
        return [int(index) for index in value["indices"]]
    if "samples" in value:
        return [int(row["dataset_index"]) for row in value["samples"]]
    raise ValueError("exclusion artifact has no supported index collection")


def select_clean_correct(
    predictions: Sequence[int],
    labels: Sequence[int],
    *,
    start_index: int,
    sample_count: int,
    excluded_indices: set[int],
) -> list[dict[str, int]]:
    """Apply the frozen ordered, clean-correct-only selection rule."""
    if len(predictions) != len(labels):
        raise ValueError("prediction and label lengths differ")
    if start_index < 0 or sample_count <= 0:
        raise ValueError("invalid selection bounds")
    selected: list[dict[str, int]] = []
    clean_rank_after_start = 0
    for dataset_index in range(start_index, len(labels)):
        prediction = int(predictions[dataset_index])
        label = int(labels[dataset_index])
        if prediction != label:
            continue
        rank = clean_rank_after_start
        clean_rank_after_start += 1
        if dataset_index in excluded_indices:
            continue
        selected.append(
            {
                "sample_rank": len(selected),
                "clean_correct_rank_after_start": rank,
                "dataset_index": dataset_index,
                "label": label,
                "clean_prediction": prediction,
            }
        )
        if len(selected) == sample_count:
            break
    if len(selected) != sample_count:
        raise RuntimeError(
            f"only {len(selected)} eligible samples found; need {sample_count}"
        )
    return selected


@torch.no_grad()
def freeze(output: Path = OUTPUT) -> Path:
    output = _inside(output, PROJECT_ROOT)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen selection {output}")
    if Path(sys.executable).resolve() != Path(
        "/data1/Kane/miniconda3/envs/act-py312/bin/python"
    ).resolve():
        raise RuntimeError("selection must use the frozen act-py312 interpreter")
    checkpoint = _inside(CHECKPOINT, WRITE_ROOT)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("seed-2 checkpoint identity changed")

    exclusions: list[dict[str, Any]] = []
    excluded_indices: set[int] = set()
    for source in EXCLUSION_SOURCES:
        source = _inside(source, PROJECT_ROOT)
        value = json.loads(source.read_text(encoding="utf-8"))
        indices = _indices_from_artifact(value)
        excluded_indices.update(indices)
        exclusions.append(
            {
                "path": str(source),
                "sha256": _sha256(source),
                "count": len(indices),
                "minimum": min(indices),
                "maximum": max(indices),
            }
        )

    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.cpu().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    predictions: list[int] = []
    labels: list[int] = []
    for inputs, batch_labels in loader:
        logits, _ = model.forward_with_routing(inputs)
        predictions.extend(int(value) for value in logits.argmax(dim=1).tolist())
        labels.extend(int(value) for value in batch_labels.tolist())
    samples = select_clean_correct(
        predictions,
        labels,
        start_index=START_DATASET_INDEX,
        sample_count=SAMPLE_COUNT,
        excluded_indices=excluded_indices,
    )
    if excluded_indices.intersection(row["dataset_index"] for row in samples):
        raise RuntimeError("new selection overlaps a frozen prior cohort")

    manifest = {
        "schema_version": 1,
        "experiment": "staged_verifier_seed2_fixed2_confirmatory_r1",
        "status": "FROZEN_BEFORE_VERIFICATION_ENDPOINTS",
        "classification": "NEW_FIXED_RADIUS_HZ_COHORT",
        "selection_rule": (
            "first 100 ordered CIFAR-10 test indices at or after 2000 that "
            "are clean-correct for the frozen seed-2 checkpoint and absent "
            "from all listed prior HZ cohorts"
        ),
        "selection_predicates_only": [
            "dataset_index >= 2000",
            "clean_prediction == official test label",
            "dataset_index absent from the frozen exclusion union",
        ],
        "forbidden_selection_predicates": [
            "candidate count",
            "route stability or route-boundary radius",
            "guard elimination",
            "solver status",
            "certificate status",
        ],
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "dataset": payload["dataset"],
        },
        "dataset": {
            "name": "CIFAR10",
            "split": "official torchvision test",
            "length": len(dataset),
            "ordered": True,
        },
        "request": {
            "property": "TOP1_ROBUST relative to clean prediction",
            "epsilon_label": "2/255",
            "epsilon": EPSILON,
            "boundary_search": False,
            "route_instability_prefilter": False,
        },
        "excluded_cohorts": exclusions,
        "excluded_dataset_index_count": len(excluded_indices),
        "samples": samples,
        "primary_endpoint": (
            "number of complete SAFE requests with more than one exact feasible "
            "unordered top-2 route set, divided by all 100 selected inputs"
        ),
        "integrity_gates": {
            "independent_audit_issues": 0,
            "all_unsafe_full_model_replayed": True,
        },
        "preregistered_replication_signal": {
            "route_changing_safe_requests": 1,
            "interpretation": (
                "existence replication for the production entry point; the "
                "full count and 100-input denominator must always be reported"
            ),
        },
        "secondary_endpoints": [
            "overall SAFE/UNSAFE/UNKNOWN/TIMEOUT counts",
            "Tier-1 and F0 SAFE counts",
            "F0 invocation and resolution counts",
            "exact route-stable and route-changing request counts",
            "stage timing with timed-out stages right-censored",
        ],
        "claim_boundary": (
            "No boundary adaptation, no route-instability selection, no "
            "experiment-only no-support control, no certified-accuracy claim, "
            "and no comparison to the locked AdvMoE Lagrangian holdout."
        ),
    }
    _write_json(output, manifest)
    return output


def main() -> None:
    path = freeze()
    print(path)


if __name__ == "__main__":
    main()
