"""Freeze a common, endpoint-blind three-model staged-verifier cohort."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json
from act.pipeline.moe.freeze_staged_verifier_confirmatory import _indices_from_artifact
from act.pipeline.moe.train import _load_dataset


OUTPUT = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_multimodel_fixed2_selection_r1.json"
)
START_DATASET_INDEX = 3000
SAMPLE_COUNT = 100
EPSILON = 2.0 / 255.0
MODELS = {
    "seed0": {
        "checkpoint": PROJECT_ROOT / "data/moe/checkpoints/cifar10_top2_e8_seed0_bal010.pt",
        "checkpoint_sha256": "fbaa7c871d28763ac5acb29a9502dc5d146e1d5af0b4a03e9911899251bd43f7",
    },
    "seed1": {
        "checkpoint": PROJECT_ROOT / "data/moe/checkpoints/cifar10_top2_e8_seed1_bal010.pt",
        "checkpoint_sha256": "cfd5fb07b1c426c5b98309d56a5d0439f8a99dc52ff936bcbea43d4fe641139a",
    },
    "seed2": {
        "checkpoint": PROJECT_ROOT / "data/moe/checkpoints/cifar10_top2_e8_seed2_bal010.pt",
        "checkpoint_sha256": "a60517c7964177858d10303bc7b306293c8da3cde6caa93a5d8f442968e97758",
    },
}
EXCLUSION_SOURCES = (
    PROJECT_ROOT / "data/moe/results/experiment1_bal010/sample_indices.json",
    PROJECT_ROOT / "data/moe/results/experiment1_confirmatory_bal010_r1/sample_indices.json",
    PROJECT_ROOT / "act/pipeline/moe/configs/experiment1_multiseed_selection_r1.json",
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_seed2_fixed2_confirmatory_selection_r1.json",
)


def select_common_clean_correct(
    predictions: Mapping[str, Sequence[int]],
    labels: Sequence[int],
    *,
    start_index: int,
    sample_count: int,
    excluded_indices: set[int],
) -> list[dict[str, Any]]:
    """Select the first ordered indices clean-correct for every model."""
    if not predictions:
        raise ValueError("at least one model is required")
    if start_index < 0 or sample_count <= 0:
        raise ValueError("invalid selection bounds")
    size = len(labels)
    if any(len(values) != size for values in predictions.values()):
        raise ValueError("prediction and label lengths differ")
    selected: list[dict[str, Any]] = []
    eligible_rank = 0
    for dataset_index in range(start_index, size):
        label = int(labels[dataset_index])
        by_model = {
            model_id: int(values[dataset_index])
            for model_id, values in sorted(predictions.items())
        }
        if any(value != label for value in by_model.values()):
            continue
        rank = eligible_rank
        eligible_rank += 1
        if dataset_index in excluded_indices:
            continue
        selected.append(
            {
                "sample_rank": len(selected),
                "common_clean_correct_rank_after_start": rank,
                "dataset_index": dataset_index,
                "label": label,
                "clean_prediction": label,
                "clean_predictions": by_model,
            }
        )
        if len(selected) == sample_count:
            break
    if len(selected) != sample_count:
        raise RuntimeError(
            f"only {len(selected)} common clean-correct inputs found; need {sample_count}"
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

    exclusions: list[dict[str, Any]] = []
    excluded_indices: set[int] = set()
    for source_value in EXCLUSION_SOURCES:
        source = _inside(source_value, PROJECT_ROOT)
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

    model_records: dict[str, Any] = {}
    predictions: dict[str, list[int]] = {}
    labels: list[int] | None = None
    dataset = None
    for model_id, record in MODELS.items():
        checkpoint = _inside(Path(record["checkpoint"]), WRITE_ROOT)
        if _sha256(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"{model_id} checkpoint identity changed")
        model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
        model.cpu().eval()
        current_dataset = _load_dataset(payload["dataset"], False, download=False)
        if dataset is None:
            dataset = current_dataset
        elif len(current_dataset) != len(dataset):
            raise RuntimeError("model datasets differ in length")
        current_predictions: list[int] = []
        current_labels: list[int] = []
        for inputs, batch_labels in DataLoader(
            current_dataset, batch_size=512, shuffle=False, num_workers=0
        ):
            logits, _ = model.forward_with_routing(inputs)
            current_predictions.extend(
                int(value) for value in logits.argmax(dim=1).tolist()
            )
            current_labels.extend(int(value) for value in batch_labels.tolist())
        if labels is None:
            labels = current_labels
        elif current_labels != labels:
            raise RuntimeError("ordered dataset labels differ between models")
        predictions[model_id] = current_predictions
        model_records[model_id] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "dataset": payload["dataset"],
            "clean_accuracy": sum(
                value == target
                for value, target in zip(current_predictions, current_labels)
            )
            / len(current_labels),
        }
    assert dataset is not None and labels is not None
    samples = select_common_clean_correct(
        predictions,
        labels,
        start_index=START_DATASET_INDEX,
        sample_count=SAMPLE_COUNT,
        excluded_indices=excluded_indices,
    )
    if excluded_indices.intersection(int(row["dataset_index"]) for row in samples):
        raise RuntimeError("new selection overlaps a frozen prior cohort")

    manifest = {
        "schema_version": 1,
        "experiment": "staged_verifier_multimodel_fixed2_r1",
        "status": "FROZEN_BEFORE_VERIFICATION_ENDPOINTS",
        "classification": "PREREGISTERED_COMMON_FIXED_TASK_CROSS_MODEL_BUNDLE",
        "selection_rule": (
            "first 100 ordered CIFAR-10 test indices at or after 3000 that "
            "are clean-correct for all three frozen bal010 training seeds and "
            "absent from every listed prior HZ cohort"
        ),
        "selection_predicates_only": [
            "dataset_index >= 3000",
            "clean prediction equals the official label for seed0, seed1, and seed2",
            "dataset_index absent from the frozen exclusion union",
        ],
        "forbidden_selection_predicates": [
            "candidate count or candidate reduction",
            "route stability or route-boundary radius",
            "guard elimination or structural width",
            "solver status, certificate status, or runtime",
        ],
        "models": model_records,
        "dataset": {
            "name": "CIFAR10",
            "split": "official torchvision test",
            "length": len(dataset),
            "ordered": True,
        },
        "request": {
            "property": "TOP1_ROBUST relative to each model's clean prediction",
            "epsilon_label": "2/255",
            "epsilon": EPSILON,
            "boundary_search": False,
            "route_instability_prefilter": False,
        },
        "excluded_cohorts": exclusions,
        "excluded_dataset_index_count": len(excluded_indices),
        "samples": samples,
        "integrity_gates": {
            "independent_audit_issues": 0,
            "all_unsafe_full_model_replayed": True,
            "exact_candidate_subset_consistency": True,
            "guard_accounting_identity_closed": True,
        },
        "per_model_performance_bundle": {
            "minimum_route_changing_safe_requests": 1,
            "minimum_complete_outcome_rate": 0.5,
            "minimum_exact_vs_zonotope_strict_reduction_rate_on_route_unstable_rows": 0.2,
            "maximum_route_unstable_width_ratio_median": 0.7,
            "maximum_route_unstable_width_ratio_p90_strict": 1.0,
            "minimum_f0_complete_resolution_rate_when_invoked": 0.25,
            "minimum_guard_binary_eliminations": 1,
        },
        "cross_model_success_rule": (
            "every integrity gate and every per-model bundle threshold must pass "
            "for seed0, seed1, and seed2; no partial conjunction is relabeled"
        ),
        "execution_separation": {
            "verdict": (
                "production staged verifier only; no boundary search, matched "
                "no-support solve, or unguarded accounting inside its budget"
            ),
            "census": (
                "separate fixed-radius candidate/width/guard measurement; its "
                "runtime is not charged to or subtracted from the verdict run"
            ),
        },
        "claim_boundary": (
            "The common 100-input clean-correct cohort is not certified accuracy. "
            "All three models share an architecture and training recipe, so a pass "
            "supports stability across registered training runs, not across model families."
        ),
    }
    _write_json(output, manifest)
    return output


def main() -> None:
    path = freeze()
    print(path)


if __name__ == "__main__":
    main()
