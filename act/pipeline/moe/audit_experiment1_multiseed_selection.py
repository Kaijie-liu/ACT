# ===- act/pipeline/moe/audit_experiment1_multiseed_selection.py ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Independently reconstruct the multi-seed formal replication cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import _sha256, _write_json
from act.pipeline.moe.train import _load_dataset


PROJECT_ROOT = Path("/data1/Kane/MOE/ACT").resolve()
WRITE_ROOT = Path("/data1/Kane/MOE").resolve()
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/experiment1_multiseed_selection_r1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/moe/results/experiment1_multiseed_selection_r1_audit.json"
)


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"path escapes {root}: {resolved}")
    return resolved


def _indices(payload: dict[str, Any]) -> list[int]:
    if "indices" in payload:
        return [int(value) for value in payload["indices"]]
    return [int(row["dataset_index"]) for row in payload["samples"]]


def audit(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest_path = _inside(manifest_path, PROJECT_ROOT)
    output_path = _inside(output_path, WRITE_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    excluded: set[int] = set()
    exclusions: list[dict[str, Any]] = []
    for row in manifest["excluded_cohorts"]:
        path = _inside(Path(row["path"]), WRITE_ROOT)
        actual_hash = _sha256(path)
        if actual_hash != row["sha256"]:
            issues.append(f"excluded cohort hash differs: {path}")
        values = _indices(json.loads(path.read_text(encoding="utf-8")))
        excluded.update(values)
        exclusions.append({"path": str(path), "sha256": actual_hash, "indices": len(values)})

    dataset = _load_dataset("CIFAR10", False, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    model_records: list[dict[str, Any]] = []
    for model_id in ("seed1", "seed2"):
        registered = manifest["models"][model_id]
        checkpoint = _inside(Path(registered["checkpoint"]), WRITE_ROOT)
        actual_hash = _sha256(checkpoint)
        if actual_hash != registered["checkpoint_sha256"]:
            issues.append(f"{model_id} checkpoint hash differs")
        model, _ = load_output_moe_checkpoint(checkpoint, map_location=device)
        model.to(device).eval()
        models.append(model)
        model_records.append(
            {"model_id": model_id, "checkpoint": str(checkpoint), "sha256": actual_hash}
        )

    reconstructed: list[int] = []
    with torch.no_grad():
        for start in range(1000, len(dataset), 256):
            stop = min(start + 256, len(dataset))
            inputs = torch.stack([dataset[index][0] for index in range(start, stop)]).to(device)
            labels = torch.tensor(
                [dataset[index][1] for index in range(start, stop)], device=device
            )
            eligible = torch.ones(stop - start, dtype=torch.bool, device=device)
            for model in models:
                eligible &= model(inputs).argmax(dim=1).eq(labels)
            for offset in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
                index = start + int(offset)
                if index not in excluded:
                    reconstructed.append(index)
                if len(reconstructed) == len(manifest["samples"]):
                    break
            if len(reconstructed) == len(manifest["samples"]):
                break
    recorded = [int(row["dataset_index"]) for row in manifest["samples"]]
    if recorded != reconstructed:
        issues.append("recorded selection differs from independent reconstruction")
    if [int(row["sample_rank"]) for row in manifest["samples"]] != list(
        range(len(recorded))
    ):
        issues.append("sample ranks are not consecutive")
    if len(recorded) != len(set(recorded)):
        issues.append("selection contains duplicate indices")
    if any(index in excluded for index in recorded):
        issues.append("selection overlaps a frozen seed-0 cohort")

    report = {
        "audit": "experiment1_multiseed_selection_independent",
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "device": str(device),
        "models": model_records,
        "excluded_cohorts": exclusions,
        "excluded_union_size": len(excluded),
        "recorded_indices": recorded,
        "reconstructed_indices": reconstructed,
        "claim_boundary": "Selection audit uses predictions and labels only; no verification endpoint is queried.",
    }
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.manifest, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
