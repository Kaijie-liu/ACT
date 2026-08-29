# ===- act/pipeline/moe/audit_route_invariance_baseline.py - Audit --====#

"""Independent audit for the explicit confirmatory route-invariance baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.route_invariance_baseline import _summary
from act.pipeline.moe.train import _load_dataset


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/route_invariance_baseline_confirmatory.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _witness_file(
    row: dict[str, Any], output_dir: Path, parent_dir: Path
) -> Path | None:
    relative = row.get("witness_path")
    if not relative:
        return None
    if row.get("route_a_source") == "frozen_confirmatory_artifact":
        return parent_dir / relative
    return output_dir / relative


def _replay_witness(
    row: dict[str, Any], model, dataset, path: Path
) -> tuple[bool, str | None]:
    if not path.exists():
        return False, f"missing witness {path}"
    if row.get("witness_sha256") != _sha256(path):
        return False, f"witness hash mismatch at rank {row['sample_rank']}"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidate = payload["input"].double()
    if candidate.ndim == 3:
        candidate = candidate.unsqueeze(0)
    image, _ = dataset[int(row["dataset_index"])]
    clean = image.unsqueeze(0).double()
    epsilon = float(row["epsilon"])
    lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
    inside = bool(torch.all(candidate >= lower - 1e-7)) and bool(
        torch.all(candidate <= upper + 1e-7)
    )
    with torch.no_grad():
        clean_prediction = int(model(clean).argmax(dim=1).item())
        prediction = int(model(candidate).argmax(dim=1).item())
    if not inside:
        return False, f"witness outside endpoint box at rank {row['sample_rank']}"
    if prediction == clean_prediction:
        return False, f"witness does not flip full-model prediction at rank {row['sample_rank']}"
    return True, None


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.load(config_path.open(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    rows = _read_jsonl(results_path)
    recorded_summary = json.load(summary_path.open(encoding="utf-8"))
    independent_summary = _summary(rows)
    parent_path = _inside(Path(config["parent_results_jsonl"]), WRITE_ROOT)
    parent_dir = parent_path.parent
    parent_rows = _read_jsonl(parent_path)
    issues: list[str] = []
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)

    if _sha256(parent_path) != config["parent_results_sha256"]:
        issues.append("parent artifact hash changed")
    if len(rows) != int(config["expected_samples"]):
        issues.append(f"expected 100 rows, found {len(rows)}")
    ranks = [int(row["sample_rank"]) for row in rows]
    if len(ranks) != len(set(ranks)):
        issues.append("duplicate sample rank")
    if set(ranks) != set(range(100, 200)):
        issues.append("sample ranks are not exactly 100--199")
    counts = Counter(row.get("route_precondition_status") for row in rows)
    if dict(counts) != config["expected_precondition_counts"]:
        issues.append(f"precondition count mismatch: {dict(counts)}")

    parent_by_rank = {int(row["sample_rank"]): row for row in parent_rows}
    replayed = 0
    for row in rows:
        rank = int(row["sample_rank"])
        precondition = row.get("route_precondition_status")
        pair_count = int(row.get("exact_feasible_pair_count", 0))
        if precondition == "INVARIANT":
            if pair_count != 1:
                issues.append(f"rank {rank}: invariant row does not have one pair")
            if row.get("route_a_source") != "shared_fresh_invariant_endpoint_solve":
                issues.append(f"rank {rank}: invariant downstream solve was not shared")
            if row.get("baseline_status") != row.get("route_a_status"):
                issues.append(f"rank {rank}: shared statuses differ")
        elif precondition == "UNSTABLE":
            if pair_count <= 1:
                issues.append(f"rank {rank}: unstable row lacks multiple pairs")
            if row.get("baseline_status") != "UNKNOWN":
                issues.append(f"rank {rank}: failed precondition did not return UNKNOWN")
            if row.get("baseline_reason") != "ROUTE_INVARIANCE_PRECONDITION_FAILED":
                issues.append(f"rank {rank}: failed precondition reason changed")
            if row.get("route_a_source") != "frozen_confirmatory_artifact":
                issues.append(f"rank {rank}: unstable Route A source changed")
            parent = parent_by_rank[rank]
            if row.get("route_a_status") != parent.get("status"):
                issues.append(f"rank {rank}: frozen Route A status drift")
            if row.get("route_a_reason") != parent.get("reason"):
                issues.append(f"rank {rank}: frozen Route A reason drift")
        else:
            issues.append(f"rank {rank}: unknown precondition label {precondition}")
        expected_only = (
            precondition == "UNSTABLE" and row.get("route_a_status") == "SAFE"
        )
        if bool(row.get("route_a_only_safe")) != expected_only:
            issues.append(f"rank {rank}: Route A-only SAFE flag mismatch")
        if row.get("baseline_status") == "UNSAFE" or row.get("route_a_status") == "UNSAFE":
            path = _witness_file(row, output_dir, parent_dir)
            if path is None:
                issues.append(f"rank {rank}: UNSAFE lacks witness path")
            else:
                ok, issue = _replay_witness(row, model, dataset, path)
                if ok:
                    replayed += 1
                else:
                    issues.append(issue or f"rank {rank}: witness replay failed")
        if row.get("baseline_status") == "SAFE":
            if row.get("baseline_reason") not in {
                "SAFE_GATE_ELIMINATION",
                "SAFE_WEIGHTED_RANGE",
            }:
                issues.append(f"rank {rank}: unregistered SAFE reason")

    if independent_summary != recorded_summary:
        issues.append("recorded summary differs from independent recomputation")
    return {
        "audit": "route_invariance_baseline_independent",
        "config_sha256": _sha256(config_path),
        "results_sha256": _sha256(results_path),
        "summary_sha256": _sha256(summary_path),
        "rows": len(rows),
        "unique_ranks": len(set(ranks)),
        "precondition_counts": dict(counts),
        "unsafe_witnesses_replayed": replayed,
        "independent_summary": independent_summary,
        "issue_count": len(issues),
        "issues": issues,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = audit(Path(args.config))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = _inside(Path(args.output), WRITE_ROOT)
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
