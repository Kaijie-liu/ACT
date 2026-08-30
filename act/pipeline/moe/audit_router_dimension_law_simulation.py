"""Independently audit the preregistered dimension-law simulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.baseline_icml2025_b1_smoke import PROJECT_ROOT, _inside, _sha256
from act.pipeline.moe.router_dimension_law_simulation import (
    local_affine_top1_radii,
    summarize_rows,
    synthetic_input_seed,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _typed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                **row,
                "dimension": int(row["dimension"]),
                "dimension_index": int(row["dimension_index"]),
                "router_seed": int(row["router_seed"]),
                "moment_index": int(row["moment_index"]),
                "fixed_second_moment": float(row["fixed_second_moment"]),
                "sample_index": int(row["sample_index"]),
                "local_radius": float(row["local_radius"]),
                "weight_pair_l1_median": float(row["weight_pair_l1_median"]),
            }
        )
    return result


def audit(config_path: Path, result_dir: Path, output_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    result_dir = _inside(result_dir)
    output_path = _inside(output_path)
    if output_path.exists():
        raise RuntimeError(f"audit output exists: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    rows_path = result_dir / "rows.csv"
    rows = _typed_rows(_read_rows(rows_path))
    issues: list[str] = []
    if raw["config"]["sha256"] != _sha256(config_path):
        issues.append("config hash mismatch")
    if raw["rows"]["sha256"] != _sha256(rows_path):
        issues.append("row hash mismatch")
    expected_count = (
        len(config["dimensions"])
        * len(config["router_seeds"])
        * len(config["input_families"])
        * int(config["samples_per_seed_and_moment"])
    )
    identities = {
        (
            row["dimension"],
            row["router_seed"],
            row["moment_index"],
            row["sample_index"],
        )
        for row in rows
    }
    if len(rows) != expected_count or len(identities) != expected_count:
        issues.append("row count or identity uniqueness mismatch")
    recomputed = summarize_rows(rows, config)
    if json.dumps(recomputed, sort_keys=True) != json.dumps(raw["result"], sort_keys=True):
        issues.append("summary does not independently recompute")
    scalar_checks = 0
    scalar_max_error = 0.0
    for dimension_index in (0, len(config["dimensions"]) - 1):
        dimension = int(config["dimensions"][dimension_index])
        for router_seed in (int(config["router_seeds"][0]), int(config["router_seeds"][-1])):
            torch.manual_seed(router_seed)
            layer = torch.nn.Linear(
                dimension, int(config["num_experts"]), bias=True, dtype=torch.float32
            )
            weights = layer.weight.detach().to(torch.float64).numpy()
            bias = layer.bias.detach().to(torch.float64).numpy()
            for moment_index, family in enumerate(config["input_families"]):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    synthetic_input_seed(
                        config, dimension_index, router_seed, moment_index
                    )
                )
                signs = torch.randint(
                    0, 2, (1, dimension), generator=generator, dtype=torch.int8
                )
                inputs = signs.to(torch.float64).mul_(2.0).sub_(1.0)
                inputs.mul_(math.sqrt(float(family["fixed_second_moment"])))
                scores = inputs.numpy() @ weights.T + bias
                observed = float(local_affine_top1_radii(scores, weights)[0])
                stored = next(
                    float(row["local_radius"])
                    for row in rows
                    if row["dimension"] == dimension
                    and row["router_seed"] == router_seed
                    and row["moment_index"] == moment_index
                    and row["sample_index"] == 0
                )
                error = abs(observed - stored)
                scalar_max_error = max(scalar_max_error, error)
                scalar_checks += 1
                if error > 1e-15:
                    issues.append(
                        f"scalar replay mismatch d={dimension} seed={router_seed} family={family['label']}"
                    )
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "raw_summary": {
            "path": str(result_dir / "summary.json"),
            "sha256": _sha256(result_dir / "summary.json"),
        },
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        "recomputed": recomputed,
        "scalar_replay_checks": scalar_checks,
        "scalar_replay_maximum_absolute_error": scalar_max_error,
        "claim_boundary": config["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if issues:
        raise RuntimeError(f"dimension-law simulation audit found {len(issues)} issue(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.config, args.result_dir, args.output)


if __name__ == "__main__":
    main()
