"""Run the preregistered affine-router dimension-law simulation grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.baseline_icml2025_b1_smoke import PROJECT_ROOT, _inside, _sha256


def local_affine_top1_radii(
    scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return unbounded local top-1 L-infinity radii for a batch."""
    if scores.ndim != 2 or weights.ndim != 2:
        raise ValueError("scores and weights must be matrices")
    if scores.shape[1] != weights.shape[0]:
        raise ValueError("expert dimensions disagree")
    winners = np.argmax(scores, axis=1)
    result = np.full(scores.shape[0], np.inf, dtype=np.float64)
    for row, winner in enumerate(winners.tolist()):
        for competitor in range(scores.shape[1]):
            if competitor == winner:
                continue
            margin = float(scores[row, winner] - scores[row, competitor])
            denominator = float(np.abs(weights[winner] - weights[competitor]).sum())
            candidate = 0.0 if margin <= 0.0 else margin / denominator
            result[row] = min(result[row], candidate)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("local radius calculation produced a nonfinite value")
    return result


def synthetic_input_seed(
    config: dict[str, Any], dimension_index: int, router_seed: int, moment_index: int
) -> int:
    return (
        int(config["synthetic_input_seed_base"])
        + 100000 * dimension_index
        + 1000 * router_seed
        + moment_index
    )


def generate_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_count = int(config["samples_per_seed_and_moment"])
    experts = int(config["num_experts"])
    for dimension_index, dimension_raw in enumerate(config["dimensions"]):
        dimension = int(dimension_raw)
        for router_seed_raw in config["router_seeds"]:
            router_seed = int(router_seed_raw)
            torch.manual_seed(router_seed)
            layer = torch.nn.Linear(dimension, experts, bias=True, dtype=torch.float32)
            weights = layer.weight.detach().to(torch.float64).numpy()
            bias = layer.bias.detach().to(torch.float64).numpy()
            weight_l1_median = float(
                np.median(
                    [
                        np.abs(weights[i] - weights[j]).sum()
                        for i in range(experts)
                        for j in range(i + 1, experts)
                    ]
                )
            )
            for moment_index, family in enumerate(config["input_families"]):
                moment = float(family["fixed_second_moment"])
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    synthetic_input_seed(
                        config, dimension_index, router_seed, moment_index
                    )
                )
                signs = torch.randint(
                    0,
                    2,
                    (sample_count, dimension),
                    generator=generator,
                    dtype=torch.int8,
                )
                inputs = signs.to(torch.float64).mul_(2.0).sub_(1.0)
                inputs.mul_(math.sqrt(moment))
                scores = inputs.numpy() @ weights.T + bias
                radii = local_affine_top1_radii(scores, weights)
                for sample_index, radius in enumerate(radii.tolist()):
                    rows.append(
                        {
                            "dimension": dimension,
                            "dimension_index": dimension_index,
                            "router_seed": router_seed,
                            "moment_index": moment_index,
                            "input_family": str(family["label"]),
                            "fixed_second_moment": moment,
                            "sample_index": sample_index,
                            "local_radius": float(radius),
                            "weight_pair_l1_median": weight_l1_median,
                        }
                    )
    return rows


def _slope(log_dimensions: np.ndarray, log_radii: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(log_dimensions, log_radii, 1)
    return float(slope), float(intercept)


def summarize_rows(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    dimensions = [int(value) for value in config["dimensions"]]
    seeds = [int(value) for value in config["router_seeds"]]
    fits: dict[str, Any] = {}
    per_dimension: dict[str, Any] = {}
    rng = np.random.default_rng(20260830)
    for family in config["input_families"]:
        label = str(family["label"])
        family_rows = [row for row in rows if row["input_family"] == label]
        medians: list[float] = []
        for dimension in dimensions:
            values = [
                float(row["local_radius"])
                for row in family_rows
                if int(row["dimension"]) == dimension
            ]
            median = float(np.median(values))
            medians.append(median)
            per_dimension[f"{label}:{dimension}"] = {
                "count": len(values),
                "median_local_radius": median,
                "q25_local_radius": float(np.quantile(values, 0.25)),
                "q75_local_radius": float(np.quantile(values, 0.75)),
            }
        slope, intercept = _slope(
            np.log(np.asarray(dimensions, dtype=np.float64)),
            np.log(np.asarray(medians, dtype=np.float64)),
        )
        bootstrap_slopes: list[float] = []
        for _ in range(int(config["bootstrap_replicates"])):
            selected = rng.choice(seeds, size=len(seeds), replace=True).tolist()
            boot_medians: list[float] = []
            for dimension in dimensions:
                values: list[float] = []
                for seed in selected:
                    values.extend(
                        float(row["local_radius"])
                        for row in family_rows
                        if int(row["dimension"]) == dimension
                        and int(row["router_seed"]) == seed
                    )
                boot_medians.append(float(np.median(values)))
            boot_slope, _ = _slope(
                np.log(np.asarray(dimensions, dtype=np.float64)),
                np.log(np.asarray(boot_medians, dtype=np.float64)),
            )
            bootstrap_slopes.append(boot_slope)
        lower, upper = np.quantile(bootstrap_slopes, [0.025, 0.975]).tolist()
        expected = float(config["expected_slope"])
        tolerance = float(
            config["preregistered_consistency_rule"][
                "absolute_point_slope_error_at_most"
            ]
        )
        fits[label] = {
            "slope": slope,
            "intercept": intercept,
            "cluster_bootstrap_95_percent_interval": [float(lower), float(upper)],
            "absolute_error_from_expected": abs(slope - expected),
            "point_rule_passed": abs(slope - expected) <= tolerance,
            "interval_rule_passed": float(lower) <= expected <= float(upper),
            "preregistered_consistency_passed": bool(
                abs(slope - expected) <= tolerance
                and float(lower) <= expected <= float(upper)
            ),
        }
    first_label = str(config["input_families"][0]["label"])
    second_label = str(config["input_families"][1]["label"])
    radius_ratios = [
        per_dimension[f"{first_label}:{dimension}"]["median_local_radius"]
        / per_dimension[f"{second_label}:{dimension}"]["median_local_radius"]
        for dimension in dimensions
    ]
    expected_ratio = math.sqrt(
        float(config["input_families"][0]["fixed_second_moment"])
        / float(config["input_families"][1]["fixed_second_moment"])
    )
    return {
        "row_count": len(rows),
        "per_dimension": per_dimension,
        "fits": fits,
        "moment_scale_check": {
            "expected_sqrt_second_moment_ratio": expected_ratio,
            "median_observed_radius_ratio": float(np.median(radius_ratios)),
            "per_dimension_radius_ratios": radius_ratios,
        },
        "all_preregistered_consistency_rules_passed": all(
            bool(item["preregistered_consistency_passed"])
            for item in fits.values()
        ),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"simulation output exists: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("simulation config is not frozen before execution")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    rows = generate_rows(config)
    rows_path = output_dir / "rows.csv"
    write_rows(rows_path, rows)
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        "result": summarize_rows(rows, config),
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
