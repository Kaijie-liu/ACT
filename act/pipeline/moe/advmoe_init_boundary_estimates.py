"""Analyze two empirical AdvMoE init-router boundary estimates per sample."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def compute_boundary_estimates(
    clean_margin: np.ndarray,
    gradient_l1: np.ndarray,
    compression: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local first-order and attack-slope extrapolated radii."""
    clean_margin = np.asarray(clean_margin, dtype=np.float64)
    gradient_l1 = np.asarray(gradient_l1, dtype=np.float64)
    compression = np.asarray(compression, dtype=np.float64)
    if not (
        clean_margin.shape == gradient_l1.shape == compression.shape
        and clean_margin.ndim == 1
    ):
        raise ValueError("boundary diagnostic arrays must be aligned vectors")
    if np.any(clean_margin <= 0) or np.any(gradient_l1 <= 0):
        raise ValueError("clean margins and gradient L1 norms must be positive")
    if np.any(compression <= 0) or np.any(compression >= 1):
        raise ValueError("compression must lie strictly between zero and one")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return clean_margin / gradient_l1, float(epsilon) / compression


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
    }


def _plot(path: Path, indices: np.ndarray, first: np.ndarray, pgd: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    first_255 = first * 255.0
    pgd_255 = pgd * 255.0
    low = float(min(first_255.min(), pgd_255.min())) - 2.0
    high = float(max(first_255.max(), pgd_255.max())) + 2.0
    figure, axis = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    axis.plot([low, high], [low, high], color="0.45", linestyle="--", linewidth=1)
    axis.scatter(first_255, pgd_255, s=30, color="#2c7fb8", zorder=3)
    for index, x_value, y_value in zip(indices, first_255, pgd_255):
        axis.annotate(str(int(index)), (x_value, y_value), xytext=(3, 2),
                      textcoords="offset points", fontsize=6)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel(r"First-order estimate ($\epsilon\times255$)")
    axis.set_ylabel(r"PGD-slope extrapolation ($\epsilon\times255$)")
    axis.set_title("AdvMoE initialization: per-input route-boundary estimates")
    axis.grid(alpha=0.2)
    figure.savefig(path, format="svg")
    plt.close(figure)


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_dir = _inside(Path(config["advmoe_result_dir"]), MOE_ROOT)
    output_dir = _inside(Path(config["output_dir"]), MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refuses to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    input_path = result_dir / "inputs.npz"
    prepare_path = result_dir / "prepare.json"
    bounds_path = result_dir / "crown_bounds.json"
    with np.load(input_path, allow_pickle=False) as arrays:
        indices = arrays["dataset_indices"].astype(np.int64)
        clean_margin = arrays["clean_margin"].astype(np.float64)
        gradient_l1 = arrays["gradient_l1"].astype(np.float64)
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    epsilon = float(config["attack_epsilon"])
    attack_rows = {
        float(row["epsilon"]): row for row in prepare.get("attack_rows", [])
    }
    if epsilon not in attack_rows:
        raise RuntimeError("registered attack epsilon is absent from source artifact")
    compression = np.asarray(
        attack_rows[epsilon]["margin_compression_fraction"], dtype=np.float64
    )
    first_order, pgd_extrapolation = compute_boundary_estimates(
        clean_margin, gradient_l1, compression, epsilon
    )

    rt_path = _inside(Path(config["rt_er_radii_artifact"]), MOE_ROOT)
    with np.load(rt_path, allow_pickle=False) as arrays:
        rt_radii = arrays["radii"].astype(np.float64)
        rt_seeds = arrays["seeds"].astype(np.int64)
    rt_aggregate_median = float(np.median(rt_radii))
    seed0_slot = int(np.flatnonzero(rt_seeds == 0)[0])
    rt_seed0_median = float(np.median(rt_radii[seed0_slot]))

    ratio = pgd_extrapolation / first_order
    symmetric_relative = np.abs(pgd_extrapolation - first_order) / (
        (pgd_extrapolation + first_order) / 2.0
    )
    pearson = float(np.corrcoef(first_order, pgd_extrapolation)[0, 1])
    spearman = spearmanr(first_order, pgd_extrapolation)

    csv_path = output_dir / "per_sample.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_rank",
                "clean_margin",
                "margin_gradient_l1",
                "compression_at_8_over_255",
                "first_order_radius",
                "first_order_radius_x255",
                "pgd_extrapolated_radius",
                "pgd_extrapolated_radius_x255",
                "pgd_over_first_order",
            ]
        )
        for row in zip(
            indices,
            clean_margin,
            gradient_l1,
            compression,
            first_order,
            first_order * 255.0,
            pgd_extrapolation,
            pgd_extrapolation * 255.0,
            ratio,
        ):
            writer.writerow(row)

    figure_path = output_dir / "per_sample_boundary_estimates.svg"
    _plot(figure_path, indices, first_order, pgd_extrapolation)
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "sources": {
            "inputs": {"path": str(input_path), "sha256": _sha256(input_path)},
            "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
            "bounds": {"path": str(bounds_path), "sha256": _sha256(bounds_path)},
            "rt_er_radii": {"path": str(rt_path), "sha256": _sha256(rt_path)},
        },
        "gradient_identity": "gradient of clean-route margin r_clean-r_competitor with respect to unit-pixel input",
        "samples": int(len(indices)),
        "attack_epsilon": epsilon,
        "first_order_radius": _distribution(first_order),
        "first_order_radius_x255": _distribution(first_order * 255.0),
        "pgd_extrapolated_radius": _distribution(pgd_extrapolation),
        "pgd_extrapolated_radius_x255": _distribution(pgd_extrapolation * 255.0),
        "paired_agreement": {
            "pgd_over_first_order": _distribution(ratio),
            "symmetric_relative_difference": _distribution(symmetric_relative),
            "pearson": pearson,
            "spearman": float(spearman.statistic),
            "spearman_pvalue": float(spearman.pvalue),
            "within_3_percent": int(np.sum(np.abs(ratio - 1.0) <= 0.03)),
            "within_5_percent": int(np.sum(np.abs(ratio - 1.0) <= 0.05)),
            "within_10_percent": int(np.sum(np.abs(ratio - 1.0) <= 0.10)),
        },
        "rt_er_exact_unit_pixel_reference": {
            "aggregate_median": rt_aggregate_median,
            "aggregate_median_x255": rt_aggregate_median * 255.0,
            "seed0_median": rt_seed0_median,
            "seed0_median_x255": rt_seed0_median * 255.0,
        },
        "architecture_regime_ratio": {
            "first_order_median_over_rt_er_aggregate": float(
                np.median(first_order) / rt_aggregate_median
            ),
            "pgd_extrapolated_median_over_rt_er_aggregate": float(
                np.median(pgd_extrapolation) / rt_aggregate_median
            ),
        },
        "artifacts": {
            "per_sample": {"path": str(csv_path), "sha256": _sha256(csv_path)},
            "figure": {"path": str(figure_path), "sha256": _sha256(figure_path)},
        },
        "interpretation": config["interpretation"],
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
