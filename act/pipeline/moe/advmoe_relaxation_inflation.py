"""Compute a dimensionless CROWN relaxation-inflation diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def relaxation_inflation(
    clean_margin: np.ndarray,
    lower_bound: np.ndarray,
    attacked_margin: np.ndarray,
) -> np.ndarray:
    clean_margin = np.asarray(clean_margin, dtype=np.float64)
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    attacked_margin = np.asarray(attacked_margin, dtype=np.float64)
    denominator = clean_margin - attacked_margin
    if np.any(denominator <= 0):
        raise ValueError("attack must produce a strictly positive observed margin drop")
    numerator = clean_margin - lower_bound
    if np.any(numerator <= 0):
        raise ValueError("relaxation drop must be positive")
    return numerator / denominator


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_dir = _inside(Path(config["source_result_dir"]), MOE_ROOT)
    output_dir = _inside(Path(config["output_dir"]), MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refuses to reuse {output_dir}")
    input_path = source_dir / "inputs.npz"
    prepare_path = source_dir / "prepare.json"
    bounds_path = source_dir / "crown_bounds.json"
    with np.load(input_path, allow_pickle=False) as arrays:
        clean_margin = arrays["clean_margin"].astype(np.float64)
        ranks = arrays["dataset_indices"].astype(np.int64)
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    attacks = {float(row["epsilon"]): row for row in prepare["attack_rows"]}
    bound_rows = {float(row["epsilon"]): row for row in bounds["rows"]}
    epsilons = sorted(set(attacks) & set(bound_rows))
    matrix: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for epsilon in epsilons:
        attacked = np.asarray(attacks[epsilon]["attacked_margin"], dtype=np.float64)
        lower = np.asarray(bound_rows[epsilon]["lower_bounds"], dtype=np.float64)
        values = relaxation_inflation(clean_margin, lower, attacked)
        matrix.append(values)
        rows.append(
            {
                "epsilon": epsilon,
                "epsilon_over_255": epsilon * 255.0,
                "inflation": _distribution(values),
                "log10_inflation": _distribution(np.log10(values)),
            }
        )
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "per_sample.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_rank", *[f"epsilon_{value*255:g}_over_255" for value in epsilons]])
        for sample, rank in enumerate(ranks):
            writer.writerow([int(rank), *[values[sample] for values in matrix]])

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    axis.boxplot(
        [np.log10(values) for values in matrix],
        tick_labels=[f"{value*255:g}/255" for value in epsilons],
        showfliers=True,
    )
    axis.set_xlabel("Perturbation radius")
    axis.set_ylabel(r"$\log_{10}$ relaxation inflation")
    axis.set_title("AdvMoE init router: CROWN relaxation vs observed PGD drop")
    axis.grid(axis="y", alpha=0.2)
    figure_path = output_dir / "relaxation_inflation.svg"
    figure.savefig(figure_path, format="svg")
    plt.close(figure)

    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "sources": {
            "inputs": {"path": str(input_path), "sha256": _sha256(input_path)},
            "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
            "bounds": {"path": str(bounds_path), "sha256": _sha256(bounds_path)},
        },
        "definition": config["definition"],
        "samples": int(len(ranks)),
        "rows": rows,
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
