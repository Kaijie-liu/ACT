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
    with np.load(input_path, allow_pickle=False) as arrays:
        clean_margin = arrays["clean_margin"].astype(np.float64)
        ranks = arrays["dataset_indices"].astype(np.int64)
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    attacks = {float(row["epsilon"]): row for row in prepare["attack_rows"]}
    registered_sources = config.get("bound_sources")
    if registered_sources is None:
        registered_sources = [
            {
                "label": "CROWN",
                "path": str(source_dir / "crown_bounds.json"),
            }
        ]
    layers: list[dict[str, Any]] = []
    layer_matrices: dict[str, list[np.ndarray]] = {}
    source_records: dict[str, dict[str, str]] = {}
    epsilons: list[float] | None = None
    for registered in registered_sources:
        label = str(registered["label"])
        if label in layer_matrices:
            raise ValueError(f"duplicate bound-source label: {label}")
        bounds_path = _inside(Path(registered["path"]), MOE_ROOT)
        bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
        bound_rows = {float(row["epsilon"]): row for row in bounds["rows"]}
        current_epsilons = sorted(set(attacks) & set(bound_rows))
        if epsilons is None:
            epsilons = current_epsilons
        elif current_epsilons != epsilons:
            raise RuntimeError("bound sources do not share the registered epsilon grid")
        matrix: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for epsilon in current_epsilons:
            attacked = np.asarray(
                attacks[epsilon]["attacked_margin"], dtype=np.float64
            )
            lower = np.asarray(
                bound_rows[epsilon]["lower_bounds"], dtype=np.float64
            )
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
        layer_matrices[label] = matrix
        layers.append({"label": label, "method": bounds.get("method"), "rows": rows})
        source_records[label] = {"path": str(bounds_path), "sha256": _sha256(bounds_path)}
    if epsilons is None or not layers:
        raise RuntimeError("no registered bound sources were evaluated")
    primary_label = str(config.get("primary_label", layers[-1]["label"]))
    primary = next((layer for layer in layers if layer["label"] == primary_label), None)
    if primary is None:
        raise ValueError("primary_label is absent from bound_sources")
    rows = primary["rows"]
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "per_sample.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        columns = [
            f"{label}_epsilon_{value*255:g}_over_255"
            for label in layer_matrices
            for value in epsilons
        ]
        writer.writerow(["sample_rank", *columns])
        for sample, rank in enumerate(ranks):
            writer.writerow(
                [
                    int(rank),
                    *[
                        values[sample]
                        for matrix in layer_matrices.values()
                        for values in matrix
                    ],
                ]
            )

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    labels = list(layer_matrices)
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    base = np.arange(len(epsilons), dtype=np.float64) + 1.0
    width = 0.68 / len(labels)
    for slot, label in enumerate(labels):
        offset = (slot - (len(labels) - 1) / 2.0) * width
        boxes = axis.boxplot(
            [np.log10(values) for values in layer_matrices[label]],
            positions=base + offset,
            widths=width * 0.82,
            showfliers=True,
            patch_artist=True,
        )
        color = colors[slot % len(colors)]
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in boxes["medians"]:
            median.set_color("black")
    axis.set_xticks(base, [f"{value*255:g}/255" for value in epsilons])
    from matplotlib.patches import Patch
    axis.legend(
        handles=[
            Patch(facecolor=colors[i % len(colors)], alpha=0.55, label=label)
            for i, label in enumerate(labels)
        ],
        title="Router bound",
    )
    axis.set_xlabel("Perturbation radius")
    axis.set_ylabel(r"$\log_{10}$ relaxation inflation")
    axis.set_title("AdvMoE init router: relaxation vs observed PGD drop")
    axis.grid(axis="y", alpha=0.2)
    figure_path = output_dir / "relaxation_inflation.svg"
    figure.savefig(figure_path, format="svg")
    plt.close(figure)

    source_payload: dict[str, Any] = {
        "inputs": {"path": str(input_path), "sha256": _sha256(input_path)},
        "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        "bound_sources": source_records,
    }
    # Preserve the original one-source result schema for existing consumers.
    if len(source_records) == 1 and "CROWN" in source_records:
        source_payload["bounds"] = source_records["CROWN"]
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "sources": source_payload,
        "definition": config["definition"],
        "samples": int(len(ranks)),
        "primary_label": primary_label,
        "rows": rows,
        "layers": layers,
        "median_ratios": {
            f"{labels[i]}_over_{labels[j]}": [
                float(
                    np.median(layer_matrices[labels[i]][epsilon_slot])
                    / np.median(layer_matrices[labels[j]][epsilon_slot])
                )
                for epsilon_slot in range(len(epsilons))
            ]
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
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
