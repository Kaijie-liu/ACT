"""Render the frozen RT-ER route-geometry opening figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256


DEFAULT_RESULT = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/icml2025_rt_er/seed0_epoch020_route_telemetry_20260830.json"
)


def strict_route_census(
    radii: np.ndarray, thresholds_over_255: Sequence[float]
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(radii, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("route radii must be nonempty and finite")
    result: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds_over_255:
        count = int((values < float(threshold) / 255.0).sum())
        result[str(threshold)] = {
            "strict_count": count,
            "denominator": int(values.size),
            "strict_fraction": count / int(values.size),
        }
    return result


def unordered_confusion_counts(
    clean_experts: np.ndarray,
    boundary_competitors: np.ndarray,
    *,
    experts: int,
) -> np.ndarray:
    clean = np.asarray(clean_experts, dtype=np.int64).reshape(-1)
    boundary = np.asarray(boundary_competitors, dtype=np.int64).reshape(-1)
    if clean.shape != boundary.shape:
        raise ValueError("clean and boundary expert arrays differ")
    if np.any(clean == boundary):
        raise ValueError("a boundary competitor must differ from the clean expert")
    if np.any(clean < 0) or np.any(boundary < 0):
        raise ValueError("expert indices must be nonnegative")
    if np.any(clean >= int(experts)) or np.any(boundary >= int(experts)):
        raise ValueError("expert index exceeds configured width")
    counts = np.zeros((int(experts), int(experts)), dtype=np.int64)
    for first, second in zip(clean.tolist(), boundary.tolist()):
        low, high = sorted((int(first), int(second)))
        counts[low, high] += 1
    return counts


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _finish_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "ACT MoE route-geometry audit"},
    )
    plt.close(figure)


def run(result_path: Path, output_dir: Path) -> dict[str, Any]:
    result_path = _inside(result_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"route-geometry figure output already exists: {output_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    raw = result["raw_artifacts"]
    artifact = _inside(Path(raw["per_input_path"]), WRITE_ROOT)
    summary = _inside(Path(raw["summary_path"]), WRITE_ROOT)
    if _sha256(artifact) != raw["per_input_sha256"]:
        raise RuntimeError("route-geometry per-input artifact changed")
    if _sha256(summary) != raw["summary_sha256"]:
        raise RuntimeError("route-geometry summary changed")
    with np.load(artifact, allow_pickle=False) as arrays:
        radii = arrays["radii"].astype(np.float64)
        clean_experts = arrays["clean_experts"].astype(np.int64)
        boundary_competitors = arrays["boundary_competitors"].astype(np.int64)
    samples = int(radii.size)
    thresholds = [2.0, 4.0, 8.0]
    census = strict_route_census(radii, thresholds)
    load_counts = np.bincount(clean_experts, minlength=4)
    confusion = unordered_confusion_counts(
        clean_experts, boundary_competitors, experts=4
    )
    if int(load_counts.sum()) != samples or int(confusion.sum()) != samples:
        raise RuntimeError("route-geometry accounting does not close")

    output_dir.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "act-moe-rt-er-seed0"})

    ordered = np.sort(radii * 255.0)
    empirical = 100.0 * np.arange(1, samples + 1) / samples
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(ordered, empirical, color="#1f5a94", linewidth=2.0)
    for threshold, color in zip(thresholds, ("#d9822b", "#9c4dcc", "#2f855a")):
        fraction = 100.0 * float(census[str(threshold)]["strict_fraction"])
        axis.axvline(threshold, color=color, linestyle="--", linewidth=1.2)
        axis.text(
            threshold,
            max(4.0, fraction - 8.0),
            f"{threshold:g}/255: {fraction:.2f}%",
            color=color,
            rotation=90,
            va="top",
            ha="right",
        )
    axis.set_xlabel(r"Exact nearest route-boundary radius ($\epsilon\times255$)")
    axis.set_ylabel("Test inputs below radius (%)")
    axis.set_xlim(left=0.0, right=max(8.2, float(ordered[-1]) * 1.03))
    axis.set_ylim(0.0, 101.0)
    axis.grid(alpha=0.25)
    axis.set_title("Official-code RT-ER: static-router boundary census")
    _finish_figure(figure, output_dir / "route_boundary_cdf.svg")

    load_percent = 100.0 * load_counts / samples
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    bars = axis.bar(np.arange(4), load_percent, color="#3976a8")
    axis.axhline(25.0, color="#555555", linestyle="--", linewidth=1.1)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in load_percent])
    axis.set_xticks(np.arange(4), [f"Expert {index}" for index in range(4)])
    axis.set_ylabel("Clean test-set load (%)")
    axis.set_ylim(0.0, max(45.0, float(load_percent.max()) + 6.0))
    axis.set_title("Static-router expert load")
    axis.grid(axis="y", alpha=0.25)
    _finish_figure(figure, output_dir / "route_load.svg")

    percentages = 100.0 * confusion.astype(np.float64) / samples
    shown = percentages.copy()
    shown[np.tril_indices(4)] = np.nan
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    image = axis.imshow(shown, cmap="Blues", vmin=0.0, vmax=float(np.nanmax(shown)))
    for first in range(4):
        for second in range(first + 1, 4):
            axis.text(
                second,
                first,
                f"{confusion[first, second]}\n({percentages[first, second]:.2f}%)",
                ha="center",
                va="center",
                color=("white" if percentages[first, second] > 20.0 else "black"),
            )
    axis.set_xticks(np.arange(4), [f"E{index}" for index in range(4)])
    axis.set_yticks(np.arange(4), [f"E{index}" for index in range(4)])
    axis.set_xlabel("Other expert in nearest boundary pair")
    axis.set_ylabel("First expert in unordered pair")
    axis.set_title("Nearest route-boundary pair over 10,000 inputs")
    figure.colorbar(image, ax=axis, label="Test inputs (%)")
    _finish_figure(figure, output_dir / "route_confusion.svg")

    table_path = output_dir / "route_geometry_table.csv"
    with table_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "key", "count", "denominator", "percent"])
        for threshold in thresholds:
            row = census[str(threshold)]
            writer.writerow(
                [
                    "strict_route_boundary",
                    f"<{threshold:g}/255",
                    row["strict_count"],
                    row["denominator"],
                    100.0 * float(row["strict_fraction"]),
                ]
            )
        for expert, count in enumerate(load_counts.tolist()):
            writer.writerow(["expert_load", expert, count, samples, 100.0 * count / samples])
        for first in range(4):
            for second in range(first + 1, 4):
                count = int(confusion[first, second])
                writer.writerow(
                    ["route_boundary_pair", f"{first}-{second}", count, samples, 100.0 * count / samples]
                )
        handle.flush()
        os.fsync(handle.fileno())

    files = [
        "route_boundary_cdf.svg",
        "route_load.svg",
        "route_confusion.svg",
        "route_geometry_table.csv",
    ]
    manifest = {
        "schema_version": 1,
        "status": "COMPLETED",
        "source_result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "source_artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "samples": samples,
        "strict_route_boundary_census": census,
        "expert_load_counts": load_counts.tolist(),
        "expert_load_probabilities": (load_counts / samples).tolist(),
        "unordered_boundary_pair_counts": {
            f"{first}-{second}": int(confusion[first, second])
            for first in range(4)
            for second in range(first + 1, 4)
        },
        "outputs": {
            name: {"path": str(output_dir / name), "sha256": _sha256(output_dir / name)}
            for name in files
        },
        "scope": "seed0 epoch20 static-router geometry; exact final router geometry if the frozen drift guards continue to pass",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.result, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
