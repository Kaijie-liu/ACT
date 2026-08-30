"""Render the audited CIFAR-10/TinyImageNet router-applicability figure."""

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


DEFAULT_CIFAR = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/icml2025_rt_er/router_init_census_k20_20260830.json"
)
DEFAULT_TINY = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/icml2025_rt_er/"
    "tinyimagenet_router_census_k20_20260830.json"
)


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
        metadata={"Date": None, "Creator": "ACT cross-dataset router census audit"},
    )
    plt.close(figure)


def _boxplot_panel(
    axis: plt.Axes,
    fractions: np.ndarray,
    epsilon_over_255: Sequence[float],
    title: str,
    color: str,
) -> None:
    values = np.asarray(fractions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(epsilon_over_255):
        raise ValueError("stable fractions must be seed by epsilon")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("stable fractions escape [0,1]")
    positions = np.arange(values.shape[1], dtype=np.float64)
    axis.boxplot(
        [100.0 * values[:, slot] for slot in range(values.shape[1])],
        positions=positions,
        widths=0.52,
        patch_artist=True,
        boxprops={"facecolor": color, "edgecolor": "#334e68"},
        medianprops={"color": "#b23b3b", "linewidth": 1.6},
        whiskerprops={"color": "#334e68"},
        capprops={"color": "#334e68"},
    )
    for seed_slot in range(values.shape[0]):
        jitter = ((seed_slot % 5) - 2) * 0.025
        axis.scatter(
            positions + jitter,
            100.0 * values[seed_slot],
            s=13,
            color="#263849",
            alpha=0.64,
            linewidths=0,
        )
    axis.set_yscale("symlog", linthresh=0.01, linscale=0.7)
    axis.set_xticks(positions, [f"{value:g}/255" for value in epsilon_over_255])
    axis.set_title(title)
    axis.set_xlabel(r"$L_\infty$ radius")
    axis.grid(axis="y", which="both", alpha=0.23)


def _tiny_fraction_matrix(arrays: dict[str, np.ndarray], domain: str) -> np.ndarray:
    stable = np.asarray(arrays[f"{domain}__stable"], dtype=np.bool_)
    if stable.ndim != 3:
        raise ValueError("TinyImageNet stable array must be seed by sample by epsilon")
    return np.mean(stable, axis=1)


def run(cifar_path: Path, tiny_path: Path, output_dir: Path) -> dict[str, Any]:
    cifar_path = _inside(cifar_path, PROJECT_ROOT)
    tiny_path = _inside(tiny_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"cross-dataset figure output exists: {output_dir}")
    cifar = json.loads(cifar_path.read_text(encoding="utf-8"))
    tiny = json.loads(tiny_path.read_text(encoding="utf-8"))
    if cifar.get("status") != "COMPLETED_AUDITED" or tiny.get("status") != "COMPLETED_AUDITED":
        raise RuntimeError("both router censuses must be independently audited")

    cifar_npz = _inside(
        Path(cifar["raw_result"]["directory"]) / "per_seed.npz", WRITE_ROOT
    )
    tiny_npz = _inside(
        Path(tiny["raw_result"]["directory"]) / "per_seed.npz", WRITE_ROOT
    )
    if _sha256(cifar_npz) != cifar["raw_result"]["per_seed_sha256"]:
        raise RuntimeError("CIFAR census artifact changed")
    if _sha256(tiny_npz) != tiny["raw_result"]["per_seed_sha256"]:
        raise RuntimeError("TinyImageNet census artifact changed")
    with np.load(cifar_npz, allow_pickle=False) as artifact:
        cifar_lowers = artifact["radius_lowers"].astype(np.float64)
        cifar_seeds = artifact["seeds"].astype(np.int64)
    with np.load(tiny_npz, allow_pickle=False) as artifact:
        tiny_arrays = {name: artifact[name].copy() for name in artifact.files}
    epsilon_over_255 = tiny_arrays["epsilon_over_255"].astype(np.float64)
    epsilons = epsilon_over_255 / 255.0
    cifar_stable = np.stack(
        [np.mean(value < cifar_lowers, axis=1) for value in epsilons], axis=1
    )
    tiny_primary = _tiny_fraction_matrix(tiny_arrays, "official_post_resize_224")
    tiny_raw = _tiny_fraction_matrix(tiny_arrays, "official_composed_raw_64")
    tiny_seeds = tiny_arrays["seeds"].astype(np.int64)
    if cifar_seeds.tolist() != tiny_seeds.tolist():
        raise RuntimeError("cross-dataset seed grids changed")

    output_dir.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "act-moe-figure2-census"})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    _boxplot_panel(
        axes[0], cifar_stable, epsilon_over_255, "CIFAR-10 · ResNet18-MoE", "#cfe5f3"
    )
    _boxplot_panel(
        axes[1],
        tiny_primary,
        epsilon_over_255,
        "TinyImageNet · ViT-MoE · official 224 domain",
        "#dce8c8",
    )
    axes[0].set_ylabel("Formally route-stable inputs (%)")
    figure.suptitle("Route-invariance applicability across 20 official constructions")
    _finish_figure(figure, output_dir / "figure2_cross_dataset_route_stability.svg")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(epsilon_over_255), dtype=np.float64)
    axis.plot(
        positions,
        100.0 * np.mean(tiny_primary, axis=0),
        marker="o",
        label="official post-resize 224 domain",
    )
    axis.plot(
        positions,
        100.0 * np.mean(tiny_raw, axis=0),
        marker="s",
        label="same router composed to raw 64 domain",
    )
    axis.set_yscale("symlog", linthresh=0.01, linscale=0.7)
    axis.set_xticks(positions, [f"{value:g}/255" for value in epsilon_over_255])
    axis.set_xlabel(r"$L_\infty$ radius")
    axis.set_ylabel("Mean formally route-stable inputs (%)")
    axis.set_title("TinyImageNet preprocessing-domain sensitivity")
    axis.grid(which="both", alpha=0.23)
    axis.legend()
    _finish_figure(figure, output_dir / "tinyimagenet_preprocessing_domains.svg")

    table_path = output_dir / "figure2_route_stability.csv"
    with table_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "architecture",
                "domain",
                "seed",
                "epsilon_over_255",
                "stable_fraction",
            ]
        )
        for dataset, architecture, domain, values in (
            ("CIFAR-10", "ResNet18-MoE", "official unit-pixel", cifar_stable),
            (
                "TinyImageNet",
                "ViT-MoE",
                "official post-resize 224",
                tiny_primary,
            ),
            (
                "TinyImageNet",
                "ViT-MoE",
                "same-router composed raw 64",
                tiny_raw,
            ),
        ):
            for seed_slot, seed in enumerate(tiny_seeds):
                for epsilon_slot, epsilon in enumerate(epsilon_over_255):
                    writer.writerow(
                        [
                            dataset,
                            architecture,
                            domain,
                            int(seed),
                            float(epsilon),
                            float(values[seed_slot, epsilon_slot]),
                        ]
                    )
        handle.flush()
        os.fsync(handle.fileno())

    output_names = [
        "figure2_cross_dataset_route_stability.svg",
        "tinyimagenet_preprocessing_domains.svg",
        "figure2_route_stability.csv",
    ]
    result = {
        "schema_version": 1,
        "status": "COMPLETED_AUDITED_INPUTS",
        "sources": {
            "cifar": {"path": str(cifar_path), "sha256": _sha256(cifar_path)},
            "tinyimagenet": {"path": str(tiny_path), "sha256": _sha256(tiny_path)},
        },
        "seeds": tiny_seeds.tolist(),
        "epsilon_over_255": epsilon_over_255.tolist(),
        "stable_fraction": {
            "cifar_official": cifar_stable.tolist(),
            "tinyimagenet_official_224": tiny_primary.tolist(),
            "tinyimagenet_composed_raw64": tiny_raw.tolist(),
        },
        "outputs": {
            name: {"path": str(output_dir / name), "sha256": _sha256(output_dir / name)}
            for name in output_names
        },
        "scope": "Router applicability only. Main figure compares official CIFAR-10 and official post-resize TinyImageNet domains; raw64 is secondary.",
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar", type=Path, default=DEFAULT_CIFAR)
    parser.add_argument("--tiny", type=Path, default=DEFAULT_TINY)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.cifar, args.tiny, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
