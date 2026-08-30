"""Render the audited CIFAR-10 RT-ER router initialization lottery."""

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
    / "act/pipeline/moe/results/icml2025_rt_er/router_init_census_k20_20260830.json"
)


def formal_stable_fraction_matrix(
    radius_lowers: np.ndarray, epsilon_over_255: Sequence[float]
) -> np.ndarray:
    lower = np.asarray(radius_lowers, dtype=np.float64)
    if lower.ndim != 2 or not lower.size or not np.all(np.isfinite(lower)):
        raise ValueError("radius lower bounds must be a finite seed-by-sample matrix")
    epsilons = np.asarray(epsilon_over_255, dtype=np.float64).reshape(-1) / 255.0
    if not epsilons.size or np.any(epsilons <= 0.0):
        raise ValueError("epsilon grid must be positive and nonempty")
    return np.mean(epsilons[None, :, None] < lower[:, None, :], axis=2)


def maximum_route_boundary(
    radii: np.ndarray, seeds: np.ndarray
) -> dict[str, float | int]:
    values = np.asarray(radii, dtype=np.float64)
    seed_values = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != seed_values.size:
        raise ValueError("radii and seeds have incompatible shapes")
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("radii must be finite and nonempty")
    seed_slot, sample = np.unravel_index(int(np.argmax(values)), values.shape)
    value = float(values[seed_slot, sample])
    return {
        "radius": value,
        "radius_over_255": value * 255.0,
        "seed": int(seed_values[seed_slot]),
        "sample_index": int(sample),
    }


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
        metadata={"Date": None, "Creator": "ACT MoE router-init census audit"},
    )
    plt.close(figure)


def run(result_path: Path, output_dir: Path) -> dict[str, Any]:
    result_path = _inside(result_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"router-init figure output already exists: {output_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED_AUDITED":
        raise RuntimeError("router-init census result is not audited")
    raw = result["raw_result"]
    artifact = _inside(Path(raw["directory"]) / "per_seed.npz", WRITE_ROOT)
    summary = _inside(Path(raw["directory"]) / "summary.json", WRITE_ROOT)
    if _sha256(artifact) != raw["per_seed_sha256"]:
        raise RuntimeError("router-init per-seed artifact changed")
    if _sha256(summary) != raw["summary_sha256"]:
        raise RuntimeError("router-init summary changed")
    with np.load(artifact, allow_pickle=False) as arrays:
        seeds = arrays["seeds"].astype(np.int64)
        radii = arrays["radii"].astype(np.float64)
        lowers = arrays["radius_lowers"].astype(np.float64)
        uppers = arrays["radius_uppers"].astype(np.float64)
    epsilon_over_255 = [0.5, 1.0, 2.0, 4.0, 8.0]
    stable = formal_stable_fraction_matrix(lowers, epsilon_over_255)
    reachable = np.stack(
        [np.mean(uppers <= value / 255.0, axis=1) for value in epsilon_over_255],
        axis=1,
    )
    undecided = 1.0 - stable - reachable
    if np.any(undecided < -1e-12):
        raise RuntimeError("stable/reachable accounting overlaps")
    maximum = maximum_route_boundary(radii, seeds)
    per_seed_maximum = np.max(radii, axis=1) * 255.0
    stable_at_eight = stable[:, -1]
    empty_seeds = seeds[stable_at_eight == 0.0]

    output_dir.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "act-moe-rt-er-init-k20"})
    positions = np.arange(len(epsilon_over_255), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.boxplot(
        [100.0 * stable[:, slot] for slot in range(stable.shape[1])],
        positions=positions,
        widths=0.52,
        patch_artist=True,
        boxprops={"facecolor": "#cfe5f3", "edgecolor": "#1f5a94"},
        medianprops={"color": "#b23b3b", "linewidth": 1.6},
        whiskerprops={"color": "#1f5a94"},
        capprops={"color": "#1f5a94"},
    )
    for seed_slot in range(len(seeds)):
        jitter = ((seed_slot % 5) - 2) * 0.025
        axis.scatter(
            positions + jitter,
            100.0 * stable[seed_slot],
            s=14,
            color="#334e68",
            alpha=0.68,
            linewidths=0,
        )
    axis.set_yscale("symlog", linthresh=0.01, linscale=0.7)
    axis.set_xticks(positions, [f"{value:g}/255" for value in epsilon_over_255])
    axis.set_ylabel("Formally route-stable test inputs (%)")
    axis.set_xlabel(r"$L_\infty$ radius")
    axis.set_title("RT-ER initialization lottery across 20 official constructions")
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.text(
        positions[-1],
        0.0007,
        f"{len(empty_seeds)}/20 seeds: empty",
        ha="center",
        va="bottom",
        color="#7c2d12",
    )
    _finish_figure(figure, output_dir / "route_stability_by_seed.svg")

    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.scatter(seeds, per_seed_maximum, color="#1f5a94", s=30)
    axis.axhline(8.0, color="#b23b3b", linestyle="--", linewidth=1.2, label="8/255")
    axis.scatter(
        [maximum["seed"]],
        [maximum["radius_over_255"]],
        color="#d97706",
        s=52,
        zorder=3,
    )
    axis.annotate(
        f"global max {maximum['radius_over_255']:.3f}/255\nseed {maximum['seed']}, sample {maximum['sample_index']}",
        xy=(maximum["seed"], maximum["radius_over_255"]),
        xytext=(28, -34),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#6b4f1d"},
        ha="left",
        va="top",
    )
    axis.set_xticks(seeds)
    axis.set_xlabel("Official construction seed")
    axis.set_ylabel(r"Maximum route-boundary radius ($\epsilon\times255$)")
    axis.set_title("Largest observed route-stable radius per initialization")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    _finish_figure(figure, output_dir / "maximum_route_boundary_by_seed.svg")

    table_path = output_dir / "router_init_census_table.csv"
    with table_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "epsilon_over_255",
                "stable_count",
                "reachable_count",
                "undecided_count",
                "maximum_radius_over_255",
            ]
        )
        samples = radii.shape[1]
        for seed_slot, seed in enumerate(seeds):
            for epsilon_slot, epsilon in enumerate(epsilon_over_255):
                writer.writerow(
                    [
                        int(seed),
                        epsilon,
                        int(round(stable[seed_slot, epsilon_slot] * samples)),
                        int(round(reachable[seed_slot, epsilon_slot] * samples)),
                        int(round(max(0.0, undecided[seed_slot, epsilon_slot]) * samples)),
                        per_seed_maximum[seed_slot],
                    ]
                )
        handle.flush()
        os.fsync(handle.fileno())

    outputs = [
        "route_stability_by_seed.svg",
        "maximum_route_boundary_by_seed.svg",
        "router_init_census_table.csv",
    ]
    manifest = {
        "schema_version": 1,
        "status": "COMPLETED",
        "source_result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "source_artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "seeds": seeds.tolist(),
        "samples_per_seed": int(radii.shape[1]),
        "epsilon_over_255": epsilon_over_255,
        "formally_stable_fraction": stable.tolist(),
        "formally_reachable_fraction": reachable.tolist(),
        "undecided_fraction": np.maximum(undecided, 0.0).tolist(),
        "empty_route_stable_seeds_at_8_over_255": empty_seeds.tolist(),
        "maximum_route_boundary": maximum,
        "per_seed_maximum_radius_over_255": per_seed_maximum.tolist(),
        "outputs": {
            name: {"path": str(output_dir / name), "sha256": _sha256(output_dir / name)}
            for name in outputs
        },
        "scope": "CIFAR-10 route-invariance applicability over 20 exact official full-model initializations; no output certificate",
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
