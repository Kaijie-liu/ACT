"""Run a frozen multi-seed census of the official untrained RT-ER router."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from act.back_end.moe import affine_top1_route_boundary_batch
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_route_telemetry import (
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _raw_dataset_hash,
    fold_official_router,
)
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/icml2025_router_init_census_r2.json"
BLACKWELL_PYTHON = Path("/data1/Kane/MOE/envs/rt-er-blackwell/bin/python")
BLACKWELL_JPEG = Path("/data1/Kane/MOE/envs/rt-er-blackwell/lib/libjpeg.so.8")
WORKER = PROJECT_ROOT / "act/pipeline/moe/icml2025_router_init_worker.py"


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize_seed_fractions(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("seed fractions must be a finite nonempty vector")
    return {
        "mean": float(np.mean(array)),
        "std_population": float(np.std(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0].astype(np.float64) / int(np.sum(counts))
    return float(-np.sum(probabilities * np.log(probabilities)))


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"router-init census refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("router-init census config is not frozen")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")
    reference = _inside(Path(config["seed0_reference"]["per_input"]), WRITE_ROOT)
    if _sha256(reference) != config["seed0_reference"]["per_input_sha256"]:
        raise RuntimeError("seed-0 per-input reference changed")
    torchvision_root = Path(get_torchvision_data_root()).resolve()
    if not torchvision_root.is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision root escapes /data1/Kane/MOE")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA grid oracle is unavailable")

    output_dir.mkdir(parents=True)
    worker_dir = output_dir / "router_initializations"
    worker_log = output_dir / "router_initializations.log"
    command = [
        str(BLACKWELL_PYTHON),
        str(WORKER),
        "--config",
        str(config_path),
        "--output-dir",
        str(worker_dir),
    ]
    with worker_log.open("xb") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "LD_PRELOAD": str(BLACKWELL_JPEG),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode or not (worker_dir / "manifest.json").is_file():
        raise RuntimeError("official router-initialization worker failed; artifact retained")
    worker = json.loads((worker_dir / "manifest.json").read_text(encoding="utf-8"))
    if not all(worker["seed0_bitwise_checkpoint_match"].values()):
        raise RuntimeError("worker did not reproduce the seed-0 router")
    arrays_path = _inside(Path(worker["arrays"]["path"]), WRITE_ROOT)
    if _sha256(arrays_path) != worker["arrays"]["sha256"]:
        raise RuntimeError("router-initialization arrays changed")

    import torchvision.datasets as datasets

    dataset = datasets.CIFAR10(
        root=str(torchvision_root / "CIFAR10" / "raw"),
        train=False,
        download=False,
    )
    raw_images = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    points = raw_images.transpose(0, 3, 1, 2).astype(np.float64).reshape(len(dataset), -1) / 255.0
    if len(points) != int(config["dataset"]["ordered_samples"]):
        raise RuntimeError("official CIFAR-10 test-set size changed")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        seeds = arrays["seeds"].copy()
        weights = arrays["weights"].copy()
        biases = arrays["biases"].copy()
    if seeds.tolist() != config["initialization"]["seeds"]:
        raise RuntimeError("worker seed order changed")

    epsilons = [float(value) / 255.0 for value in config["epsilon_over_255"]]
    oracle = config["oracle"]
    all_radii: list[np.ndarray] = []
    all_lowers: list[np.ndarray] = []
    all_uppers: list[np.ndarray] = []
    all_routes: list[np.ndarray] = []
    all_competitors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for slot, seed in enumerate(seeds):
        folded_weight, folded_bias = fold_official_router(weights[slot], biases[slot])
        boundary = affine_top1_route_boundary_batch(
            folded_weight,
            folded_bias,
            points,
            input_lower=float(oracle["input_lower"]),
            input_upper=float(oracle["input_upper"]),
            outward_absolute=float(oracle["outward_absolute"]),
            outward_relative=float(oracle["outward_relative"]),
            compute_device=str(oracle["compute_device"]),
            capacity_grid_steps=int(oracle["capacity_grid_steps"]),
        )
        counts = np.bincount(boundary.clean_experts, minlength=4)
        entropy = _entropy(counts)
        epsilon_rows = {}
        for epsilon_over_255, epsilon in zip(config["epsilon_over_255"], epsilons):
            epsilon_rows[str(float(epsilon_over_255))] = {
                "strict_count": int(np.count_nonzero(boundary.radii < epsilon)),
                "strict_fraction": float(np.mean(boundary.radii < epsilon)),
                "reachable_count": int(np.count_nonzero(boundary.radius_uppers <= epsilon)),
                "stable_count": int(np.count_nonzero(epsilon < boundary.radius_lowers)),
                "undecided_count": int(
                    np.count_nonzero(
                        (boundary.radius_lowers <= epsilon)
                        & (epsilon < boundary.radius_uppers)
                    )
                ),
            }
        pair_counts = Counter(
            tuple(sorted((int(clean), int(other))))
            for clean, other in zip(boundary.clean_experts, boundary.boundary_competitors)
        )
        rows.append(
            {
                "seed": int(seed),
                "route_load_counts": counts.tolist(),
                "route_load_probabilities": (counts / len(points)).tolist(),
                "load_entropy": entropy,
                "effective_experts": math.exp(entropy),
                "maximum_expert_load": float(np.max(counts) / len(points)),
                "radius_median": float(np.median(boundary.radii)),
                "radius_p90": float(np.quantile(boundary.radii, 0.9)),
                "radius_maximum": float(np.max(boundary.radii)),
                "epsilon_census": epsilon_rows,
                "unordered_boundary_pair_counts": {
                    f"{key[0]}-{key[1]}": int(value)
                    for key, value in sorted(pair_counts.items())
                },
            }
        )
        all_radii.append(boundary.radii.copy())
        all_lowers.append(boundary.radius_lowers.copy())
        all_uppers.append(boundary.radius_uppers.copy())
        all_routes.append(boundary.clean_experts.copy())
        all_competitors.append(boundary.boundary_competitors.copy())

    with np.load(reference, allow_pickle=False) as seed0_reference:
        radius_errors = {
            "radii": float(np.max(np.abs(all_radii[0] - seed0_reference["radii"]))),
            "radius_lowers": float(
                np.max(np.abs(all_lowers[0] - seed0_reference["radius_lowers"]))
            ),
            "radius_uppers": float(
                np.max(np.abs(all_uppers[0] - seed0_reference["radius_uppers"]))
            ),
        }
        epsilon_classifications_exact = True
        for epsilon in epsilons:
            epsilon_classifications_exact &= bool(
                np.array_equal(all_radii[0] < epsilon, seed0_reference["radii"] < epsilon)
                and np.array_equal(
                    all_uppers[0] <= epsilon,
                    seed0_reference["radius_uppers"] <= epsilon,
                )
                and np.array_equal(
                    epsilon < all_lowers[0],
                    epsilon < seed0_reference["radius_lowers"],
                )
            )
        seed0_checks = {
            "radius_maximum_abs_errors": radius_errors,
            "radius_within_atol": bool(
                max(radius_errors.values())
                <= float(config["seed0_reference"]["reference_radius_atol"])
            ),
            "clean_experts_exact": bool(np.array_equal(all_routes[0], seed0_reference["clean_experts"])),
            "boundary_competitors_exact": bool(
                np.array_equal(all_competitors[0], seed0_reference["boundary_competitors"])
            ),
            "epsilon_classifications_exact": epsilon_classifications_exact,
        }
    if not (
        seed0_checks["radius_within_atol"]
        and seed0_checks["clean_experts_exact"]
        and seed0_checks["boundary_competitors_exact"]
        and seed0_checks["epsilon_classifications_exact"]
    ):
        raise RuntimeError("seed-0 census does not reproduce immutable telemetry")

    per_seed_path = output_dir / "per_seed.npz"
    with per_seed_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            seeds=seeds,
            radii=np.stack(all_radii),
            radius_lowers=np.stack(all_lowers),
            radius_uppers=np.stack(all_uppers),
            clean_experts=np.stack(all_routes),
            boundary_competitors=np.stack(all_competitors),
        )
        handle.flush()
        os.fsync(handle.fileno())
    aggregate = {}
    for epsilon_over_255 in config["epsilon_over_255"]:
        key = str(float(epsilon_over_255))
        aggregate[key] = summarize_seed_fractions(
            [row["epsilon_census"][key]["strict_fraction"] for row in rows]
        )
    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "worker": {"path": str(worker_dir / "manifest.json"), "sha256": _sha256(worker_dir / "manifest.json")},
        "worker_log": {"path": str(worker_log), "sha256": _sha256(worker_log)},
        "dataset": {
            "samples": len(points),
            "raw_sha256": _raw_dataset_hash(raw_images, labels),
        },
        "seeds": rows,
        "aggregate_strict_fraction": aggregate,
        "seed0_reference_checks": seed0_checks,
        "per_seed_artifact": {"path": str(per_seed_path), "sha256": _sha256(per_seed_path)},
        "runtime_seconds": time.monotonic() - started,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "scientific_scope": config["reporting"]["claim_scope"],
        "official_clone_clean_after": not bool(_repo_value("status", "--porcelain")),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
