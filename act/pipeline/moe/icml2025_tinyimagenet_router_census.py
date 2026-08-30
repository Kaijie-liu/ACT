"""Run the official TinyImageNet MOE_ViT K=20 router census."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from act.back_end.moe import (
    fold_affine_input_map,
    fold_bilinear_resize_input_map,
)
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256


MOE_ROOT = Path("/data1/Kane/MOE")
OFFICIAL_REPO = MOE_ROOT / "baselines/Robust-MoE-Dual-Model"
OFFICIAL_COMMIT = "30ef94d77b5451595b82e739aa8938e1f4c4521f"
BLACKWELL_PYTHON = MOE_ROOT / "envs/rt-er-blackwell/bin/python"
BLACKWELL_JPEG = MOE_ROOT / "envs/rt-er-blackwell/lib/libjpeg.so.8"
WORKER = PROJECT_ROOT / "act/pipeline/moe/icml2025_tinyimagenet_router_worker.py"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/icml2025_tinyimagenet_router_census_r2.json"
)


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


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite and nonempty")
    return {
        "mean": float(np.mean(array)),
        "std_population": float(np.std(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def fixed_epsilon_route_partition_torch(
    weight: torch.Tensor,
    bias: torch.Tensor,
    points: torch.Tensor,
    epsilons: torch.Tensor,
    *,
    input_lower: float,
    input_upper: float,
    outward_absolute: float,
    outward_relative: float,
) -> dict[str, torch.Tensor]:
    """Classify exact affine top-1 reachability at fixed radii.

    For every clean/competitor pair and epsilon, the support maximization over
    the clipped L-infinity box is separable and exact.  Outward gap brackets
    distinguish proven reachability, proven stability, and numerical
    undecidedness without sorting all feature capacities.
    """

    if weight.ndim != 2 or bias.ndim != 1 or points.ndim != 2:
        raise ValueError("weight, bias, and points have invalid ranks")
    if weight.shape[0] < 2 or bias.shape[0] != weight.shape[0]:
        raise ValueError("router output shapes are inconsistent")
    if points.shape[1] != weight.shape[1]:
        raise ValueError("point width does not match router width")
    if epsilons.ndim != 1 or not epsilons.numel() or bool(torch.any(epsilons < 0)):
        raise ValueError("epsilon vector must be nonnegative and nonempty")
    if input_lower >= input_upper:
        raise ValueError("input box is empty")
    if outward_absolute < 0.0 or outward_relative < 0.0:
        raise ValueError("outward rounding tolerances must be nonnegative")
    if bool(torch.any(points < input_lower)) or bool(torch.any(points > input_upper)):
        raise ValueError("points escape the input box")

    scores = points @ weight.T + bias
    clean = torch.argmax(scores, dim=1)
    shape = (points.shape[0], epsilons.numel())
    minimum_gap = torch.full(shape, torch.inf, dtype=points.dtype, device=points.device)
    minimum_lower = minimum_gap.clone()
    minimum_upper = minimum_gap.clone()
    competitor = torch.full(shape, -1, dtype=torch.int64, device=points.device)
    negative_infinity = torch.full((), -torch.inf, dtype=points.dtype, device=points.device)
    positive_infinity = torch.full((), torch.inf, dtype=points.dtype, device=points.device)

    for clean_expert in range(weight.shape[0]):
        rows = torch.nonzero(clean == clean_expert, as_tuple=False).flatten()
        if not rows.numel():
            continue
        selected_points = points[rows]
        for other in range(weight.shape[0]):
            if other == clean_expert:
                continue
            difference = weight[clean_expert] - weight[other]
            absolute_difference = torch.abs(difference)
            capacities = torch.where(
                difference > 0,
                selected_points - input_lower,
                torch.where(
                    difference < 0,
                    input_upper - selected_points,
                    torch.zeros_like(selected_points),
                ),
            )
            margin = scores[rows, clean_expert] - scores[rows, other]
            support = torch.sum(
                absolute_difference[None, :, None]
                * torch.minimum(capacities[:, :, None], epsilons[None, None, :]),
                dim=1,
            )
            gap = margin[:, None] - support
            slack = float(outward_absolute) + float(outward_relative) * torch.maximum(
                torch.ones_like(gap), torch.abs(margin[:, None]) + torch.abs(support)
            )
            lower_gap = torch.nextafter(gap - slack, negative_infinity)
            upper_gap = torch.nextafter(gap + slack, positive_infinity)
            current = minimum_gap[rows]
            current_competitor = competitor[rows]
            improve = (gap < current) | (
                (gap == current)
                & ((current_competitor < 0) | (other < current_competitor))
            )
            minimum_gap[rows] = torch.minimum(current, gap)
            competitor[rows] = torch.where(
                improve,
                torch.full_like(current_competitor, other),
                current_competitor,
            )
            minimum_lower[rows] = torch.minimum(minimum_lower[rows], lower_gap)
            minimum_upper[rows] = torch.minimum(minimum_upper[rows], upper_gap)

    stable = minimum_lower > 0
    reachable = minimum_upper <= 0
    if bool(torch.any(stable & reachable)):
        raise RuntimeError("outward route partition overlaps")
    undecided = ~(stable | reachable)
    return {
        "clean_experts": clean,
        "nominal_minimum_gap": minimum_gap,
        "gap_lowers": minimum_lower,
        "gap_uppers": minimum_upper,
        "boundary_competitors": competitor,
        "formally_stable": stable,
        "formally_reachable": reachable,
        "undecided": undecided,
    }


def _ordered_validation_images(root: Path, expected: int) -> tuple[list[Path], str]:
    images = sorted((root / "val/images").glob("*.JPEG"))
    if len(images) != expected:
        raise RuntimeError(f"TinyImageNet validation count changed: {len(images)}")
    digest = hashlib.sha256()
    for path in images:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
    return images, digest.hexdigest()


def _load_image_chunk(paths: Sequence[Path]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape != (64, 64, 3):
            raise RuntimeError(f"unexpected TinyImageNet image shape at {path}")
        rows.append(array.transpose(2, 0, 1))
    return np.stack(rows)


def _normalization_vectors(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(config["router"]["normalization_mean_255"], dtype=np.float64)
    std = np.asarray(config["router"]["normalization_std_255"], dtype=np.float64)
    scale = np.broadcast_to((255.0 / std)[:, None, None], (3, 224, 224))
    shift = np.broadcast_to((-mean / std)[:, None, None], (3, 224, 224))
    return scale.reshape(-1), shift.reshape(-1)


def _seed_domain_summary(
    clean: np.ndarray,
    competitors: np.ndarray,
    stable: np.ndarray,
    reachable: np.ndarray,
    undecided: np.ndarray,
    epsilon_over_255: Sequence[float],
) -> dict[str, Any]:
    counts = np.bincount(clean, minlength=4)
    probabilities = counts.astype(np.float64) / clean.size
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    epsilon_rows: dict[str, Any] = {}
    for slot, epsilon in enumerate(epsilon_over_255):
        first_competitors = competitors[:, slot]
        pairs = Counter(
            tuple(sorted((int(route), int(other))))
            for route, other, is_reachable in zip(
                clean, first_competitors, reachable[:, slot]
            )
            if is_reachable
        )
        epsilon_rows[str(float(epsilon))] = {
            "stable_count": int(np.count_nonzero(stable[:, slot])),
            "stable_fraction": float(np.mean(stable[:, slot])),
            "reachable_count": int(np.count_nonzero(reachable[:, slot])),
            "reachable_fraction": float(np.mean(reachable[:, slot])),
            "undecided_count": int(np.count_nonzero(undecided[:, slot])),
            "unordered_reachable_competitor_pairs": {
                f"{left}-{right}": int(value)
                for (left, right), value in sorted(pairs.items())
            },
        }
    return {
        "route_load_counts": counts.tolist(),
        "route_load_probabilities": probabilities.tolist(),
        "load_entropy": entropy,
        "effective_experts": math.exp(entropy),
        "maximum_expert_load": float(np.max(probabilities)),
        "epsilon_census": epsilon_rows,
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"TinyImageNet census refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("TinyImageNet census config is not frozen")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")

    dataset = config["dataset"]
    archive = Path(dataset["archive_path"]).resolve()
    root = Path(dataset["root"]).resolve()
    if not archive.is_relative_to(MOE_ROOT) or not root.is_relative_to(MOE_ROOT):
        raise RuntimeError("TinyImageNet dataset path escapes /data1/Kane/MOE")
    if archive.stat().st_size != int(dataset["archive_bytes"]) or _sha256(
        archive
    ) != dataset["archive_sha256"]:
        raise RuntimeError("TinyImageNet archive identity changed")
    annotations = root / "val/val_annotations.txt"
    if _sha256(annotations) != dataset["val_annotations_sha256"]:
        raise RuntimeError("TinyImageNet validation annotations changed")
    image_paths, ordered_image_names_sha256 = _ordered_validation_images(
        root, int(dataset["ordered_samples"])
    )
    if not torch.cuda.is_available() and config["oracle"]["compute_device"] == "cuda":
        raise RuntimeError("configured CUDA fixed-epsilon oracle is unavailable")

    output_dir.mkdir(parents=True)
    worker_dir = output_dir / "router_initializations"
    worker_log = output_dir / "router_initializations.log"
    cache_root = Path(config["initialization"]["cache_root"]).resolve()
    if not cache_root.is_relative_to(MOE_ROOT):
        raise RuntimeError("TinyImageNet cache escapes /data1/Kane/MOE")
    cache_root.mkdir(parents=True, exist_ok=True)
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
                "HF_HOME": str(cache_root / "huggingface"),
                "HUGGINGFACE_HUB_CACHE": str(cache_root / "huggingface/hub"),
                "TORCH_HOME": str(cache_root / "torch"),
                "XDG_CACHE_HOME": str(cache_root / "xdg"),
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
        raise RuntimeError("official MOE_ViT router worker failed; artifact retained")
    worker = json.loads((worker_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays_path = Path(worker["arrays"]["path"]).resolve()
    if not arrays_path.is_relative_to(MOE_ROOT) or _sha256(arrays_path) != worker[
        "arrays"
    ]["sha256"]:
        raise RuntimeError("TinyImageNet router arrays changed")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        seeds = arrays["seeds"].copy()
        original_weights = arrays["weights"].copy()
        original_biases = arrays["biases"].copy()
    if seeds.tolist() != config["initialization"]["seeds"]:
        raise RuntimeError("TinyImageNet worker seed order changed")

    scale, shift = _normalization_vectors(config)
    weights_224: list[np.ndarray] = []
    biases_224: list[np.ndarray] = []
    weights_64: list[np.ndarray] = []
    for weight, bias in zip(original_weights, original_biases):
        pixel_weight, pixel_bias = fold_affine_input_map(weight, bias, scale, shift)
        raw_weight, raw_bias = fold_bilinear_resize_input_map(
            pixel_weight,
            pixel_bias,
            channels=3,
            input_size=(64, 64),
            output_size=(224, 224),
            align_corners=False,
            antialias=True,
        )
        weights_224.append(pixel_weight)
        biases_224.append(pixel_bias)
        weights_64.append(raw_weight)
        if not np.array_equal(raw_bias, pixel_bias):
            raise RuntimeError("zero-shift resize changed router bias")

    device = torch.device(config["oracle"]["compute_device"])
    tensor_weights_224 = torch.as_tensor(
        np.stack(weights_224), dtype=torch.float64, device=device
    )
    tensor_biases_224 = torch.as_tensor(
        np.stack(biases_224), dtype=torch.float64, device=device
    )
    tensor_weights_64 = torch.as_tensor(
        np.stack(weights_64), dtype=torch.float64, device=device
    )
    tensor_original_weights = torch.as_tensor(
        original_weights, dtype=torch.float64, device=device
    )
    tensor_original_biases = torch.as_tensor(
        original_biases, dtype=torch.float64, device=device
    )
    epsilon_over_255 = [float(value) for value in config["epsilon_over_255"]]
    epsilons = torch.as_tensor(
        np.asarray(epsilon_over_255) / 255.0, dtype=torch.float64, device=device
    )
    shape = (len(seeds), len(image_paths), len(epsilon_over_255))
    domain_arrays: dict[str, dict[str, np.ndarray]] = {}
    for domain in config["domains"]:
        domain_arrays[domain] = {
            "clean_experts": np.empty((len(seeds), len(image_paths)), dtype=np.int8),
            "competitors": np.empty(shape, dtype=np.int8),
            "stable": np.empty(shape, dtype=np.bool_),
            "reachable": np.empty(shape, dtype=np.bool_),
            "undecided": np.empty(shape, dtype=np.bool_),
        }
    literal_route_mismatches = np.zeros(len(seeds), dtype=np.int64)
    maximum_literal_score_drift = np.zeros(len(seeds), dtype=np.float64)
    fold_route_mismatches = np.zeros(len(seeds), dtype=np.int64)
    maximum_fold_score_drift = np.zeros(len(seeds), dtype=np.float64)
    resize_route_mismatches = np.zeros(len(seeds), dtype=np.int64)
    maximum_resize_score_drift = np.zeros(len(seeds), dtype=np.float64)
    mean = torch.as_tensor(
        config["router"]["normalization_mean_255"],
        dtype=torch.float16,
        device=device,
    )[None, :, None, None]
    std = torch.as_tensor(
        config["router"]["normalization_std_255"],
        dtype=torch.float16,
        device=device,
    )[None, :, None, None]
    oracle = config["oracle"]
    batch_size = int(oracle["batch_size"])
    started = time.monotonic()
    for start in range(0, len(image_paths), batch_size):
        stop = min(start + batch_size, len(image_paths))
        raw_uint8 = _load_image_chunk(image_paths[start:stop])
        raw_unit = torch.as_tensor(raw_uint8, dtype=torch.float64, device=device) / 255.0
        real_resized_unit = functional.interpolate(
            raw_unit,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        literal_resized_255 = functional.interpolate(
            torch.as_tensor(raw_uint8, dtype=torch.float16, device=device),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        literal_normalized = ((literal_resized_255 - mean) / std).double().flatten(1)
        points_224 = (literal_resized_255.double() / 255.0).flatten(1)
        real_points_224 = real_resized_unit.flatten(1)
        points_64 = raw_unit.flatten(1)
        for seed_slot in range(len(seeds)):
            result_224 = fixed_epsilon_route_partition_torch(
                tensor_weights_224[seed_slot],
                tensor_biases_224[seed_slot],
                points_224,
                epsilons,
                input_lower=float(oracle["input_lower"]),
                input_upper=float(oracle["input_upper"]),
                outward_absolute=float(oracle["outward_absolute"]),
                outward_relative=float(oracle["outward_relative"]),
            )
            result_64 = fixed_epsilon_route_partition_torch(
                tensor_weights_64[seed_slot],
                tensor_biases_224[seed_slot],
                points_64,
                epsilons,
                input_lower=float(oracle["input_lower"]),
                input_upper=float(oracle["input_upper"]),
                outward_absolute=float(oracle["outward_absolute"]),
                outward_relative=float(oracle["outward_relative"]),
            )
            for domain, result in (
                ("official_post_resize_224", result_224),
                ("official_composed_raw_64", result_64),
            ):
                arrays = domain_arrays[domain]
                arrays["clean_experts"][seed_slot, start:stop] = (
                    result["clean_experts"].cpu().numpy()
                )
                arrays["competitors"][seed_slot, start:stop] = (
                    result["boundary_competitors"].cpu().numpy()
                )
                arrays["stable"][seed_slot, start:stop] = (
                    result["formally_stable"].cpu().numpy()
                )
                arrays["reachable"][seed_slot, start:stop] = (
                    result["formally_reachable"].cpu().numpy()
                )
                arrays["undecided"][seed_slot, start:stop] = (
                    result["undecided"].cpu().numpy()
                )

            primary_scores = points_224 @ tensor_weights_224[seed_slot].T + tensor_biases_224[
                seed_slot
            ]
            real_resize_scores = (
                real_points_224 @ tensor_weights_224[seed_slot].T
                + tensor_biases_224[seed_slot]
            )
            raw_scores = points_64 @ tensor_weights_64[seed_slot].T + tensor_biases_224[
                seed_slot
            ]
            literal_scores = (
                literal_normalized @ tensor_original_weights[seed_slot].T
                + tensor_original_biases[seed_slot]
            )
            maximum_fold_score_drift[seed_slot] = max(
                maximum_fold_score_drift[seed_slot],
                float(torch.max(torch.abs(real_resize_scores - raw_scores)).item()),
            )
            maximum_literal_score_drift[seed_slot] = max(
                maximum_literal_score_drift[seed_slot],
                float(torch.max(torch.abs(primary_scores - literal_scores)).item()),
            )
            maximum_resize_score_drift[seed_slot] = max(
                maximum_resize_score_drift[seed_slot],
                float(torch.max(torch.abs(primary_scores - real_resize_scores)).item()),
            )
            fold_route_mismatches[seed_slot] += int(
                torch.count_nonzero(
                    torch.argmax(real_resize_scores, dim=1)
                    != torch.argmax(raw_scores, dim=1)
                ).item()
            )
            literal_route_mismatches[seed_slot] += int(
                torch.count_nonzero(
                    torch.argmax(primary_scores, dim=1)
                    != torch.argmax(literal_scores, dim=1)
                ).item()
            )
            resize_route_mismatches[seed_slot] += int(
                torch.count_nonzero(
                    torch.argmax(primary_scores, dim=1)
                    != torch.argmax(real_resize_scores, dim=1)
                ).item()
            )

    per_seed_path = output_dir / "per_seed.npz"
    payload: dict[str, np.ndarray] = {
        "seeds": seeds,
        "epsilon_over_255": np.asarray(epsilon_over_255, dtype=np.float64),
        "literal_route_mismatches": literal_route_mismatches,
        "maximum_literal_score_drift": maximum_literal_score_drift,
        "fold_route_mismatches": fold_route_mismatches,
        "maximum_fold_score_drift": maximum_fold_score_drift,
        "resize_route_mismatches": resize_route_mismatches,
        "maximum_resize_score_drift": maximum_resize_score_drift,
    }
    for domain, arrays in domain_arrays.items():
        for name, values in arrays.items():
            payload[f"{domain}__{name}"] = values
    with per_seed_path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())

    rows: list[dict[str, Any]] = []
    for seed_slot, seed in enumerate(seeds):
        domains = {}
        for domain, arrays in domain_arrays.items():
            domains[domain] = _seed_domain_summary(
                arrays["clean_experts"][seed_slot],
                arrays["competitors"][seed_slot],
                arrays["stable"][seed_slot],
                arrays["reachable"][seed_slot],
                arrays["undecided"][seed_slot],
                epsilon_over_255,
            )
        rows.append(
            {
                "seed": int(seed),
                "domains": domains,
                "preprocessing_audit": {
                    "fold_route_mismatches": int(fold_route_mismatches[seed_slot]),
                    "maximum_fold_score_drift": float(maximum_fold_score_drift[seed_slot]),
                    "literal_float16_route_mismatches": int(
                        literal_route_mismatches[seed_slot]
                    ),
                    "maximum_literal_float16_score_drift": float(
                        maximum_literal_score_drift[seed_slot]
                    ),
                    "real_vs_float16_resize_route_mismatches": int(
                        resize_route_mismatches[seed_slot]
                    ),
                    "maximum_real_vs_float16_resize_score_drift": float(
                        maximum_resize_score_drift[seed_slot]
                    ),
                },
            }
        )

    aggregate: dict[str, Any] = {}
    for domain, arrays in domain_arrays.items():
        aggregate[domain] = {}
        for epsilon_slot, epsilon in enumerate(epsilon_over_255):
            aggregate[domain][str(float(epsilon))] = {
                "stable_fraction": _summarize(
                    np.mean(arrays["stable"][:, :, epsilon_slot], axis=1).tolist()
                ),
                "reachable_fraction": _summarize(
                    np.mean(arrays["reachable"][:, :, epsilon_slot], axis=1).tolist()
                ),
                "undecided_sample_seed_pairs": int(
                    np.count_nonzero(arrays["undecided"][:, :, epsilon_slot])
                ),
            }

    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "scope": config["reporting"]["claim_scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "official_commit": OFFICIAL_COMMIT,
        "dataset": {
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "ordered_samples": len(image_paths),
            "ordered_image_names_sha256": ordered_image_names_sha256,
            "labels_used": False,
        },
        "worker": {
            "manifest": str(worker_dir / "manifest.json"),
            "manifest_sha256": _sha256(worker_dir / "manifest.json"),
            "arrays_sha256": _sha256(arrays_path),
        },
        "elapsed_seconds": time.monotonic() - started,
        "epsilon_over_255": epsilon_over_255,
        "rows": rows,
        "aggregate": aggregate,
        "preprocessing_audit": {
            "total_fold_route_mismatches": int(np.sum(fold_route_mismatches)),
            "maximum_fold_score_drift": float(np.max(maximum_fold_score_drift)),
            "total_literal_float16_route_mismatches": int(
                np.sum(literal_route_mismatches)
            ),
            "maximum_literal_float16_score_drift": float(
                np.max(maximum_literal_score_drift)
            ),
            "total_real_vs_float16_resize_route_mismatches": int(
                np.sum(resize_route_mismatches)
            ),
            "maximum_real_vs_float16_resize_score_drift": float(
                np.max(maximum_resize_score_drift)
            ),
            "literal_comparisons": len(seeds) * len(image_paths),
        },
        "per_seed": {"path": str(per_seed_path), "sha256": _sha256(per_seed_path)},
        "official_clone_clean_after": not bool(_repo_value("status", "--porcelain")),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
