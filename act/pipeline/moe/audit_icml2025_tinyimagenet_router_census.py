"""Independently audit the official TinyImageNet K=20 router census."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from act.back_end.moe import fold_affine_input_map, fold_bilinear_resize_input_map
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256


MOE_ROOT = Path("/data1/Kane/MOE")
OFFICIAL_REPO = MOE_ROOT / "baselines/Robust-MoE-Dual-Model"
OFFICIAL_COMMIT = "30ef94d77b5451595b82e739aa8938e1f4c4521f"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/icml2025_tinyimagenet_router_census_r3.json"
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


def _normalization_vectors(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(config["router"]["normalization_mean_255"], dtype=np.float64)
    std = np.asarray(config["router"]["normalization_std_255"], dtype=np.float64)
    scale = np.broadcast_to((255.0 / std)[:, None, None], (3, 224, 224))
    shift = np.broadcast_to((-mean / std)[:, None, None], (3, 224, 224))
    return scale.reshape(-1), shift.reshape(-1)


def _ordered_images(root: Path, expected: int) -> tuple[list[Path], str]:
    paths = sorted((root / "val/images").glob("*.JPEG"))
    if len(paths) != expected:
        raise RuntimeError("validation image count changed")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
    return paths, digest.hexdigest()


def _load_unit_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.shape != (64, 64, 3):
        raise RuntimeError(f"unexpected image shape at {path}")
    return value.transpose(2, 0, 1).astype(np.float64) / 255.0


def _direct_fixed_partition(
    weight: np.ndarray,
    bias: np.ndarray,
    point: np.ndarray,
    epsilons: np.ndarray,
    absolute: float,
    relative: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Independent scalar transcription of the clipped-box support formula."""

    scores = weight @ point + bias
    clean = int(np.argmax(scores))
    minimum_lower = np.full(epsilons.shape, np.inf, dtype=np.float64)
    minimum_upper = np.full(epsilons.shape, np.inf, dtype=np.float64)
    for other in range(len(scores)):
        if other == clean:
            continue
        difference = weight[clean] - weight[other]
        capacity = np.where(
            difference > 0.0,
            point,
            np.where(difference < 0.0, 1.0 - point, 0.0),
        )
        margin = scores[clean] - scores[other]
        for slot, epsilon in enumerate(epsilons):
            support = float(np.dot(np.abs(difference), np.minimum(capacity, epsilon)))
            gap = float(margin - support)
            slack = absolute + relative * max(1.0, abs(margin) + abs(support))
            lower = float(np.nextafter(gap - slack, -np.inf))
            upper = float(np.nextafter(gap + slack, np.inf))
            minimum_lower[slot] = min(minimum_lower[slot], lower)
            minimum_upper[slot] = min(minimum_upper[slot], upper)
    stable = minimum_lower > 0.0
    reachable = minimum_upper <= 0.0
    return clean, stable, reachable, ~(stable | reachable)


def run(config_path: Path, raw_dir: Path, output_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    raw_dir = _inside(raw_dir, WRITE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"audit output already exists: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = raw_dir / "summary.json"
    per_seed_path = raw_dir / "per_seed.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETED":
        raise RuntimeError("TinyImageNet census did not complete")

    issues: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    require(summary["config"]["sha256"] == _sha256(config_path), "config hash mismatch")
    require(summary["per_seed"]["sha256"] == _sha256(per_seed_path), "NPZ hash mismatch")
    require(summary["official_commit"] == OFFICIAL_COMMIT, "summary commit mismatch")
    require(_repo_value("rev-parse", "HEAD") == OFFICIAL_COMMIT, "clone HEAD changed")
    require(not bool(_repo_value("status", "--porcelain")), "official clone dirty")
    worker_manifest = Path(summary["worker"]["manifest"]).resolve()
    worker_arrays = worker_manifest.parent / "routers.npz"
    require(worker_manifest.is_relative_to(MOE_ROOT), "worker manifest escaped scope")
    require(_sha256(worker_manifest) == summary["worker"]["manifest_sha256"], "worker manifest hash mismatch")
    require(_sha256(worker_arrays) == summary["worker"]["arrays_sha256"], "worker arrays hash mismatch")
    preprocessing_manifest = Path(summary["preprocessing"]["manifest"]).resolve()
    require(preprocessing_manifest.is_relative_to(MOE_ROOT), "preprocessing manifest escaped scope")
    require(
        _sha256(preprocessing_manifest) == summary["preprocessing"]["manifest_sha256"],
        "preprocessing manifest hash mismatch",
    )
    preprocessing = json.loads(preprocessing_manifest.read_text(encoding="utf-8"))
    literal_resized_path = Path(
        preprocessing["outputs"]["literal_resized"]["path"]
    ).resolve()
    literal_scores_path = Path(preprocessing["outputs"]["literal_scores"]["path"]).resolve()
    require(
        _sha256(literal_resized_path) == summary["preprocessing"]["literal_resized_sha256"],
        "literal resize cache hash mismatch",
    )
    require(
        _sha256(literal_scores_path) == summary["preprocessing"]["literal_scores_sha256"],
        "literal score cache hash mismatch",
    )
    literal_resized_memmap = np.load(literal_resized_path, mmap_mode="r", allow_pickle=False)

    dataset_root = Path(config["dataset"]["root"]).resolve()
    image_paths, ordered_digest = _ordered_images(
        dataset_root, int(config["dataset"]["ordered_samples"])
    )
    require(
        ordered_digest == summary["dataset"]["ordered_image_names_sha256"],
        "ordered validation image digest mismatch",
    )
    require(summary["dataset"]["labels_used"] is False, "labels-used flag changed")
    require(
        _sha256(Path(config["dataset"]["archive_path"]))
        == config["dataset"]["archive_sha256"],
        "dataset archive hash mismatch",
    )

    with np.load(per_seed_path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    seeds = arrays["seeds"].astype(np.int64)
    epsilons_over_255 = arrays["epsilon_over_255"].astype(np.float64)
    require(seeds.tolist() == config["initialization"]["seeds"], "seed vector changed")
    require(
        np.array_equal(epsilons_over_255, np.asarray(config["epsilon_over_255"])),
        "epsilon vector changed",
    )
    expected_shape = (len(seeds), len(image_paths), len(epsilons_over_255))
    recomputed_aggregate: dict[str, Any] = {}
    for domain in config["domains"]:
        clean = arrays[f"{domain}__clean_experts"]
        competitors = arrays[f"{domain}__competitors"]
        stable = arrays[f"{domain}__stable"]
        reachable = arrays[f"{domain}__reachable"]
        undecided = arrays[f"{domain}__undecided"]
        require(clean.shape == expected_shape[:2], f"{domain} clean shape changed")
        for name, value in (
            ("competitors", competitors),
            ("stable", stable),
            ("reachable", reachable),
            ("undecided", undecided),
        ):
            require(value.shape == expected_shape, f"{domain} {name} shape changed")
        require(not bool(np.any(stable & reachable)), f"{domain} partition overlaps")
        require(
            bool(np.all(stable | reachable | undecided)),
            f"{domain} partition is incomplete",
        )
        require(
            bool(np.all((competitors >= 0) & (competitors < 4))),
            f"{domain} competitor index escaped",
        )
        recomputed_aggregate[domain] = {}
        for epsilon_slot, epsilon in enumerate(epsilons_over_255):
            stable_fractions = np.mean(stable[:, :, epsilon_slot], axis=1)
            reachable_fractions = np.mean(reachable[:, :, epsilon_slot], axis=1)
            expected = summary["aggregate"][domain][str(float(epsilon))]
            require(
                math.isclose(
                    float(np.mean(stable_fractions)),
                    expected["stable_fraction"]["mean"],
                    abs_tol=1e-15,
                ),
                f"{domain} epsilon {epsilon} stable aggregate mismatch",
            )
            require(
                math.isclose(
                    float(np.mean(reachable_fractions)),
                    expected["reachable_fraction"]["mean"],
                    abs_tol=1e-15,
                ),
                f"{domain} epsilon {epsilon} reachable aggregate mismatch",
            )
            recomputed_aggregate[domain][str(float(epsilon))] = {
                "stable_fraction_mean": float(np.mean(stable_fractions)),
                "stable_fraction_minimum": float(np.min(stable_fractions)),
                "stable_fraction_maximum": float(np.max(stable_fractions)),
                "reachable_fraction_mean": float(np.mean(reachable_fractions)),
                "undecided_sample_seed_pairs": int(
                    np.count_nonzero(undecided[:, :, epsilon_slot])
                ),
            }

        for slot, seed in enumerate(seeds):
            row = summary["rows"][slot]["domains"][domain]
            counts = np.bincount(clean[slot], minlength=4)
            require(row["route_load_counts"] == counts.tolist(), f"{domain} seed {seed} load mismatch")
            for epsilon_slot, epsilon in enumerate(epsilons_over_255):
                epsilon_row = row["epsilon_census"][str(float(epsilon))]
                require(
                    epsilon_row["stable_count"]
                    == int(np.count_nonzero(stable[slot, :, epsilon_slot])),
                    f"{domain} seed {seed} stable count mismatch",
                )
                pair_counts = Counter(
                    tuple(sorted((int(route), int(other))))
                    for route, other, is_reachable in zip(
                        clean[slot], competitors[slot, :, epsilon_slot], reachable[slot, :, epsilon_slot]
                    )
                    if is_reachable
                )
                expected_pairs = {
                    f"{left}-{right}": int(value)
                    for (left, right), value in sorted(pair_counts.items())
                }
                require(
                    epsilon_row["unordered_reachable_competitor_pairs"] == expected_pairs,
                    f"{domain} seed {seed} route-pair counts changed",
                )

    with np.load(worker_arrays, allow_pickle=False) as artifact:
        weights = artifact["weights"].astype(np.float64)
        biases = artifact["biases"].astype(np.float64)
    scale, shift = _normalization_vectors(config)
    epsilons = epsilons_over_255 / 255.0
    sample_slots = [0, len(image_paths) // 2, len(image_paths) - 1]
    seed_slots = [0, len(seeds) // 2, len(seeds) - 1]
    differential_checks = 0
    for seed_slot, sample_slot in zip(seed_slots, sample_slots):
        pixel_weight, pixel_bias = fold_affine_input_map(
            weights[seed_slot], biases[seed_slot], scale, shift
        )
        raw_weight, raw_bias = fold_bilinear_resize_input_map(
            pixel_weight,
            pixel_bias,
            channels=3,
            input_size=(64, 64),
            output_size=(224, 224),
            align_corners=False,
            antialias=True,
        )
        raw = _load_unit_image(image_paths[sample_slot])
        resized = functional.interpolate(
            torch.from_numpy(raw)[None],
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].numpy()
        literal_resized = (
            np.asarray(literal_resized_memmap[sample_slot], dtype=np.float64) / 255.0
        )
        for domain, point, weight, bias in (
            (
                "official_post_resize_224",
                literal_resized.reshape(-1),
                pixel_weight,
                pixel_bias,
            ),
            ("official_composed_raw_64", raw.reshape(-1), raw_weight, raw_bias),
        ):
            clean, stable, reachable, undecided = _direct_fixed_partition(
                weight,
                bias,
                point,
                epsilons,
                float(config["oracle"]["outward_absolute"]),
                float(config["oracle"]["outward_relative"]),
            )
            require(
                clean == int(arrays[f"{domain}__clean_experts"][seed_slot, sample_slot]),
                f"{domain} differential clean route mismatch",
            )
            for name, expected in (
                ("stable", stable),
                ("reachable", reachable),
                ("undecided", undecided),
            ):
                require(
                    np.array_equal(
                        expected,
                        arrays[f"{domain}__{name}"][seed_slot, sample_slot],
                    ),
                    f"{domain} differential {name} mismatch",
                )
            differential_checks += 1

    require(
        int(np.sum(arrays["fold_route_mismatches"])) == 0,
        "real-arithmetic resize fold changed a clean route",
    )
    require(
        int(summary["preprocessing_audit"]["total_fold_route_mismatches"]) == 0,
        "summary reports real-arithmetic fold route mismatch",
    )
    result = {
        "schema_version": 1,
        "status": "COMPLETED_AUDITED" if not issues else "AUDIT_FAILED",
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "scope": config["reporting"]["claim_scope"],
        "raw_result": {
            "directory": str(raw_dir),
            "summary_sha256": _sha256(summary_path),
            "per_seed_sha256": _sha256(per_seed_path),
            "worker_manifest_sha256": _sha256(worker_manifest),
            "worker_arrays_sha256": _sha256(worker_arrays),
            "preprocessing_manifest_sha256": _sha256(preprocessing_manifest),
            "literal_resized_sha256": _sha256(literal_resized_path),
            "literal_scores_sha256": _sha256(literal_scores_path),
        },
        "dataset": {
            "name": config["dataset"]["name"],
            "split": config["dataset"]["split"],
            "samples_per_seed": len(image_paths),
            "labels_used": False,
            "archive_sha256": config["dataset"]["archive_sha256"],
            "ordered_image_names_sha256": ordered_digest,
        },
        "initialization": {
            "seeds": seeds.tolist(),
            "official_model": "MOE_ViT",
            "router_shape": [4, 150528],
            "seed0_router_sha256": config["initialization"]["seed0_expected_router_sha256"],
        },
        "aggregate": recomputed_aggregate,
        "preprocessing_audit": summary["preprocessing_audit"],
        "audit": {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues,
            "issue_count": len(issues),
            "differential_seed_sample_domain_checks": differential_checks,
            "checks": [
                "recomputed all tracked and raw artifact hashes",
                "recomputed all K=20 by 10000 by five-radius partitions",
                "recomputed all expert loads and reachable competitor-pair counts",
                "recomputed six deterministic seed-sample-domain cases from the scalar support formula",
                "confirmed real-arithmetic resize folding preserves clean routes",
                "confirmed official clone identity and cleanliness",
            ],
        },
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.raw_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["audit"]["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
