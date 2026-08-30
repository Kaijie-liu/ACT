"""Measure the local initialization-radius scaling across official RT-ER routers.

This is deliberately separate from the exact, box-capped census.  The quantity
measured here is the unbounded local affine radius in the router's normalized
input coordinates,

    min_{j != i} (r_i(x) - r_j(x)) / ||w_i - w_j||_1.

It is the quantity addressed by the standard-initialization ``1/sqrt(d)``
scaling argument.  It is not an output certificate or an exact clipped-box
route boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_route_telemetry import CIFAR_MEAN_255, CIFAR_STD_255
from act.util.path_config import get_torchvision_data_root


DEFAULT_CIFAR_AUDIT = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/icml2025_rt_er/router_init_census_k20_20260830.json"
)
DEFAULT_TINY_AUDIT = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/icml2025_rt_er/"
    "tinyimagenet_router_census_k20_20260830_r2.json"
)
TINY_LITERAL_SCORES = Path(
    "/data1/Kane/MOE/cache/icml2025_tinyimagenet/preprocessing_r3/"
    "literal_preprocessing_scores_float64.npy"
)
TINY_LITERAL_CENTERS = Path(
    "/data1/Kane/MOE/cache/icml2025_tinyimagenet/preprocessing_r3/"
    "literal_resized_255_float16.npy"
)


def local_affine_top1_radii(
    scores: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact unbounded local L-inf radii for one affine top-1 router."""

    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 2 or weights.ndim != 2:
        raise ValueError("scores and weights must be matrices")
    if scores.shape[1] != weights.shape[0] or weights.shape[0] < 2:
        raise ValueError("router score/weight dimensions differ")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)):
        raise ValueError("router scores and weights must be finite")
    experts = weights.shape[0]
    clean = np.argmax(scores, axis=1).astype(np.int64)
    row = np.arange(scores.shape[0])
    gaps = scores[row, clean, None] - scores
    denominators = np.sum(
        np.abs(weights[:, None, :] - weights[None, :, :]), axis=2
    )
    candidates = np.full_like(gaps, np.inf, dtype=np.float64)
    for other in range(experts):
        active = clean != other
        selected_denominator = denominators[clean[active], other]
        if np.any(selected_denominator <= 0.0):
            raise ValueError("distinct router rows have zero L1 separation")
        candidates[active, other] = (
            gaps[active, other] / selected_denominator
        )
    competitor = np.argmin(candidates, axis=1).astype(np.int64)
    radii = candidates[row, competitor]
    if np.any(radii < -1e-15) or not np.all(np.isfinite(radii)):
        raise RuntimeError("invalid local affine route radius")
    return np.maximum(radii, 0.0), clean, competitor


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_artifacts(audit_path: Path) -> tuple[dict[str, Any], Path, Path]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_issues = audit["audit"].get("issue_count")
    if audit_issues is None:
        raw_issues = audit["audit"].get("issues", [])
        audit_issues = raw_issues if isinstance(raw_issues, int) else len(raw_issues)
    if audit.get("status") != "COMPLETED_AUDITED" or int(audit_issues) != 0:
        raise RuntimeError(f"source census is not independently clean: {audit_path}")
    raw = _inside(Path(audit["raw_result"]["directory"]), WRITE_ROOT)
    routers = raw / "router_initializations/routers.npz"
    per_seed = raw / "per_seed.npz"
    router_hash_key = (
        "router_initializations_sha256"
        if "router_initializations_sha256" in audit["raw_result"]
        else "worker_arrays_sha256"
    )
    if _sha256(routers) != audit["raw_result"][router_hash_key]:
        raise RuntimeError(f"router initialization artifact changed: {routers}")
    if _sha256(per_seed) != audit["raw_result"]["per_seed_sha256"]:
        raise RuntimeError(f"per-seed census artifact changed: {per_seed}")
    return audit, routers, per_seed


def _cifar_scores(weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, float]:
    import torchvision.datasets as datasets

    root = Path(get_torchvision_data_root()).resolve()
    if not root.is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision root escapes /data1/Kane/MOE")
    dataset = datasets.CIFAR10(
        root=str(root / "CIFAR10/raw"), train=False, download=False
    )
    pixels = np.asarray(dataset.data, dtype=np.float64).transpose(0, 3, 1, 2)
    normalized = (
        pixels - CIFAR_MEAN_255[None, :, None, None]
    ) / CIFAR_STD_255[None, :, None, None]
    points = normalized.reshape(len(dataset), -1)
    flat_weights = weights.reshape(-1, weights.shape[-1])
    flat_biases = biases.reshape(-1)
    flat_scores = points @ flat_weights.T + flat_biases[None, :]
    scores = flat_scores.reshape(len(dataset), *weights.shape[:2]).transpose(1, 0, 2)
    return scores, float(np.mean(points * points))


def _tiny_real_affine_second_moment(
    centers_path: Path,
    mean_255: np.ndarray,
    std_255: np.ndarray,
    *,
    batch_size: int = 64,
) -> float:
    centers = np.load(centers_path, mmap_mode="r", allow_pickle=False)
    if centers.shape != (10000, 3, 224, 224) or centers.dtype != np.float16:
        raise RuntimeError("released TinyImageNet centre cache layout changed")
    total = 0.0
    count = 0
    for start in range(0, len(centers), int(batch_size)):
        values = np.asarray(centers[start : start + batch_size], dtype=np.float64)
        normalized = (
            values - mean_255[None, :, None, None]
        ) / std_255[None, :, None, None]
        total += float(np.sum(normalized * normalized, dtype=np.float64))
        count += normalized.size
    return total / count


def run(cifar_audit_path: Path, tiny_audit_path: Path, output_dir: Path) -> dict[str, Any]:
    cifar_audit_path = _inside(cifar_audit_path, PROJECT_ROOT)
    tiny_audit_path = _inside(tiny_audit_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"dimension-law output exists: {output_dir}")
    cifar, cifar_routers_path, cifar_per_seed_path = _source_artifacts(
        cifar_audit_path
    )
    tiny, tiny_routers_path, tiny_per_seed_path = _source_artifacts(tiny_audit_path)
    literal_scores_path = _inside(TINY_LITERAL_SCORES, WRITE_ROOT)
    literal_centers_path = _inside(TINY_LITERAL_CENTERS, WRITE_ROOT)
    if _sha256(literal_scores_path) != tiny["raw_result"]["literal_scores_sha256"]:
        raise RuntimeError("released-runtime TinyImageNet score cache changed")
    if _sha256(literal_centers_path) != tiny["raw_result"]["literal_resized_sha256"]:
        raise RuntimeError("released-runtime TinyImageNet centre cache changed")
    tiny_config_path = PROJECT_ROOT / (
        "act/pipeline/moe/configs/icml2025_tinyimagenet_router_census_r3.json"
    )
    tiny_config = json.loads(tiny_config_path.read_text(encoding="utf-8"))
    tiny_mean = np.asarray(
        tiny_config["router"]["normalization_mean_255"], dtype=np.float64
    )
    tiny_std = np.asarray(
        tiny_config["router"]["normalization_std_255"], dtype=np.float64
    )

    with np.load(cifar_routers_path, allow_pickle=False) as arrays:
        cifar_seeds = arrays["seeds"].copy()
        cifar_weights = arrays["weights"].copy()
        cifar_biases = arrays["biases"].copy()
    with np.load(tiny_routers_path, allow_pickle=False) as arrays:
        tiny_seeds = arrays["seeds"].copy()
        tiny_weights = arrays["weights"].copy()
    if not np.array_equal(cifar_seeds, tiny_seeds):
        raise RuntimeError("cross-dataset seed grids differ")
    cifar_scores, cifar_second_moment = _cifar_scores(cifar_weights, cifar_biases)
    tiny_second_moment = _tiny_real_affine_second_moment(
        literal_centers_path, tiny_mean, tiny_std
    )
    tiny_scores = np.load(literal_scores_path, mmap_mode="r", allow_pickle=False)
    if tiny_scores.shape != (len(tiny_seeds), 10000, tiny_weights.shape[1]):
        raise RuntimeError("TinyImageNet score-cache layout changed")

    cifar_radii = np.empty((len(cifar_seeds), 10000), dtype=np.float64)
    tiny_radii = np.empty_like(cifar_radii)
    cifar_routes = np.empty_like(cifar_radii, dtype=np.int8)
    tiny_routes = np.empty_like(cifar_routes)
    rows: list[dict[str, Any]] = []
    for slot, seed in enumerate(cifar_seeds):
        c_radius, c_route, _ = local_affine_top1_radii(
            cifar_scores[slot], cifar_weights[slot]
        )
        t_radius, t_route, _ = local_affine_top1_radii(
            np.asarray(tiny_scores[slot]), tiny_weights[slot]
        )
        cifar_radii[slot] = c_radius
        tiny_radii[slot] = t_radius
        cifar_routes[slot] = c_route
        tiny_routes[slot] = t_route
        c_median = float(np.median(c_radius))
        t_median = float(np.median(t_radius))
        rows.append(
            {
                "seed": int(seed),
                "cifar_local_radius_median": c_median,
                "tinyimagenet_local_radius_median": t_median,
                "cifar_over_tiny_median_ratio": c_median / t_median,
            }
        )

    with np.load(cifar_per_seed_path, allow_pickle=False) as arrays:
        cifar_route_mismatches = int(
            np.count_nonzero(cifar_routes != arrays["clean_experts"])
        )
    with np.load(tiny_per_seed_path, allow_pickle=False) as arrays:
        tiny_formal_route_mismatches = int(
            np.count_nonzero(
                tiny_routes != arrays["official_post_resize_224__clean_experts"]
            )
        )
    if cifar_route_mismatches:
        raise RuntimeError("CIFAR local-radius scores disagree with exact census routes")
    if tiny_formal_route_mismatches != int(
        tiny["primary_semantics_agreement"]["literal_clean_route_mismatches"]
    ):
        raise RuntimeError("Tiny literal/formal route mismatch count changed")

    output_dir.mkdir(parents=True)
    arrays_path = output_dir / "local_radii.npz"
    with arrays_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            seeds=cifar_seeds,
            cifar_local_radii=cifar_radii,
            tinyimagenet_local_radii=tiny_radii,
            cifar_clean_experts=cifar_routes,
            tinyimagenet_literal_clean_experts=tiny_routes,
        )
        handle.flush()
        os.fsync(handle.fileno())
    predicted = math.sqrt(tiny_weights.shape[-1] / cifar_weights.shape[-1])
    moment_adjusted_prediction = predicted * math.sqrt(
        cifar_second_moment / tiny_second_moment
    )
    observed = float(np.median(cifar_radii) / np.median(tiny_radii))
    per_seed_ratios = np.asarray(
        [row["cifar_over_tiny_median_ratio"] for row in rows], dtype=np.float64
    )
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "quantity": (
            "unbounded local L-infinity radius in each affine router's normalized "
            "input coordinates; not the exact box-capped census radius"
        ),
        "dimensions": {
            "cifar10": int(cifar_weights.shape[-1]),
            "tinyimagenet_224": int(tiny_weights.shape[-1]),
        },
        "nominal_prediction": {
            "law": "radius scales as 1/sqrt(d) under standard affine initialization",
            "cifar_over_tiny": predicted,
            "input_second_moment_adjusted_cifar_over_tiny": moment_adjusted_prediction,
        },
        "observed": {
            "cifar_aggregate_median": float(np.median(cifar_radii)),
            "tinyimagenet_aggregate_median": float(np.median(tiny_radii)),
            "cifar_over_tiny_aggregate_median_ratio": observed,
            "observed_over_nominal": observed / predicted,
            "paired_seed_median_ratio": float(np.median(per_seed_ratios)),
            "paired_seed_minimum_ratio": float(np.min(per_seed_ratios)),
            "paired_seed_maximum_ratio": float(np.max(per_seed_ratios)),
            "cifar_normalized_input_second_moment": cifar_second_moment,
            "tiny_real_affine_at_literal_centres_second_moment": tiny_second_moment,
            "observed_over_second_moment_adjusted_prediction": (
                observed / moment_adjusted_prediction
            ),
        },
        "per_seed": rows,
        "semantics_checks": {
            "cifar_route_mismatches_vs_exact_census": cifar_route_mismatches,
            "tiny_literal_vs_formal_route_mismatches": tiny_formal_route_mismatches,
            "tiny_formal_stable_mismatch_intersections_at_registered_radii": tiny[
                "primary_semantics_agreement"
            ]["formal_stable_literal_route_mismatches_by_epsilon"],
        },
        "sources": {
            "cifar_audit": {"path": str(cifar_audit_path), "sha256": _sha256(cifar_audit_path)},
            "tiny_audit": {"path": str(tiny_audit_path), "sha256": _sha256(tiny_audit_path)},
            "cifar_routers": {"path": str(cifar_routers_path), "sha256": _sha256(cifar_routers_path)},
            "tiny_routers": {"path": str(tiny_routers_path), "sha256": _sha256(tiny_routers_path)},
            "tiny_literal_scores": {"path": str(literal_scores_path), "sha256": _sha256(literal_scores_path)},
            "tiny_literal_centres": {"path": str(literal_centers_path), "sha256": _sha256(literal_centers_path)},
            "tiny_config": {"path": str(tiny_config_path), "sha256": _sha256(tiny_config_path)},
        },
        "artifact": {"path": str(arrays_path), "sha256": _sha256(arrays_path)},
        "scope": (
            "Two official full-model construction families only. The raw64 fold is "
            "not a third dimension point. Agreement is an empirical scale check, "
            "not a universal law or a preregistered pass/fail endpoint."
        ),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-audit", type=Path, default=DEFAULT_CIFAR_AUDIT)
    parser.add_argument("--tiny-audit", type=Path, default=DEFAULT_TINY_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.cifar_audit, args.tiny_audit, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
