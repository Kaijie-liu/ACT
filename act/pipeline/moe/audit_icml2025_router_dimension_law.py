"""Independently audit the cross-dataset local route-radius scale result."""

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
from act.pipeline.moe.icml2025_router_dimension_law import TINY_LITERAL_SCORES
from act.util.path_config import get_torchvision_data_root


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _scalar_radius(score: np.ndarray, weight: np.ndarray) -> tuple[float, int]:
    clean = int(np.argmax(score))
    best = math.inf
    competitor = -1
    for other in range(len(score)):
        if other == clean:
            continue
        denominator = float(np.sum(np.abs(weight[clean] - weight[other])))
        radius = float(score[clean] - score[other]) / denominator
        if radius < best:
            best = radius
            competitor = other
    return max(best, 0.0), competitor


def run(raw_summary_path: Path, output_path: Path) -> dict[str, Any]:
    raw_summary_path = _inside(raw_summary_path, WRITE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"dimension-law audit output exists: {output_path}")
    raw = json.loads(raw_summary_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if raw.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw result status is not audit-ready")
    artifact_path = _inside(Path(raw["artifact"]["path"]), WRITE_ROOT)
    if _sha256(artifact_path) != raw["artifact"]["sha256"]:
        issues.append("local-radius artifact hash changed")
    for source in raw["sources"].values():
        path = _inside(Path(source["path"]), WRITE_ROOT)
        if _sha256(path) != source["sha256"]:
            issues.append(f"source hash changed: {path}")

    with np.load(artifact_path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    required = {
        "seeds",
        "cifar_local_radii",
        "tinyimagenet_local_radii",
        "cifar_clean_experts",
        "tinyimagenet_literal_clean_experts",
    }
    if set(arrays) != required:
        issues.append("local-radius artifact fields changed")
    cifar_radii = arrays["cifar_local_radii"]
    tiny_radii = arrays["tinyimagenet_local_radii"]
    if cifar_radii.shape != (20, 10000) or tiny_radii.shape != (20, 10000):
        issues.append("local-radius artifact shape changed")
    if np.any(cifar_radii < 0.0) or np.any(tiny_radii < 0.0):
        issues.append("local-radius artifact contains a negative value")

    observed = float(np.median(cifar_radii) / np.median(tiny_radii))
    if not math.isclose(
        observed,
        float(raw["observed"]["cifar_over_tiny_aggregate_median_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        issues.append("aggregate observed ratio does not recompute")
    predicted = math.sqrt(
        float(raw["dimensions"]["tinyimagenet_224"])
        / float(raw["dimensions"]["cifar10"])
    )
    if not math.isclose(
        predicted,
        float(raw["nominal_prediction"]["cifar_over_tiny"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        issues.append("nominal dimension prediction does not recompute")
    adjusted = predicted * math.sqrt(
        float(raw["observed"]["cifar_normalized_input_second_moment"])
        / float(
            raw["observed"][
                "tiny_real_affine_at_literal_centres_second_moment"
            ]
        )
    )
    if not math.isclose(
        adjusted,
        float(
            raw["nominal_prediction"][
                "input_second_moment_adjusted_cifar_over_tiny"
            ]
        ),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        issues.append("second-moment-adjusted prediction does not recompute")

    cifar_router_path = Path(raw["sources"]["cifar_routers"]["path"])
    tiny_router_path = Path(raw["sources"]["tiny_routers"]["path"])
    with np.load(cifar_router_path, allow_pickle=False) as value:
        cifar_weights = value["weights"].copy()
        cifar_biases = value["biases"].copy()
    with np.load(tiny_router_path, allow_pickle=False) as value:
        tiny_weights = value["weights"].copy()
    tiny_scores = np.load(TINY_LITERAL_SCORES, mmap_mode="r", allow_pickle=False)

    import torchvision.datasets as datasets

    torchvision_root = Path(get_torchvision_data_root()).resolve()
    dataset = datasets.CIFAR10(
        root=str(torchvision_root / "CIFAR10/raw"), train=False, download=False
    )
    pixels = np.asarray(dataset.data, dtype=np.float64).transpose(0, 3, 1, 2)
    points = (
        (pixels - CIFAR_MEAN_255[None, :, None, None])
        / CIFAR_STD_255[None, :, None, None]
    ).reshape(10000, -1)
    checked = 0
    maximum_error = 0.0
    for seed_slot in range(20):
        for sample in ((seed_slot * 499 + offset * 997) % 10000 for offset in range(4)):
            c_score = points[sample] @ cifar_weights[seed_slot].T + cifar_biases[seed_slot]
            c_radius, _ = _scalar_radius(c_score, cifar_weights[seed_slot])
            t_radius, _ = _scalar_radius(
                np.asarray(tiny_scores[seed_slot, sample]), tiny_weights[seed_slot]
            )
            maximum_error = max(
                maximum_error,
                abs(c_radius - float(cifar_radii[seed_slot, sample])),
                abs(t_radius - float(tiny_radii[seed_slot, sample])),
            )
            checked += 2
    if maximum_error > 2e-15:
        issues.append(f"scalar local-radius differential error {maximum_error:.3g}")

    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "raw_result": {"path": str(raw_summary_path), "sha256": _sha256(raw_summary_path)},
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "recomputed": {
            "nominal_cifar_over_tiny_ratio": predicted,
            "observed_cifar_over_tiny_aggregate_median_ratio": observed,
            "observed_over_nominal": observed / predicted,
            "second_moment_adjusted_cifar_over_tiny_ratio": adjusted,
            "observed_over_second_moment_adjusted": observed / adjusted,
            "scalar_differential_checks": checked,
            "scalar_differential_maximum_absolute_error": maximum_error,
        },
        "interpretation": (
            "The audit validates identities, aggregation, and scalar formula replay. "
            "It does not turn proximity to 7x into a preregistered acceptance test, "
            "a universal theorem, an exact box-capped boundary, or an output certificate."
        ),
    }
    _write_json(output_path, result)
    if issues:
        raise RuntimeError(f"dimension-law audit found {len(issues)} issue(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw_summary, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
