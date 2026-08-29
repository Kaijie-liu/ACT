"""Execute exact affine-router telemetry on an immutable RT-ER checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from act.back_end.moe import (
    affine_top1_route_boundary_batch,
    audit_eta_overcheck_band,
    fold_affine_input_map,
)
from act.back_end.solver.solver_hz import HZ_NUMERICAL_POLICY
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json
from act.util.path_config import get_torchvision_data_root


OFFICIAL_REPO = Path("/data1/Kane/MOE/baselines/Robust-MoE-Dual-Model")
OFFICIAL_COMMIT = "30ef94d77b5451595b82e739aa8938e1f4c4521f"
DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/icml2025_route_telemetry.json"
CIFAR_MEAN_255 = np.asarray([125.307, 122.961, 113.8575], dtype=np.float64)
CIFAR_STD_255 = np.asarray([51.5865, 50.847, 51.255], dtype=np.float64)


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def _raw_dataset_hash(images: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(images, dtype=np.uint8).tobytes(order="C"))
    digest.update(np.asarray(labels, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _normalization_vectors(features: int) -> tuple[np.ndarray, np.ndarray]:
    if features != 3 * 32 * 32:
        raise ValueError("official CIFAR router must have 3072 input features")
    scale = np.broadcast_to((255.0 / CIFAR_STD_255)[:, None, None], (3, 32, 32))
    shift = np.broadcast_to((-CIFAR_MEAN_255 / CIFAR_STD_255)[:, None, None], (3, 32, 32))
    return scale.reshape(-1), shift.reshape(-1)


def fold_official_router(
    weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scale, shift = _normalization_vectors(np.asarray(weight).shape[1])
    return fold_affine_input_map(weight, bias, scale, shift)


def _load_official_model(checkpoint_path: Path, device: torch.device):
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise RuntimeError("official RT-ER repository commit changed")
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official RT-ER repository is not clean")
    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        from models.resnet import ResNet18  # type: ignore
    finally:
        sys.path.pop(0)

    class Router(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(3 * 32 * 32, 4)

    class ResnetExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = ResNet18()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.net(value)

    class OfficialStateLayout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ResnetExpert() for _ in range(4)])
            self.router = Router()

    model = OfficialStateLayout()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("net")
    if not isinstance(state, dict):
        raise ValueError("official checkpoint lacks a net state dict")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def _grouped_official_forward(model, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorize the official selected-expert loop without changing dispatch."""

    scores = model.router.gate(normalized.flatten(1))
    routes = scores.argmax(dim=1)
    output = torch.empty(
        normalized.shape[0], 10, dtype=normalized.dtype, device=normalized.device
    )
    for expert_index, expert in enumerate(model.experts):
        mask = routes == expert_index
        if bool(mask.any()):
            output[mask] = expert(normalized[mask])
    return output, scores


def _classification_outputs(
    model,
    images_nchw: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    mean = torch.as_tensor(CIFAR_MEAN_255, dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.as_tensor(CIFAR_STD_255, dtype=torch.float32, device=device)[None, :, None, None]
    with torch.no_grad():
        for start in range(0, images_nchw.shape[0], batch_size):
            pixels = torch.as_tensor(
                images_nchw[start : start + batch_size], dtype=torch.float32, device=device
            )
            normalized = (pixels * 255.0 - mean) / std
            output, router_scores = _grouped_official_forward(model, normalized)
            predictions.append(output.argmax(dim=1).cpu().numpy())
            scores.append(router_scores.cpu().double().numpy())
    return np.concatenate(predictions), np.concatenate(scores)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if not finite.size:
        return {key: None for key in ("p10", "p25", "p50", "p75", "p90", "p95")}
    result = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return dict(zip(("p10", "p25", "p50", "p75", "p90", "p95"), map(float, result)))


def summarize_boundaries(
    clean_experts: np.ndarray,
    competitors: np.ndarray,
    radii: np.ndarray,
    radius_lowers: np.ndarray,
    radius_uppers: np.ndarray,
    *,
    num_experts: int,
    epsilon_over_255: Sequence[float],
) -> dict[str, Any]:
    counts = np.bincount(clean_experts, minlength=num_experts).astype(np.int64)
    probabilities = counts / max(1, counts.sum())
    nonzero = probabilities[probabilities > 0.0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    directed = Counter(
        (int(clean), int(competitor))
        for clean, competitor in zip(clean_experts, competitors)
        if competitor >= 0
    )
    unordered = Counter(
        tuple(sorted((int(clean), int(competitor))))
        for clean, competitor in zip(clean_experts, competitors)
        if competitor >= 0
    )
    radius_counts = {}
    for numerator in epsilon_over_255:
        epsilon = float(numerator) / 255.0
        stable = epsilon < radius_lowers
        reachable = radius_uppers <= epsilon
        undecided = ~(stable | reachable)
        radius_counts[str(numerator)] = {
            "epsilon": epsilon,
            "proven_stable": int(stable.sum()),
            "proven_reachable": int(reachable.sum()),
            "numerically_undecided": int(undecided.sum()),
        }
    return {
        "samples": int(clean_experts.size),
        "finite_route_boundaries": int(np.isfinite(radii).sum()),
        "radius_quantiles": _quantiles(radii),
        "route_load_counts": counts.tolist(),
        "route_load_probabilities": probabilities.tolist(),
        "route_load_entropy": entropy,
        "effective_expert_count": float(math.exp(entropy)),
        "directed_boundary_competitor_counts": {
            f"{left}->{right}": count for (left, right), count in sorted(directed.items())
        },
        "unordered_boundary_pair_counts": {
            f"{left}-{right}": count for (left, right), count in sorted(unordered.items())
        },
        "epsilon_counts": radius_counts,
    }


def _validate_probe_witnesses(
    pixel_points: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    result,
    *,
    tolerance: float,
) -> dict[str, Any]:
    deltas = result.witness_deltas
    if deltas is None:
        raise RuntimeError("probe oracle did not return witnesses")
    checked = failures = 0
    maximum_linf_excess = maximum_score_deficit = 0.0
    for index, competitor in enumerate(result.boundary_competitors):
        if competitor < 0 or not np.isfinite(result.radius_uppers[index]):
            continue
        checked += 1
        candidate = pixel_points[index] + deltas[index]
        linf = float(np.max(np.abs(deltas[index])))
        maximum_linf_excess = max(
            maximum_linf_excess, linf - float(result.radius_uppers[index])
        )
        clean = int(result.clean_experts[index])
        scores = weight @ candidate + bias
        deficit = float(scores[clean] - scores[int(competitor)])
        maximum_score_deficit = max(maximum_score_deficit, deficit)
        valid = (
            np.all(candidate >= -tolerance)
            and np.all(candidate <= 1.0 + tolerance)
            and linf <= float(result.radius_uppers[index]) + tolerance
            and deficit <= tolerance
        )
        failures += int(not valid)
    return {
        "checked": checked,
        "failures": failures,
        "maximum_linf_excess": maximum_linf_excess,
        "maximum_competitor_score_deficit": maximum_score_deficit,
        "tolerance": tolerance,
    }


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(path)


def run(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    seed: int,
    epoch: int,
    device_name: str,
    metrics_path: Path | None,
) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    checkpoint_path = _inside(checkpoint_path, WRITE_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"telemetry refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("unexpected telemetry protocol status")
    if int(seed) not in config["training"]["seeds"]:
        raise ValueError("seed is outside the frozen telemetry schedule")
    if int(epoch) not in config["training"]["checkpoint_epochs"]:
        raise ValueError("epoch is outside the frozen telemetry schedule")
    if not Path(get_torchvision_data_root()).resolve().is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision root escapes /data1/Kane/MOE")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA telemetry requested but unavailable")
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "config.json",
        {
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "seed": int(seed),
            "epoch": int(epoch),
            "device": device_name,
            "official_repository": str(OFFICIAL_REPO),
            "official_commit": OFFICIAL_COMMIT,
            "config": config,
        },
    )
    started = time.monotonic()
    model, checkpoint = _load_official_model(checkpoint_path, device)
    if int(checkpoint.get("epoch", -1)) + 1 != int(epoch):
        raise RuntimeError("checkpoint epoch does not match epoch-qualified artifact")

    import torchvision.datasets as datasets

    dataset = datasets.CIFAR10(
        root=str(Path(get_torchvision_data_root()) / "CIFAR10" / "raw"),
        train=False,
        download=False,
    )
    raw_images = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    images = raw_images.transpose(0, 3, 1, 2).astype(np.float64) / 255.0
    points = images.reshape(images.shape[0], -1)
    predictions, normalized_scores = _classification_outputs(
        model, images, device=device, batch_size=int(config["execution"]["batch_size"])
    )
    router = model.router.gate
    raw_weight = router.weight.detach().cpu().double().numpy()
    raw_bias = router.bias.detach().cpu().double().numpy()
    weight, bias = fold_official_router(raw_weight, raw_bias)
    folded_scores = points @ weight.T + bias
    maximum_score_difference = float(np.max(np.abs(folded_scores - normalized_scores)))
    if maximum_score_difference > float(config["execution"]["score_equivalence_atol"]):
        raise RuntimeError("folded pixel router differs from normalized official router")

    oracle_started = time.monotonic()
    oracle_config = config["oracle"]
    boundaries = affine_top1_route_boundary_batch(
        weight,
        bias,
        points,
        input_lower=float(oracle_config["input_lower"]),
        input_upper=float(oracle_config["input_upper"]),
        outward_absolute=float(oracle_config["outward_absolute"]),
        outward_relative=float(oracle_config["outward_relative"]),
        compute_device=device_name,
        capacity_grid_steps=int(oracle_config["capacity_grid_steps"]),
    )
    oracle_seconds = time.monotonic() - oracle_started
    official_routes = normalized_scores.argmax(axis=1)
    if not np.array_equal(boundaries.clean_experts, official_routes):
        raise RuntimeError("oracle clean routes differ from official router")

    reference_count = int(oracle_config["reference_crosscheck_samples"])
    reference_indices = np.arange(reference_count, dtype=np.int64)
    reference = affine_top1_route_boundary_batch(
        weight,
        bias,
        points[reference_indices],
        input_lower=0.0,
        input_upper=1.0,
    )
    fast_reference_radii = boundaries.radii[reference_indices]
    same_finiteness = np.array_equal(
        np.isfinite(reference.radii), np.isfinite(fast_reference_radii)
    )
    finite_reference = np.isfinite(reference.radii) & np.isfinite(fast_reference_radii)
    reference_max_error = (
        float(np.max(np.abs(reference.radii[finite_reference] - fast_reference_radii[finite_reference])))
        if bool(finite_reference.any())
        else 0.0
    )
    reference_ok = (
        same_finiteness
        and reference_max_error <= float(oracle_config["reference_radius_atol"])
        and np.array_equal(reference.clean_experts, boundaries.clean_experts[reference_indices])
        and np.array_equal(
            reference.boundary_competitors,
            boundaries.boundary_competitors[reference_indices],
        )
    )
    if not reference_ok:
        raise RuntimeError("fast route oracle failed frozen reference crosscheck")

    probe_count = int(config["execution"]["probe_samples"])
    probe_indices = np.arange(probe_count, dtype=np.int64)
    probes = affine_top1_route_boundary_batch(
        weight,
        bias,
        points[probe_indices],
        input_lower=0.0,
        input_upper=1.0,
        outward_absolute=float(oracle_config["outward_absolute"]),
        outward_relative=float(oracle_config["outward_relative"]),
        include_witnesses=True,
        compute_device=device_name,
        capacity_grid_steps=int(oracle_config["capacity_grid_steps"]),
    )
    witness_audit = _validate_probe_witnesses(
        points[probe_indices],
        weight,
        bias,
        probes,
        tolerance=float(config["execution"]["witness_tolerance"]),
    )
    if witness_audit["failures"]:
        raise RuntimeError("route-boundary probe witness replay failed")

    metrics = None
    metrics_hash = None
    if metrics_path is not None:
        metrics_path = _inside(metrics_path, WRITE_ROOT)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_hash = _sha256(metrics_path)
    clean_correct = predictions == labels
    eta_audit = audit_eta_overcheck_band(
        torch.from_numpy(normalized_scores),
        eta=float(config["execution"]["eta"]),
        boundary_tolerance=float(config["execution"]["eta_boundary_tolerance"]),
    ).as_dict()
    artifact_path = output_dir / "per_input.npz"
    artifact_hash = _save_npz(
        artifact_path,
        dataset_indices=np.arange(points.shape[0], dtype=np.int64),
        labels=labels,
        predictions=predictions.astype(np.int64),
        clean_correct=clean_correct.astype(np.bool_),
        clean_experts=boundaries.clean_experts,
        boundary_competitors=boundaries.boundary_competitors,
        radii=boundaries.radii,
        radius_lowers=boundaries.radius_lowers,
        radius_uppers=boundaries.radius_uppers,
        probe_indices=probe_indices,
        probe_witness_deltas=probes.witness_deltas,
    )
    summary = {
        "schema_version": 1,
        "label": config["label"],
        "seed": int(seed),
        "epoch": int(epoch),
        "official_source": config["official_source"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "reported_accuracy": checkpoint.get("acc"),
        },
        "dataset": {
            "name": "CIFAR10",
            "split": "test",
            "samples": int(points.shape[0]),
            "raw_images_and_labels_sha256": _raw_dataset_hash(raw_images, labels),
        },
        "classification": {
            "computed_clean_accuracy": float(clean_correct.mean()),
            "clean_correct": int(clean_correct.sum()),
            "checkpoint_reported_accuracy_percent": checkpoint.get("acc"),
            "training_metrics": metrics,
            "training_metrics_sha256": metrics_hash,
        },
        "folded_router": {
            "maximum_score_difference": maximum_score_difference,
            "score_equivalence_atol": float(config["execution"]["score_equivalence_atol"]),
            "normalization_mean_255": CIFAR_MEAN_255.tolist(),
            "normalization_std_255": CIFAR_STD_255.tolist(),
        },
        "route_geometry": summarize_boundaries(
            boundaries.clean_experts,
            boundaries.boundary_competitors,
            boundaries.radii,
            boundaries.radius_lowers,
            boundaries.radius_uppers,
            num_experts=raw_weight.shape[0],
            epsilon_over_255=config["epsilon_over_255"],
        ),
        "eta_overcheck_band": eta_audit,
        "reference_crosscheck": {
            "samples": reference_count,
            "maximum_radius_error": reference_max_error,
            "passed": reference_ok,
        },
        "probe_witness_replay": witness_audit,
        "artifact": {"path": "per_input.npz", "sha256": artifact_hash},
        "runtime": {
            "oracle_seconds": oracle_seconds,
            "total_seconds": time.monotonic() - started,
            "device": str(device),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "interpretation": (
            "Exact affine hard-route geometry for an official-code, paper-config "
            "reproduction checkpoint; not an expert-output robustness certificate."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    result = run(
        args.config,
        args.checkpoint,
        args.output_dir,
        seed=args.seed,
        epoch=args.epoch,
        device_name=args.device,
        metrics_path=args.metrics,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
