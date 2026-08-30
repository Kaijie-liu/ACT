"""Evaluate the frozen B1 endpoint on the ordered CIFAR-10 test set."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.baseline_icml2025_b1_smoke import (
    CIFAR_MEAN,
    CIFAR_STD,
    MOE_ROOT,
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _inside,
    _repo_value,
    _sha256,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_model(checkpoint: Path, device: torch.device):
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise RuntimeError("official repository commit changed")
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official repository is not clean")
    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        from models.moe import MOE_Resnet18  # type: ignore
    finally:
        sys.path.pop(0)
    model = MOE_Resnet18(num_experts=4, num_classes=10, size=32)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("net")
    if not isinstance(state, dict):
        raise RuntimeError("endpoint checkpoint lacks net state")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload


def _normalized_batch(raw: np.ndarray, device: torch.device) -> torch.Tensor:
    import torchvision.transforms as transforms

    pixels = torch.from_numpy(raw).permute(0, 3, 1, 2).to(
        device=device, dtype=torch.float16
    )
    return transforms.Normalize(CIFAR_MEAN, CIFAR_STD)(pixels).contiguous()


def _pixel_linf(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(CIFAR_MEAN, device=clean.device)[None, :, None, None]
    std = torch.as_tensor(CIFAR_STD, device=clean.device)[None, :, None, None]
    clean_pixels = clean.float() * std + mean
    adversarial_pixels = adversarial.float() * std + mean
    return (adversarial_pixels - clean_pixels).abs().flatten(1).max(dim=1).values / 255.0


def _pixel_range(normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(CIFAR_MEAN, device=normalized.device)[None, :, None, None]
    std = torch.as_tensor(CIFAR_STD, device=normalized.device)[None, :, None, None]
    pixels = normalized.float() * std + mean
    return pixels.flatten(1).min(dim=1).values, pixels.flatten(1).max(dim=1).values


def run(
    protocol_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, Any]:
    protocol_path = _inside(protocol_path)
    checkpoint = _inside(checkpoint)
    output_dir = _inside(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"endpoint refuses to overwrite {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    endpoint = protocol["endpoint"]
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered B1 endpoint requires CUDA")
    seed = int(endpoint["attack_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, checkpoint_payload = _load_model(checkpoint, device)
    if int(checkpoint_payload.get("epoch", -1)) + 1 != int(protocol["final_epoch"]):
        raise RuntimeError("endpoint checkpoint is not the frozen final epoch")
    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        from attack.PGD import PGD  # type: ignore
    finally:
        sys.path.pop(0)
    mean = torch.tensor(np.asarray(CIFAR_MEAN, dtype=np.float32)[None, :, None, None])
    std = torch.tensor(np.asarray(CIFAR_STD, dtype=np.float32)[None, :, None, None])
    attack = PGD(
        eps=float(endpoint["epsilon"]),
        sigma=float(endpoint["step_size"]),
        nb_iter=int(endpoint["steps"]),
        DEVICE=device_name,
        mean=mean,
        std=std,
        random_start=bool(endpoint["random_start"]),
    )

    import torchvision.datasets as datasets

    dataset = datasets.CIFAR10(
        root=str(Path(protocol["run_root"]) / "data"),
        train=False,
        download=False,
    )
    raw = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    if len(raw) != int(endpoint["samples"]):
        raise RuntimeError("ordered endpoint dataset size changed")
    clean_predictions: list[np.ndarray] = []
    adversarial_predictions: list[np.ndarray] = []
    clean_routes: list[np.ndarray] = []
    adversarial_routes: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []
    linf_rows: list[np.ndarray] = []
    minimum_pixel_rows: list[np.ndarray] = []
    maximum_pixel_rows: list[np.ndarray] = []
    started = time.monotonic()
    for start in range(0, len(raw), int(endpoint["batch_size"])):
        stop = min(start + int(endpoint["batch_size"]), len(raw))
        inputs = _normalized_batch(raw[start:stop], device)
        targets = torch.from_numpy(labels[start:stop]).to(device)
        with torch.amp.autocast("cuda", enabled=True):
            with torch.no_grad():
                clean_output = model(inputs)
                clean_route = model.router(inputs)
            with torch.enable_grad():
                adversarial = attack.attack(model, inputs, targets)
            with torch.no_grad():
                adversarial_output = model(adversarial)
                adversarial_route = model.router(adversarial)
        clean_predictions.append(clean_output.argmax(dim=1).cpu().numpy())
        adversarial_predictions.append(adversarial_output.argmax(dim=1).cpu().numpy())
        clean_routes.append(clean_route.cpu().numpy())
        adversarial_routes.append(adversarial_route.cpu().numpy())
        endpoints.append(adversarial.detach().float().cpu().numpy())
        linf_rows.append(_pixel_linf(inputs, adversarial).detach().cpu().numpy())
        minimum_pixels, maximum_pixels = _pixel_range(adversarial)
        minimum_pixel_rows.append(minimum_pixels.detach().cpu().numpy())
        maximum_pixel_rows.append(maximum_pixels.detach().cpu().numpy())
    clean_prediction = np.concatenate(clean_predictions).astype(np.int64)
    adversarial_prediction = np.concatenate(adversarial_predictions).astype(np.int64)
    clean_route = np.concatenate(clean_routes).astype(np.int64)
    adversarial_route = np.concatenate(adversarial_routes).astype(np.int64)
    adversarial_endpoint = np.concatenate(endpoints).astype(np.float32)
    linf = np.concatenate(linf_rows).astype(np.float64)
    minimum_pixels = np.concatenate(minimum_pixel_rows).astype(np.float64)
    maximum_pixels = np.concatenate(maximum_pixel_rows).astype(np.float64)
    if np.any(linf > float(endpoint["epsilon"]) + 2e-5):
        raise RuntimeError("stored PGD50 endpoint exceeds the registered pixel box")
    if np.any(minimum_pixels < -2e-3) or np.any(maximum_pixels > 255.0 + 2e-3):
        raise RuntimeError("stored PGD50 endpoint exceeds the registered pixel domain")

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "ordered_endpoint_replay.npz"
    with artifact_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            labels=labels,
            clean_predictions=clean_prediction,
            adversarial_predictions=adversarial_prediction,
            clean_routes=clean_route,
            adversarial_routes=adversarial_route,
            adversarial_endpoints=adversarial_endpoint,
            pixel_linf=linf,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
        )
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": "B1_EPOCH130_ORDERED_FULL_TEST_SA_AND_OFFICIAL_PGD50",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "reported_accuracy_percent": checkpoint_payload.get("acc"),
            "public_epoch": int(checkpoint_payload.get("epoch", -1)) + 1,
        },
        "dataset": {"samples": int(len(labels)), "ordered": True},
        "attack": endpoint,
        "standard_accuracy_percent": float(100.0 * np.mean(clean_prediction == labels)),
        "pgd50_accuracy_percent": float(100.0 * np.mean(adversarial_prediction == labels)),
        "clean_route_counts": np.bincount(clean_route, minlength=4).astype(int).tolist(),
        "adversarial_route_counts": np.bincount(adversarial_route, minlength=4).astype(int).tolist(),
        "route_flip_count": int(np.sum(clean_route != adversarial_route)),
        "pixel_linf": {
            "maximum": float(linf.max()),
            "median": float(np.median(linf)),
        },
        "pixel_domain": {
            "minimum": float(minimum_pixels.min()),
            "maximum": float(maximum_pixels.max()),
        },
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "runtime": {
            "seconds": time.monotonic() - started,
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.checkpoint, args.output_dir, device_name=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
