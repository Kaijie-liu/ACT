"""Prepare the frozen RT-ER B2 semantic-conformance artifact.

The reference side compares the released checkpoint semantics with the exact
interfaces consumed by B3: a folded affine router and raw-pixel specialized
experts.  It covers 1000 ordered clean inputs and their independently audited
PGD-50 endpoints.  A separate worker in the pinned CROWN environment checks
the same logits after auto_LiRPA graph conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from act.util.typing_compat import install_typing_override

install_typing_override()

import numpy as np
import torch

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_b3 import PixelNormalizedExpert
from act.pipeline.moe.icml2025_route_telemetry import (
    CIFAR_MEAN_255,
    CIFAR_STD_255,
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _grouped_official_forward,
    _load_official_model,
    fold_official_router,
)
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/icml2025_b2_seed0_r2.json"
CROWN_WORKER = PROJECT_ROOT / "act/pipeline/moe/icml2025_b2_crown.py"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(path)


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def endpoint_normalized_to_unit_pixels(value: np.ndarray) -> np.ndarray:
    """Invert the released 0--255 normalization into unit pixel space."""

    array = np.asarray(value, dtype=np.float32)
    mean = CIFAR_MEAN_255.astype(np.float32)[None, :, None, None]
    std = CIFAR_STD_255.astype(np.float32)[None, :, None, None]
    return ((array * std + mean) / np.float32(255.0)).astype(np.float32)


def unit_pixels_to_normalized(value: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(
        CIFAR_MEAN_255, dtype=value.dtype, device=value.device
    )[None, :, None, None]
    std = torch.as_tensor(
        CIFAR_STD_255, dtype=value.dtype, device=value.device
    )[None, :, None, None]
    return (value * 255.0 - mean) / std


def _top1_margin(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _reference_outputs(
    model: torch.nn.Module,
    pixels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    router_rows: list[np.ndarray] = []
    folded_rows: list[np.ndarray] = []
    expert_rows: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    route_rows: list[np.ndarray] = []
    wrapper_rows: list[np.ndarray] = []
    folded_weight, folded_bias = fold_official_router(
        model.router.gate.weight.detach().cpu().double().numpy(),
        model.router.gate.bias.detach().cpu().double().numpy(),
    )
    wrappers = [PixelNormalizedExpert(expert).to(device).eval() for expert in model.experts]
    with torch.no_grad():
        for start in range(0, len(pixels), batch_size):
            value = torch.from_numpy(pixels[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            normalized = unit_pixels_to_normalized(value)
            selected, scores = _grouped_official_forward(model, normalized)
            expert_logits = torch.stack(
                [expert(normalized) for expert in model.experts], dim=1
            )
            wrapper_logits = torch.stack([wrapper(value) for wrapper in wrappers], dim=1)
            route = scores.argmax(dim=1)
            router_rows.append(scores.cpu().numpy())
            folded_rows.append(
                (
                    value.detach().cpu().double().numpy().reshape(len(value), -1)
                    @ folded_weight.T
                    + folded_bias
                ).astype(np.float64)
            )
            expert_rows.append(expert_logits.cpu().numpy())
            wrapper_rows.append(wrapper_logits.cpu().numpy())
            selected_rows.append(selected.cpu().numpy())
            prediction_rows.append(selected.argmax(dim=1).cpu().numpy())
            route_rows.append(route.cpu().numpy())
    return {
        "router_scores": np.concatenate(router_rows).astype(np.float32),
        "folded_router_scores": np.concatenate(folded_rows).astype(np.float64),
        "expert_logits": np.concatenate(expert_rows).astype(np.float32),
        "wrapper_logits": np.concatenate(wrapper_rows).astype(np.float32),
        "selected_logits": np.concatenate(selected_rows).astype(np.float32),
        "predictions": np.concatenate(prediction_rows).astype(np.int64),
        "routes": np.concatenate(route_rows).astype(np.int64),
    }


def _comparison_summary(
    reference: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, Any]:
    tolerance = config["tolerances"]
    folded_error = float(
        np.max(
            np.abs(
                reference["folded_router_scores"]
                - reference["router_scores"].astype(np.float64)
            )
        )
    )
    wrapper_error = float(
        np.max(np.abs(reference["wrapper_logits"] - reference["expert_logits"]))
    )
    routes_from_folded = reference["folded_router_scores"].argmax(axis=1)
    tie = (
        _top1_margin(reference["router_scores"])
        <= float(tolerance["tie_tolerance"])
    ) | (
        _top1_margin(reference["folded_router_scores"])
        <= float(tolerance["tie_tolerance"])
    )
    nontie = ~tie
    agreement = routes_from_folded == reference["routes"]
    nontie_agreement = float(agreement[nontie].mean()) if nontie.any() else 1.0
    selected_from_experts = reference["expert_logits"][
        np.arange(len(reference["routes"])), reference["routes"]
    ]
    selected_error = float(
        np.max(np.abs(selected_from_experts - reference["selected_logits"]))
    )
    status = "PASS"
    if folded_error > float(tolerance["folded_router_atol"]):
        status = "FAIL"
    if wrapper_error > float(tolerance["route_a_logit_atol"]):
        status = "FAIL"
    if selected_error > float(tolerance["route_a_logit_atol"]):
        status = "FAIL"
    if nontie_agreement < float(tolerance["required_nontie_route_agreement"]):
        status = "FAIL"
    return {
        "status": status,
        "folded_router_maximum_abs_error": folded_error,
        "specialized_wrapper_maximum_abs_error": wrapper_error,
        "selected_gather_maximum_abs_error": selected_error,
        "tie_cases": int(tie.sum()),
        "nontie_route_agreement": nontie_agreement,
        "prediction_count": int(len(reference["predictions"])),
    }


def prepare(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"B2 prepare refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("B2 config is not frozen")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")
    checkpoint = _inside(Path(config["checkpoint"]["path"]), WRITE_ROOT)
    endpoint = _inside(Path(config["endpoint"]["path"]), WRITE_ROOT)
    if _sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise RuntimeError("B2 checkpoint identity changed")
    if _sha256(endpoint) != config["endpoint"]["sha256"]:
        raise RuntimeError("B2 endpoint identity changed")
    root = Path(get_torchvision_data_root()).resolve()
    if not root.is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision root escapes /data1/Kane/MOE")
    import torchvision.datasets as datasets

    dataset = datasets.CIFAR10(
        root=str(root / "CIFAR10/raw"), train=False, download=False
    )
    selection = config["selection"]
    start = int(selection["start"])
    stop = start + int(selection["samples"])
    indices = np.arange(start, stop, dtype=np.int64)
    raw = np.asarray(dataset.data, dtype=np.uint8)[indices]
    clean_pixels = (
        raw.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255.0)
    )
    with np.load(endpoint, allow_pickle=False) as endpoint_arrays:
        labels = endpoint_arrays["labels"].astype(np.int64)[indices]
        endpoint_normalized = endpoint_arrays["adversarial_endpoints"].astype(
            np.float32
        )[indices]
        stored_clean_predictions = endpoint_arrays["clean_predictions"].astype(
            np.int64
        )[indices]
        stored_adversarial_predictions = endpoint_arrays[
            "adversarial_predictions"
        ].astype(np.int64)[indices]
        stored_clean_routes = endpoint_arrays["clean_routes"].astype(np.int64)[indices]
        stored_adversarial_routes = endpoint_arrays["adversarial_routes"].astype(
            np.int64
        )[indices]
    if not np.array_equal(labels, np.asarray(dataset.targets, dtype=np.int64)[indices]):
        raise RuntimeError("B2 ordered labels differ from endpoint artifact")
    endpoint_pixels = endpoint_normalized_to_unit_pixels(endpoint_normalized)
    device = torch.device(config["execution"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("B2 CUDA device is unavailable")
    sys.dont_write_bytecode = True
    model, payload = _load_official_model(checkpoint, device)
    if int(payload.get("epoch", -1)) + 1 != int(config["checkpoint"]["epoch"]):
        raise RuntimeError("B2 checkpoint epoch changed")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("B2 official model is not entirely in eval mode")
    families = {
        "clean_uint8": clean_pixels,
        "official_pgd50_endpoint": endpoint_pixels,
    }
    references = {
        name: _reference_outputs(
            model,
            pixels,
            device=device,
            batch_size=int(config["execution"]["batch_size"]),
        )
        for name, pixels in families.items()
    }
    comparisons = {
        name: _comparison_summary(reference, config)
        for name, reference in references.items()
    }
    if any(row["status"] != "PASS" for row in comparisons.values()):
        raise RuntimeError(f"B2 reference conformance failed: {comparisons}")
    clean_prediction_agreement = float(
        np.mean(references["clean_uint8"]["predictions"] == stored_clean_predictions)
    )
    clean_route_agreement = float(
        np.mean(references["clean_uint8"]["routes"] == stored_clean_routes)
    )
    adversarial_prediction_agreement = float(
        np.mean(
            references["official_pgd50_endpoint"]["predictions"]
            == stored_adversarial_predictions
        )
    )
    adversarial_route_agreement = float(
        np.mean(
            references["official_pgd50_endpoint"]["routes"]
            == stored_adversarial_routes
        )
    )
    required_prediction = float(
        config["tolerances"]["required_prediction_agreement"]
    )
    required_route = float(
        config["tolerances"]["required_nontie_route_agreement"]
    )
    if clean_prediction_agreement < required_prediction:
        raise RuntimeError("B2 clean predictions differ from audited B1 endpoint")
    if adversarial_prediction_agreement < required_prediction:
        raise RuntimeError("B2 adversarial predictions differ from audited B1 endpoint")
    if clean_route_agreement < required_route or adversarial_route_agreement < required_route:
        raise RuntimeError("B2 routes differ from audited B1 endpoint")
    output_dir.mkdir(parents=True)
    artifact = output_dir / "reference.npz"
    arrays: dict[str, np.ndarray] = {
        "dataset_indices": indices,
        "labels": labels,
        "clean_uint8_pixels": clean_pixels,
        "official_pgd50_endpoint_pixels": endpoint_pixels,
        "official_pgd50_endpoint_normalized": endpoint_normalized,
    }
    for family, reference in references.items():
        for name, value in reference.items():
            arrays[f"{family}__{name}"] = value
    artifact_sha256 = _save_npz(artifact, **arrays)
    bn_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)
    ]
    result = {
        "schema_version": 1,
        "status": "PREPARED_CROWN_NOT_RUN",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "epoch": int(payload.get("epoch", -1)) + 1,
        },
        "endpoint": {"path": str(endpoint), "sha256": _sha256(endpoint)},
        "artifact": {"path": str(artifact), "sha256": artifact_sha256},
        "selection": {"indices": indices.tolist(), "samples": len(indices)},
        "reference_comparisons": comparisons,
        "audited_b1_agreement": {
            "clean_prediction": clean_prediction_agreement,
            "clean_route": clean_route_agreement,
            "adversarial_prediction": adversarial_prediction_agreement,
            "adversarial_route": adversarial_route_agreement,
        },
        "model_identity": {
            "model_state_sha256": _module_state_sha256(model),
            "router_state_sha256": _module_state_sha256(model.router),
            "expert_state_sha256": [
                _module_state_sha256(expert) for expert in model.experts
            ],
            "batchnorm2d_layers": len(bn_layers),
            "batchnorm_all_eval": all(not layer.training for layer in bn_layers),
            "batchnorm_all_track_running_stats": all(
                bool(layer.track_running_stats) for layer in bn_layers
            ),
        },
        "official_source": {"commit": OFFICIAL_COMMIT, "clone_clean": True},
        "claim_scope": config["claim_scope"],
    }
    _write_json(output_dir / "prepare.json", result)
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official repository became dirty during B2 prepare")
    return result


def run_crown_worker(prepare_path: Path, output_path: Path, crown_python: Path) -> int:
    prepare_path = _inside(prepare_path, WRITE_ROOT)
    output_path = _inside(output_path, WRITE_ROOT)
    crown_python = _inside(crown_python, WRITE_ROOT)
    command = [
        str(crown_python),
        str(CROWN_WORKER),
        "--prepare",
        str(prepare_path),
        "--output",
        str(output_path),
    ]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(PROJECT_ROOT),
    }
    return subprocess.run(
        command, cwd=PROJECT_ROOT, env=environment, check=False
    ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    crown_parser = subparsers.add_parser("crown")
    crown_parser.add_argument("--prepare", type=Path, required=True)
    crown_parser.add_argument("--output", type=Path, required=True)
    crown_parser.add_argument("--crown-python", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.config, args.output_dir)
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        raise SystemExit(run_crown_worker(args.prepare, args.output, args.crown_python))


if __name__ == "__main__":
    main()
