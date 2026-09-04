"""Run the frozen RT-ER B2 concrete auto_LiRPA conversion check."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from act.util.typing_compat import install_typing_override

install_typing_override()

import numpy as np
import torch

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_b2_conformance import _save_npz
from act.pipeline.moe.icml2025_b3 import PixelNormalizedExpert, _nvidia_driver_version
from act.pipeline.moe.icml2025_route_telemetry import (
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _load_official_model,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def _maximum_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def run(prepare_path: Path, output_path: Path) -> dict[str, Any]:
    prepare_path = _inside(prepare_path, WRITE_ROOT)
    output_path = _inside(output_path, WRITE_ROOT)
    if output_path.exists():
        raise RuntimeError(f"B2 CROWN worker refuses to overwrite {output_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepare.get("status") != "PREPARED_CROWN_NOT_RUN":
        raise RuntimeError("B2 prepare artifact is not ready")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")
    checkpoint = _inside(Path(prepare["checkpoint"]["path"]), WRITE_ROOT)
    artifact = _inside(Path(prepare["artifact"]["path"]), WRITE_ROOT)
    if _sha256(checkpoint) != prepare["checkpoint"]["sha256"]:
        raise RuntimeError("B2 checkpoint identity changed")
    if _sha256(artifact) != prepare["artifact"]["sha256"]:
        raise RuntimeError("B2 reference artifact identity changed")
    config_path = _inside(Path(prepare["config"]["path"]), PROJECT_ROOT)
    if _sha256(config_path) != prepare["config"]["sha256"]:
        raise RuntimeError("B2 config identity changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    batch_size = int(config["execution"]["batch_size"])
    device = torch.device(config["execution"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("B2 CROWN CUDA device is unavailable")
    sys.dont_write_bytecode = True
    model, payload = _load_official_model(checkpoint, device)
    if int(payload.get("epoch", -1)) + 1 != int(prepare["checkpoint"]["epoch"]):
        raise RuntimeError("B2 checkpoint epoch changed")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("B2 CROWN model is not entirely in eval mode")
    from auto_LiRPA import BoundedModule

    with np.load(artifact, allow_pickle=False) as arrays:
        families = {
            "clean_uint8": arrays["clean_uint8_pixels"].astype(np.float32),
            "official_pgd50_endpoint": arrays[
                "official_pgd50_endpoint_pixels"
            ].astype(np.float32),
        }
        reference = {
            family: {
                name: arrays[f"{family}__{name}"].copy()
                for name in (
                    "router_scores",
                    "expert_logits",
                    "selected_logits",
                    "predictions",
                    "routes",
                )
            }
            for family in families
        }
    output_arrays: dict[str, np.ndarray] = {}
    family_summaries: dict[str, Any] = {}
    tolerances = config["tolerances"]
    started = time.monotonic()
    for family, pixels in families.items():
        direct_expert = np.empty(
            (len(pixels), len(model.experts), 10), dtype=np.float32
        )
        bounded_expert = np.empty_like(direct_expert)
        for expert_index, source_expert in enumerate(model.experts):
            specialized = PixelNormalizedExpert(copy.deepcopy(source_expert)).to(
                device=device, dtype=torch.float32
            ).eval()
            dummy = torch.from_numpy(pixels[:batch_size]).to(device)
            bounded = BoundedModule(specialized, dummy, device=str(device))
            for start in range(0, len(pixels), batch_size):
                stop = start + batch_size
                value = torch.from_numpy(pixels[start:stop]).to(device)
                with torch.no_grad():
                    direct = specialized(value)
                    converted = bounded(value)
                direct_expert[start:stop, expert_index] = (
                    direct.detach().cpu().numpy()
                )
                bounded_expert[start:stop, expert_index] = (
                    converted.detach().cpu().numpy()
                )
            del bounded
        routes = reference[family]["routes"].astype(np.int64)
        selected = direct_expert[np.arange(len(routes)), routes]
        predictions = selected.argmax(axis=1).astype(np.int64)
        direct_reference_error = _maximum_abs(
            direct_expert, reference[family]["expert_logits"]
        )
        bounded_direct_error = _maximum_abs(bounded_expert, direct_expert)
        selected_reference_error = _maximum_abs(
            selected, reference[family]["selected_logits"]
        )
        prediction_agreement = float(
            np.mean(predictions == reference[family]["predictions"])
        )
        status = "PASS"
        if direct_reference_error > float(
            tolerances["cross_runtime_direct_logit_atol"]
        ):
            status = "FAIL"
        if bounded_direct_error > float(tolerances["auto_lirpa_logit_atol"]):
            status = "FAIL"
        if selected_reference_error > float(
            tolerances["cross_runtime_direct_logit_atol"]
        ):
            status = "FAIL"
        if prediction_agreement < float(
            tolerances["required_prediction_agreement"]
        ):
            status = "FAIL"
        family_summaries[family] = {
            "status": status,
            "samples": len(pixels),
            "direct_cross_runtime_maximum_abs_error": direct_reference_error,
            "auto_lirpa_concrete_maximum_abs_error": bounded_direct_error,
            "selected_cross_runtime_maximum_abs_error": selected_reference_error,
            "prediction_agreement": prediction_agreement,
        }
        output_arrays[f"{family}__direct_expert_logits"] = direct_expert
        output_arrays[f"{family}__bounded_expert_logits"] = bounded_expert
        output_arrays[f"{family}__selected_logits"] = selected
        output_arrays[f"{family}__predictions"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output_path.with_suffix(output_path.suffix + ".npz")
    arrays_sha256 = _save_npz(arrays_path, **output_arrays)
    overall = "PASS" if all(
        row["status"] == "PASS" for row in family_summaries.values()
    ) else "FAIL"
    result = {
        "schema_version": 1,
        "status": overall,
        "scope": config["scope"],
        "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": prepare["checkpoint"],
        "arrays": {"path": str(arrays_path), "sha256": arrays_sha256},
        "families": family_summaries,
        "runtime_seconds": time.monotonic() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "auto_lirpa": importlib.metadata.version("auto-lirpa"),
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": _nvidia_driver_version(),
            "device": str(device),
        },
        "claim_scope": config["claim_scope"],
    }
    _write_json(output_path, result)
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official repository became dirty during B2 CROWN worker")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prepare, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
