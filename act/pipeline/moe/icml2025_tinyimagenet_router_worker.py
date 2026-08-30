"""Isolated worker for official TinyImageNet MOE_ViT router construction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


MOE_ROOT = Path("/data1/Kane/MOE")
OFFICIAL_REPO = MOE_ROOT / "baselines/Robust-MoE-Dual-Model"
OFFICIAL_COMMIT = "30ef94d77b5451595b82e739aa8938e1f4c4521f"


def _inside(path: Path) -> Path:
    value = path.resolve()
    if not value.is_relative_to(MOE_ROOT):
        raise RuntimeError(f"TinyImageNet worker path escapes {MOE_ROOT}: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path)
    output_dir = _inside(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"TinyImageNet worker refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("TinyImageNet census config is not frozen")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")
    for key in ("training_script", "model_source"):
        source = _inside(Path(config["official_source"][key]))
        expected = config["official_source"][f"{key}_sha256"]
        if _sha256(source) != expected:
            raise RuntimeError(f"official {key} changed")

    torch.set_num_threads(1)
    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        from models.moe import MOE_ViT  # type: ignore
    finally:
        sys.path.pop(0)

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    hashes: list[str] = []
    shapes: list[list[int]] = []
    for seed in config["initialization"]["seeds"]:
        torch.manual_seed(int(seed))
        model = MOE_ViT(num_experts=4, size=224).cpu().eval()
        weight_tensor = model.router.gate.weight.detach().cpu()
        bias_tensor = model.router.gate.bias.detach().cpu()
        if tuple(weight_tensor.shape) != (4, 3 * 224 * 224):
            raise RuntimeError("official TinyImageNet router shape changed")
        weight = weight_tensor.double().numpy().copy()
        bias = bias_tensor.double().numpy().copy()
        digest = hashlib.sha256()
        digest.update(weight.tobytes(order="C"))
        digest.update(bias.tobytes(order="C"))
        weights.append(weight)
        biases.append(bias)
        hashes.append(digest.hexdigest())
        shapes.append(list(weight_tensor.shape))
        del model, weight_tensor, bias_tensor
        gc.collect()
    if hashes[0] != config["initialization"]["seed0_expected_router_sha256"]:
        raise RuntimeError("seed-0 official construction smoke hash changed")

    output_dir.mkdir(parents=True)
    arrays_path = output_dir / "routers.npz"
    with arrays_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            seeds=np.asarray(config["initialization"]["seeds"], dtype=np.int64),
            weights=np.stack(weights),
            biases=np.stack(biases),
        )
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        "official_commit": OFFICIAL_COMMIT,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "seeds": [int(value) for value in config["initialization"]["seeds"]],
        "construction": config["initialization"]["policy"],
        "router_shapes": shapes,
        "router_hashes": hashes,
        "arrays": {"path": str(arrays_path), "sha256": _sha256(arrays_path)},
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "timm": importlib.metadata.version("timm"),
            "hf_home": os.environ.get("HF_HOME"),
            "torch_home": os.environ.get("TORCH_HOME"),
        },
        "official_clone_clean_after": not bool(_repo_value("status", "--porcelain")),
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
