"""Materialize released TinyImageNet float16 resize centers in the pinned runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
import torchvision


MOE_ROOT = Path("/data1/Kane/MOE")
OFFICIAL_REPO = MOE_ROOT / "baselines/Robust-MoE-Dual-Model"
OFFICIAL_COMMIT = "30ef94d77b5451595b82e739aa8938e1f4c4521f"


def _inside(path: Path) -> Path:
    value = path.resolve()
    if not value.is_relative_to(MOE_ROOT):
        raise RuntimeError(f"preprocessing path escapes {MOE_ROOT}: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def _ordered_images(root: Path, expected: int) -> tuple[list[Path], str]:
    paths = sorted((root / "val/images").glob("*.JPEG"))
    if len(paths) != expected:
        raise RuntimeError("TinyImageNet validation count changed")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
    return paths, digest.hexdigest()


def _load_chunk(paths: list[Path]) -> np.ndarray:
    rows = []
    for path in paths:
        with Image.open(path) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if value.shape != (64, 64, 3):
            raise RuntimeError(f"unexpected image shape at {path}")
        rows.append(value.transpose(2, 0, 1))
    return np.stack(rows)


def run(config_path: Path, router_arrays: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path)
    router_arrays = _inside(router_arrays)
    output_dir = _inside(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("preprocessing config is not frozen")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official clone identity/cleanliness gate failed")
    if torch.__version__ != config["preprocessing"]["runtime"]["torch"]:
        raise RuntimeError("preprocessing torch version changed")
    if torchvision.__version__ != config["preprocessing"]["runtime"][
        "torchvision_import"
    ] or importlib.metadata.version("torchvision") != config["preprocessing"]["runtime"][
        "torchvision_metadata"
    ]:
        raise RuntimeError("preprocessing torchvision version changed")

    with np.load(router_arrays, allow_pickle=False) as artifact:
        seeds = artifact["seeds"].astype(np.int64)
        weights = artifact["weights"].astype(np.float64)
        biases = artifact["biases"].astype(np.float64)
    router_digest = hashlib.sha256()
    router_digest.update(seeds.tobytes())
    router_digest.update(weights.tobytes())
    router_digest.update(biases.tobytes())
    router_content_sha256 = router_digest.hexdigest()
    if seeds.tolist() != config["initialization"]["seeds"]:
        raise RuntimeError("router seeds changed")

    if output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("incomplete preprocessing cache retained")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "COMPLETED"
            or existing.get("router_content_sha256") != router_content_sha256
            or existing.get("config_sha256") != _sha256(config_path)
        ):
            raise RuntimeError("preprocessing cache identity changed")
        for item in existing["outputs"].values():
            path = _inside(Path(item["path"]))
            if _sha256(path) != item["sha256"]:
                raise RuntimeError("preprocessing cache artifact changed")
        return existing

    root = _inside(Path(config["dataset"]["root"]))
    paths, ordered_digest = _ordered_images(root, int(config["dataset"]["ordered_samples"]))
    if ordered_digest != config["dataset"]["ordered_image_names_sha256"]:
        raise RuntimeError("ordered image digest changed")
    if not torch.cuda.is_available():
        raise RuntimeError("released float16 resize materialization requires CUDA")

    output_dir.mkdir(parents=True)
    resized_path = output_dir / "literal_resized_255_float16.npy"
    scores_path = output_dir / "literal_preprocessing_scores_float64.npy"
    resized = np.lib.format.open_memmap(
        resized_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(paths), 3, 224, 224),
    )
    scores = np.lib.format.open_memmap(
        scores_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(seeds), len(paths), 4),
    )
    device = torch.device("cuda")
    tensor_weights = torch.as_tensor(weights, dtype=torch.float64, device=device)
    tensor_biases = torch.as_tensor(biases, dtype=torch.float64, device=device)
    mean = torch.as_tensor(
        config["router"]["normalization_mean_255"], dtype=torch.float16, device=device
    )[None, :, None, None]
    std = torch.as_tensor(
        config["router"]["normalization_std_255"], dtype=torch.float16, device=device
    )[None, :, None, None]
    batch_size = int(config["preprocessing"]["materialization_batch_size"])
    for start in range(0, len(paths), batch_size):
        stop = min(start + batch_size, len(paths))
        raw = torch.as_tensor(
            _load_chunk(paths[start:stop]), dtype=torch.float16, device=device
        )
        literal = functional.interpolate(
            raw,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        if bool(torch.any(literal < 0.0)) or bool(torch.any(literal > 255.0)):
            raise RuntimeError("pinned released resize escaped [0,255]")
        resized[start:stop] = literal.cpu().numpy()
        normalized = ((literal - mean) / std).double().flatten(1)
        for seed_slot in range(len(seeds)):
            value = normalized @ tensor_weights[seed_slot].T + tensor_biases[seed_slot]
            scores[seed_slot, start:stop] = value.cpu().numpy()
    resized.flush()
    scores.flush()
    del resized, scores

    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        "config_sha256": _sha256(config_path),
        "router_content_sha256": router_content_sha256,
        "official_commit": OFFICIAL_COMMIT,
        "ordered_image_names_sha256": ordered_digest,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision_import": torchvision.__version__,
            "torchvision_metadata": importlib.metadata.version("torchvision"),
        },
        "semantics": "released Convert(float16) -> torchvision bilinear Resize(224); literal preprocessing scores additionally apply released float16 Normalize then real-affine router",
        "outputs": {
            "literal_resized": {"path": str(resized_path), "sha256": _sha256(resized_path)},
            "literal_scores": {"path": str(scores_path), "sha256": _sha256(scores_path)},
        },
        "official_clone_clean_after": not bool(_repo_value("status", "--porcelain")),
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--router-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.router_arrays, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
