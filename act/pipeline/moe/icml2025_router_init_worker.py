"""Standalone Python-3.11 worker for exact official router initialization."""

from __future__ import annotations

import argparse
import hashlib
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
        raise RuntimeError(f"router-init path escapes {MOE_ROOT}: {path}")
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
        raise RuntimeError(f"router-init worker refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("router-init config is not frozen")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")
    checkpoint = _inside(Path(config["seed0_reference"]["checkpoint"]))
    if _sha256(checkpoint) != config["seed0_reference"]["checkpoint_sha256"]:
        raise RuntimeError("seed-0 reference checkpoint changed")

    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        from models.moe import MOE_Resnet18  # type: ignore
    finally:
        sys.path.pop(0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_state = payload.get("net")
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError("seed-0 checkpoint lacks model state")
    if checkpoint_state and all(str(key).startswith("module.") for key in checkpoint_state):
        checkpoint_state = {str(key)[7:]: value for key, value in checkpoint_state.items()}

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    hashes: list[str] = []
    seed0_match: dict[str, bool] = {}
    for seed in config["initialization"]["seeds"]:
        torch.manual_seed(int(seed))
        model = MOE_Resnet18(num_experts=4, num_classes=10, size=32).cpu().eval()
        state = model.state_dict()
        weight = state["router.gate.weight"].detach().double().numpy().copy()
        bias = state["router.gate.bias"].detach().double().numpy().copy()
        if int(seed) == 0:
            seed0_match = {
                "router.gate.weight": bool(
                    torch.equal(state["router.gate.weight"], checkpoint_state["router.gate.weight"])
                ),
                "router.gate.bias": bool(
                    torch.equal(state["router.gate.bias"], checkpoint_state["router.gate.bias"])
                ),
            }
            if not all(seed0_match.values()):
                raise RuntimeError("official full-model seed-0 reconstruction changed")
        digest = hashlib.sha256()
        digest.update(weight.tobytes(order="C"))
        digest.update(bias.tobytes(order="C"))
        weights.append(weight)
        biases.append(bias)
        hashes.append(digest.hexdigest())
        del model

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
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "seeds": [int(value) for value in config["initialization"]["seeds"]],
        "router_hashes": hashes,
        "seed0_bitwise_checkpoint_match": seed0_match,
        "arrays": {"path": str(arrays_path), "sha256": _sha256(arrays_path)},
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
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
