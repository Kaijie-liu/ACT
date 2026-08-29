"""Launch the pinned official RT-ER script with disclosed runtime shims.

The official source file is executed unchanged.  This launcher fixes stochastic
state that the paper left unspecified, injects the same seed into FFCV loaders,
and replaces the broken ``--nowandb`` path with a local JSONL no-op module.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import runpy
import sys
import types
from typing import Any

import numpy as np
import torch


OFFICIAL_REPO = Path("/data1/Kane/MOE/baselines/Robust-MoE-Dual-Model")
OFFICIAL_SCRIPT = OFFICIAL_REPO / "cifar10_RT_ER.py"


def _jsonable(value: Any):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class _WandbConfig:
    def __init__(self, owner: "_WandbStub") -> None:
        self.owner = owner

    def update(self, value) -> None:
        self.owner._append("config.update", vars(value) if hasattr(value, "__dict__") else value)


class _WandbStub(types.ModuleType):
    def __init__(self, log_path: Path) -> None:
        super().__init__("wandb")
        self.log_path = log_path
        self.config = _WandbConfig(self)

    def _append(self, event: str, payload: Any) -> None:
        record = {"event": event, "payload": _jsonable(payload)}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def init(self, **kwargs):
        self._append("init", kwargs)
        return self

    def log(self, payload):
        self._append("log", payload)

    def save(self, value):
        self._append("save", value)


def author_arguments(epochs: int, seed: int) -> list[str]:
    del seed  # The seed is injected before the unchanged official main runs.
    return [
        "--net",
        "res18_moe",
        "--n_epochs",
        str(int(epochs)),
        "--beta",
        "6",
        "--bs",
        "512",
        "--lr",
        "0.0001",
        "--opt",
        "adam",
        "--nowandb",
    ]


def configure_reproduction(seed: int, wandb_log: Path) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    import ffcv.loader as ffcv_loader

    original_loader = ffcv_loader.Loader

    def seeded_loader(*args, **kwargs):
        if kwargs.get("seed") is None:
            kwargs["seed"] = int(seed)
        return original_loader(*args, **kwargs)

    ffcv_loader.Loader = seeded_loader
    sys.modules["wandb"] = _WandbStub(wandb_log)
    return {
        "seed": int(seed),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "ffcv_default_seed_injected": True,
        "wandb_mode": "local_jsonl_noop_compatibility_shim",
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "ffcv": importlib.metadata.version("ffcv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", type=Path, default=OFFICIAL_SCRIPT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--wandb-log", type=Path, required=True)
    parser.add_argument("--launcher-manifest", type=Path, required=True)
    arguments = parser.parse_args()
    script = arguments.official_script.resolve()
    if script != OFFICIAL_SCRIPT.resolve() or not script.is_file():
        raise RuntimeError("launcher requires the pinned official CIFAR RT-ER script")
    for path in (arguments.wandb_log, arguments.launcher_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(f"launcher refuses to overwrite {path}")
    runtime = configure_reproduction(arguments.seed, arguments.wandb_log)
    runtime.update(
        {
            "official_script": str(script),
            "official_arguments": author_arguments(arguments.epochs, arguments.seed),
            "patch_classification": {
                "seed_and_ffcv_seed": "reproducibility",
                "wandb_noop": "compatibility",
                "official_source_modified": False,
                "scientific_patch": False,
            },
        }
    )
    with arguments.launcher_manifest.open("x", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    sys.path.insert(0, str(OFFICIAL_REPO))
    sys.argv = [str(script), *author_arguments(arguments.epochs, arguments.seed)]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
