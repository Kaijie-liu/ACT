"""Audit the exact author-pinned RT-ER environment before any B1 data write."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def probe(repo: Path, expected_commit: str) -> dict[str, Any]:
    import torch
    import torchvision
    import ffcv
    import timm
    import einops

    source = (repo / "cifar10_RT_ER.py").read_text(encoding="utf-8")
    requirements = (repo / "requirements.txt").read_text(encoding="utf-8")
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    kernel_error = None
    try:
        value = torch.ones(8, device="cuda")
        kernel_sum = float(value.sum().item())
        kernel_passed = kernel_sum == 8.0
    except Exception as exc:  # the expected author-pin/Blackwell gate
        kernel_passed = False
        kernel_sum = None
        kernel_error = f"{type(exc).__name__}: {exc}"
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    checks = {
        "official_commit_matches": head == expected_commit,
        "official_clone_clean": status == "",
        "wandb_disable_bug_present": "usewandb = ~args.nowandb" in source,
        "unconditional_cuda_present": "net.cuda()" in source,
        "paper_epoch_override_available": "--n_epochs" in source,
        "author_torch_pin_present": "torch==2.4.0" in requirements,
        "author_torchvision_pin_present": "torchvision==0.19.0" in requirements,
        "cuda_kernel_passed": kernel_passed,
    }
    blocking = [
        name
        for name, passed in checks.items()
        if name in {"official_commit_matches", "official_clone_clean", "cuda_kernel_passed"}
        and not passed
    ]
    return {
        "schema_version": 1,
        "stage": "B1_AUTHOR_PIN_COMPATIBILITY_PROBE",
        "official_repository": str(repo),
        "official_commit": head,
        "expected_commit": expected_commit,
        "official_clone_status_porcelain": status,
        "source_sha256": {
            "cifar10_RT_ER.py": _sha256(repo / "cifar10_RT_ER.py"),
            "models/moe.py": _sha256(repo / "models/moe.py"),
            "requirements.txt": _sha256(repo / "requirements.txt"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "device": device,
            "ffcv_distribution": importlib.metadata.version("ffcv"),
            "ffcv_module_version_attribute": getattr(ffcv, "__version__", None),
            "timm": timm.__version__,
            "einops": einops.__version__,
        },
        "checks": checks,
        "cuda_kernel_sum": kernel_sum,
        "cuda_kernel_error": kernel_error,
        "blocking_checks": blocking,
        "b1_smoke_unlocked": not blocking,
        "training_started": False,
        "dataset_conversion_started": False,
        "interpretation": (
            "The exact author dependency pin imports, but B1 remains blocked "
            "when its CUDA binary cannot execute on the installed GPU. A newer "
            "PyTorch compatibility reproduction must be separately labeled."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.repo.resolve(), args.expected_commit)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
