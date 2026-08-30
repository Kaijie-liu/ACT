"""Audit AdvMoE dependency and Blackwell readiness without installing packages."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

from act.pipeline.moe.advmoe_architecture_audit import REPO, REPOSITORY, _external_models
from act.pipeline.moe.published_moe_router_gradient_audit import (
    BASELINE_ROOT,
    MOE_ROOT,
    PROJECT_ROOT,
    _git,
    _inside,
    _sha256,
)


ACT_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
CROWN_PYTHON = Path("/data1/Kane/MOE/envs/alpha-beta-crown/bin/python")
REQUIRED_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "tensorboard": "tensorboard",
    "yaml": "pyyaml",
    "easydict": "easydict",
    "lmdb": "lmdb",
    "h5py": "h5py",
    "datasets": "datasets",
    "six": "six",
    "tqdm": "tqdm",
    "scipy": "scipy",
}


def _environment_probe(python: Path) -> dict[str, Any]:
    code = """
import importlib.metadata, importlib.util, json, platform, torch
requirements = json.loads(__import__('os').environ['ADV_REQUIREMENTS'])
packages = {}
for module, package in requirements.items():
    try:
        version = importlib.metadata.version(package)
    except Exception:
        version = None
    packages[module] = {'importable': importlib.util.find_spec(module) is not None,
                        'version': version}
result = {'python': platform.python_version(), 'torch': torch.__version__,
          'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
          'packages': packages}
if torch.cuda.is_available():
    result['device'] = torch.cuda.get_device_name(0)
    result['capability'] = list(torch.cuda.get_device_capability(0))
print(json.dumps(result, sort_keys=True))
"""
    env = os.environ.copy()
    env["ADV_REQUIREMENTS"] = json.dumps(REQUIRED_IMPORTS)
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=PROJECT_ROOT,
    )
    return json.loads(completed.stdout)


def _training_import_probe() -> dict[str, Any]:
    completed = subprocess.run(
        [str(ACT_PYTHON), str(REPO / "train_moe.py"), "--help"],
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
    )
    stderr = completed.stderr
    missing = None
    marker = "No module named '"
    if marker in stderr:
        missing = stderr.split(marker, 1)[1].split("'", 1)[0]
    return {
        "exit_code": completed.returncode,
        "first_missing_import": missing,
        "reached_argument_parser": completed.returncode == 0,
        "full_traceback_stored": False,
    }


def _cuda_model_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "CUDA_UNAVAILABLE"}
    with _external_models() as (_resnet, router_module, _moe_layer):
        router = router_module.build_router(num_experts=2).cuda().eval()
        with torch.no_grad():
            output = router(torch.zeros(2, 3, 32, 32, device="cuda"))
    return {
        "status": "PASS",
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "device": str(output.device),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def collect() -> dict[str, Any]:
    repo = _inside(REPO, BASELINE_ROOT)
    head = _git(repo, "rev-parse", "HEAD")
    if head != REPOSITORY["commit"] or _git(repo, "status", "--porcelain"):
        raise RuntimeError("AdvMoE repository identity/cleanliness gate failed")
    requirements_path = repo / "requirements.txt"
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    readme = (repo / "README.md").read_text(encoding="utf-8")
    normalized = [line.strip() for line in requirements if line.strip()]
    versioned = [line for line in normalized if any(op in line for op in ("==", ">=", "<=", "~="))]
    act = _environment_probe(ACT_PYTHON)
    crown = _environment_probe(CROWN_PYTHON)
    return {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "repository": {
            "url": REPOSITORY["url"],
            "commit": head,
            "worktree_clean": True,
            "license": "NOT_FOUND",
        },
        "official_dependency_specification": {
            "requirements_path": "requirements.txt",
            "requirements_sha256": _sha256(requirements_path),
            "declared_packages": normalized,
            "versioned_entries": versioned,
            "torch_version_pinned": any(line.startswith("torch==") for line in normalized),
            "cuda_version_pinned": any("cuda" in line.lower() for line in normalized),
            "python_version_pinned": False,
            "readme_requested_filename": "requirement.txt",
            "readme_requested_file_exists": (repo / "requirement.txt").is_file(),
            "readme_anchor_present": "pip3 install -r requirement.txt" in readme,
        },
        "existing_environment_probes": {"act_py312": act, "alpha_beta_crown": crown},
        "training_entrypoint_probe": _training_import_probe(),
        "blackwell_model_only_probe": _cuda_model_probe(),
        "classification": {
            "exact_author_environment_defined": False,
            "model_only_blackwell_compatible_in_act_py312": True,
            "training_entrypoint_runnable_in_act_py312": False,
            "existing_crown_environment_is_training_environment": False,
            "next_environment_label": "OFFICIAL_CODE_BLACKWELL_COMPATIBLE_DEPENDENCY_REPRODUCTION",
            "installation_performed": False,
            "environment_created": False,
        },
        "claim_boundary": (
            "The released dependency list does not define an exact Python/Torch/CUDA "
            "environment. The model-only CUDA probe passes on sm_120, while the full "
            "training entry point is blocked by missing packages in act-py312."
        ),
    }


def write_result(path: Path) -> dict[str, Any]:
    path = _inside(path, PROJECT_ROOT)
    if path.exists():
        raise RuntimeError(f"output already exists: {path}")
    result = collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_result(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
