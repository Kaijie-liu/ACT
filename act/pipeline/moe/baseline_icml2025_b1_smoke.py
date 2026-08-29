"""Real-data B1 smoke for the official RT-ER architecture and objective."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


MOE_ROOT = Path("/data1/Kane/MOE")
PROJECT_ROOT = MOE_ROOT / "ACT"
OFFICIAL_REPO = MOE_ROOT / "baselines/Robust-MoE-Dual-Model"
OFFICIAL_COMMIT = "30ef94d77b5451595b82e739aa8938e1f4c4521f"
SOURCE_DATA = PROJECT_ROOT / "data/torchvision/CIFAR10/raw"
SOURCE_ARCHIVE_SHA256 = (
    "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
)
AUTHOR_BATCH_SIZE = 512
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path = MOE_ROOT) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"path escapes {root}: {path}")
    return resolved


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def prepare_dataset(run_root: Path) -> Path:
    """Copy the already-audited official dataset into the isolated run root."""

    run_root = _inside(run_root)
    data_root = run_root / "data"
    target = data_root / "cifar-10-batches-py"
    archive = data_root / "cifar-10-python.tar.gz"
    source_archive = SOURCE_DATA / "cifar-10-python.tar.gz"
    source_extract = SOURCE_DATA / "cifar-10-batches-py"
    if not source_archive.is_file() or not source_extract.is_dir():
        raise RuntimeError("audited CIFAR-10 source is incomplete")
    if _sha256(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("audited CIFAR-10 archive checksum changed")
    data_root.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copytree(source_extract, target)
    if not archive.exists():
        shutil.copy2(source_archive, archive)
    if _sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("isolated CIFAR-10 archive checksum mismatch")
    return data_root


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _route_counts(model, value: torch.Tensor) -> list[int]:
    with torch.no_grad():
        routes = model.router(value)
    return torch.bincount(routes, minlength=4).cpu().tolist()


def run(run_root: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    run_root = _inside(run_root)
    output_dir = _inside(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"B1 smoke refuses to overwrite {output_dir}")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise RuntimeError("official repository commit changed")
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official repository is not fully clean")
    if not torch.cuda.is_available():
        raise RuntimeError("B1 smoke requires CUDA")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data_root = prepare_dataset(run_root)

    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        from attack.PGD import PGD
        from models.moe import MOE_Resnet18
        from ffcv.fields import IntField, RGBImageField
        from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
        from ffcv.loader import Loader, OrderOption
        from ffcv.pipeline.operation import Operation
        from ffcv.transforms import (
            Convert,
            Cutout,
            RandomHorizontalFlip,
            RandomTranslate,
            Squeeze,
            ToDevice,
            ToTensor,
            ToTorchImage,
        )
        from ffcv.writer import DatasetWriter
        import torchvision
        from torch.utils.data import Subset
    finally:
        sys.path.pop(0)

    datasets = {
        "train": torchvision.datasets.CIFAR10(root=str(data_root), train=True),
        "test": torchvision.datasets.CIFAR10(root=str(data_root), train=False),
    }
    smoke_sets = {
        "train": Subset(datasets["train"], list(range(16))),
        "test": Subset(datasets["test"], list(range(16))),
    }
    beton_dir = output_dir / "beton"
    beton_dir.mkdir()
    loaders = {}
    device = torch.device("cuda")
    for name, dataset in smoke_sets.items():
        beton = beton_dir / f"cifar_{name}_smoke.beton"
        writer = DatasetWriter(
            str(beton), {"image": RGBImageField(), "label": IntField()}
        )
        writer.from_indexed_dataset(dataset)
        label_pipeline: list[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(device),
            Squeeze(),
        ]
        image_pipeline: list[Operation] = [SimpleRGBImageDecoder()]
        if name == "train":
            image_pipeline.extend(
                [
                    RandomHorizontalFlip(),
                    RandomTranslate(padding=2),
                    Cutout(8, tuple(map(int, CIFAR_MEAN))),
                ]
            )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice(device, non_blocking=True),
                ToTorchImage(),
                Convert(torch.float16),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        loaders[name] = Loader(
            str(beton),
            batch_size=4,
            num_workers=2,
            order=OrderOption.RANDOM,
            seed=seed,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    model = MOE_Resnet18(num_experts=4, num_classes=10, size=32).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    mean = torch.tensor(np.asarray(CIFAR_MEAN, dtype=np.float32)[None, :, None, None])
    std = torch.tensor(np.asarray(CIFAR_STD, dtype=np.float32)[None, :, None, None])
    pgd_train = PGD(
        eps=8 / 255,
        sigma=2 / 255,
        nb_iter=10,
        DEVICE="cuda",
        mean=mean,
        std=std,
    )
    pgd_eval = PGD(
        eps=8 / 255,
        sigma=2 / 255,
        nb_iter=50,
        DEVICE="cuda",
        mean=mean,
        std=std,
    )

    model.train()
    inputs, targets = next(iter(loaders["train"]))
    inputs, targets = inputs.contiguous(), targets.to(device)
    clean_route_counts = _route_counts(model, inputs)
    second = model.router.get_second_expert(inputs)
    second_route_counts = torch.bincount(second, minlength=4).cpu().tolist()
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(inputs)
        adversarial = pgd_train.attack(model, inputs, targets)
        adversarial_outputs = model(adversarial)
        loss_model = criterion(adversarial_outputs, targets)
        loss_expert = torch.zeros((), device=device)
        for expert_id, expert in enumerate(model.experts):
            mask = second == expert_id
            if int(mask.sum()) == 0:
                continue
            selected_inputs = inputs[mask].detach()
            selected_targets = targets[mask].detach()
            expert_adversarial = pgd_train.attack(
                expert, selected_inputs, selected_targets
            )
            expert_outputs = expert(selected_inputs)
            expert_outputs_adversarial = expert(expert_adversarial)
            loss_expert = loss_expert + criterion_kl(
                F.log_softmax(expert_outputs_adversarial, dim=1),
                F.softmax(expert_outputs, dim=1),
            )
        # Preserve the author's fixed BATCH_SIZE denominator even though the
        # deliberately bounded smoke batch is smaller.
        loss = loss_model + 6.0 * (1.0 / AUTHOR_BATCH_SIZE) * loss_expert
    scaler.scale(loss).backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    model.eval()
    test_inputs, test_targets = next(iter(loaders["test"]))
    test_inputs, test_targets = test_inputs.contiguous(), test_targets.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        restored_reference = model(test_inputs)
    checkpoint_path = output_dir / "smoke_checkpoint.t7"
    torch.save(
        {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": None,
            "epoch": 0,
        },
        checkpoint_path,
    )
    restored = MOE_Resnet18(num_experts=4, num_classes=10, size=32).cuda().eval()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["net"], strict=True)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        restored_output = restored(test_inputs)
    restore_max_error = float((restored_output - restored_reference).abs().max().item())

    evaluation_adversarial = pgd_eval.attack(model, test_inputs, test_targets)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        clean_eval = model(test_inputs)
        adversarial_eval = model(evaluation_adversarial)
    result = {
        "schema_version": 1,
        "stage": "B1_REAL_DATA_OFFICIAL_OBJECTIVE_SMOKE",
        "status": "PASSED",
        "label": "official-code, Blackwell-compatible deps + FFCV",
        "seed": int(seed),
        "official_source": {
            "repository": str(OFFICIAL_REPO),
            "commit": OFFICIAL_COMMIT,
            "tracked_status_clean": not bool(
                _repo_value("status", "--porcelain", "--untracked-files=no")
            ),
            "full_status": _repo_value("status", "--porcelain"),
        },
        "dataset": {
            "source_archive": str(data_root / "cifar-10-python.tar.gz"),
            "source_archive_sha256": _sha256(data_root / "cifar-10-python.tar.gz"),
            "official_train_samples": len(datasets["train"]),
            "official_test_samples": len(datasets["test"]),
            "smoke_samples_per_split": 16,
            "ffcv_train_beton_sha256": _sha256(beton_dir / "cifar_train_smoke.beton"),
            "ffcv_test_beton_sha256": _sha256(beton_dir / "cifar_test_smoke.beton"),
        },
        "training_batch": {
            "batch_size": int(inputs.shape[0]),
            "dtype": str(inputs.dtype),
            "device": str(inputs.device),
            "loss": float(loss.detach().cpu()),
            "whole_model_loss": float(loss_model.detach().cpu()),
            "expert_kl_loss": float(loss_expert.detach().cpu()),
            "expert_loss_denominator": AUTHOR_BATCH_SIZE,
            "gradients_finite": gradients_finite,
            "optimizer_step_executed": True,
            "clean_route_counts": clean_route_counts,
            "second_route_counts": second_route_counts,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "strict_load_passed": True,
            "restored_logit_max_abs_error": restore_max_error,
        },
        "evaluation_batch": {
            "batch_size": int(test_inputs.shape[0]),
            "clean_accuracy": float(
                (clean_eval.argmax(dim=1) == test_targets).float().mean().item()
            ),
            "pgd50_accuracy": float(
                (adversarial_eval.argmax(dim=1) == test_targets).float().mean().item()
            ),
            "clean_route_counts": _route_counts(model, test_inputs),
            "adversarial_route_counts": _route_counts(model, evaluation_adversarial),
            "outputs_finite": bool(torch.isfinite(clean_eval).all())
            and bool(torch.isfinite(adversarial_eval).all()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "ffcv": importlib.metadata.version("ffcv"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "ld_preload": os.environ.get("LD_PRELOAD"),
        },
        "runtime_seconds": time.monotonic() - started,
        "scientific_scope": (
            "real official CIFAR decoding, FFCV, official architecture, PGD loss, "
            "one optimizer step, checkpoint roundtrip, and fixed-batch evaluation; "
            "not an epoch, accuracy reproduction, or scientific endpoint"
        ),
    }
    if not gradients_finite or restore_max_error != 0.0:
        result["status"] = "FAILED"
    _write_json(output_dir / "summary.json", result)
    if result["status"] != "PASSED":
        raise RuntimeError("B1 real-data smoke did not pass")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.run_root, arguments.output_dir, arguments.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
