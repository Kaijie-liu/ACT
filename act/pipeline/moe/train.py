# ===- act/pipeline/moe/train.py - Controlled MoE Training CLI ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
from act.util.path_config import get_project_root, get_torchvision_data_root


DATASETS: dict[str, dict[str, Any]] = {
    "MNIST": {
        "class": "MNIST",
        "input_shape": (1, 28, 28),
        "num_classes": 10,
        "train": {"train": True},
        "test": {"train": False},
    },
    "FashionMNIST": {
        "class": "FashionMNIST",
        "input_shape": (1, 28, 28),
        "num_classes": 10,
        "train": {"train": True},
        "test": {"train": False},
    },
    "CIFAR10": {
        "class": "CIFAR10",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "train": {"train": True},
        "test": {"train": False},
    },
    "SVHN": {
        "class": "SVHN",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "train": {"split": "train"},
        "test": {"split": "test"},
    },
}


def _load_dataset(name: str, train: bool, download: bool):
    import torchvision.datasets as tv_datasets
    import torchvision.transforms as transforms

    info = DATASETS[name]
    dataset_class = getattr(tv_datasets, info["class"])
    root = Path(get_torchvision_data_root()) / name / "raw"
    kwargs = info["train" if train else "test"]
    return dataset_class(
        root=str(root),
        transform=transforms.ToTensor(),
        download=download,
        **kwargs,
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    correct = total = 0
    route_counts = torch.zeros(model.spec.num_experts, dtype=torch.long)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, decision = model.forward_with_routing(inputs)
        correct += int((outputs.argmax(dim=-1) == labels).sum().item())
        total += int(labels.numel())
        route_counts += torch.bincount(
            decision.indices.detach().cpu().reshape(-1),
            minlength=model.spec.num_experts,
        )
    return {
        "accuracy": correct / max(total, 1),
        "route_counts": route_counts.tolist(),
        "samples": total,
    }


def train(args) -> Path:
    torch.manual_seed(args.seed)
    device = _device(args.device)
    info = DATASETS[args.dataset]
    config = OutputMoEFactoryConfig(
        input_shape=info["input_shape"],
        num_classes=info["num_classes"],
        num_experts=args.num_experts,
        top_k=args.top_k,
        gate=GateKind(args.gate),
        router_hidden=tuple(args.router_hidden),
        expert_hidden=tuple(args.expert_hidden),
        seed=args.seed,
    )
    model = build_output_moe(config).to(device)
    train_data = _load_dataset(args.dataset, True, args.download)
    test_data = _load_dataset(args.dataset, False, args.download)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs, decision = model.forward_with_routing(inputs)
            classification = F.cross_entropy(outputs, labels)
            mean_probability = torch.softmax(decision.scores, dim=-1).mean(dim=0)
            balance = model.spec.num_experts * mean_probability.square().sum()
            loss = classification + args.balance_coefficient * balance
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.numel()
            correct += int((outputs.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
        metrics = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch:03d} loss={running_loss / max(total, 1):.5f} "
            f"train_acc={correct / max(total, 1):.4f} "
            f"test_acc={metrics['accuracy']:.4f} routes={metrics['route_counts']}"
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(get_project_root()) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_config = asdict(config)
    payload_config["input_shape"] = list(config.input_shape)
    payload_config["router_hidden"] = list(config.router_hidden)
    payload_config["expert_hidden"] = list(config.expert_hidden)
    payload_config["gate"] = config.gate.value
    torch.save(
        {
            "format": "act-output-moe-v1",
            "dataset": args.dataset,
            "factory_config": payload_config,
            "state_dict": model.state_dict(),
            "test_metrics": evaluate(model, test_loader, device),
        },
        output,
    )
    print(f"saved={output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a controlled ACT output-level MoE")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="MNIST")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--gate", choices=tuple(kind.value for kind in GateKind), default="hard_top1"
    )
    parser.add_argument("--router-hidden", type=int, nargs="*", default=[])
    parser.add_argument("--expert-hidden", type=int, nargs="*", default=[64])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balance-coefficient", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="data/moe/checkpoints/output_moe.pt")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
