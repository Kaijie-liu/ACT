# ===- act/pipeline/moe/route_flips.py - Route/Prediction Flip Study ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.train import _device, _load_dataset


def _route_margin(scores: torch.Tensor, original_route: torch.Tensor) -> torch.Tensor:
    selected = scores.gather(1, original_route[:, None]).squeeze(1)
    mask = F.one_hot(original_route, scores.shape[1]).bool()
    competitor = scores.masked_fill(mask, float("-inf")).max(dim=1).values
    return competitor - selected


def pgd(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    steps: int,
    step_size: float,
    objective: str,
) -> torch.Tensor:
    with torch.no_grad():
        original_route = model.route(inputs).indices[:, 0]
    lower = (inputs - epsilon).clamp(0.0, 1.0)
    upper = (inputs + epsilon).clamp(0.0, 1.0)
    adversarial = lower + torch.rand_like(inputs) * (upper - lower)
    for _ in range(steps):
        adversarial.requires_grad_(True)
        output, decision = model.forward_with_routing(adversarial)
        prediction_loss = F.cross_entropy(output, labels)
        routing_loss = _route_margin(decision.scores, original_route).mean()
        if objective == "prediction":
            loss = prediction_loss
        elif objective == "route":
            loss = routing_loss
        elif objective == "combined":
            loss = prediction_loss + routing_loss
        else:
            raise ValueError(f"unknown PGD objective {objective}")
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = adversarial.detach() + step_size * gradient.sign()
        adversarial = torch.maximum(torch.minimum(adversarial, upper), lower)
    return adversarial.detach()


@torch.no_grad()
def _flip_counts(model, clean, adversarial) -> dict[str, int]:
    clean_output, clean_route = model.forward_with_routing(clean)
    adv_output, adv_route = model.forward_with_routing(adversarial)
    route_flip = clean_route.indices[:, 0] != adv_route.indices[:, 0]
    prediction_flip = clean_output.argmax(-1) != adv_output.argmax(-1)
    return {
        "samples": clean.shape[0],
        "route_flip": int(route_flip.sum().item()),
        "prediction_flip": int(prediction_flip.sum().item()),
        "both": int((route_flip & prediction_flip).sum().item()),
        "route_only": int((route_flip & ~prediction_flip).sum().item()),
        "prediction_only": int((~route_flip & prediction_flip).sum().item()),
    }


def run(args) -> dict[str, float]:
    device = _device(args.device)
    model, payload = load_output_moe_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()
    dataset = _load_dataset(payload["dataset"], False, download=args.download)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = Subset(dataset, range(args.max_samples))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    totals = {
        key: 0
        for key in ("samples", "route_flip", "prediction_flip", "both", "route_only", "prediction_only")
    }
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        adversarial = pgd(
            model,
            inputs,
            labels,
            epsilon=args.epsilon,
            steps=args.steps,
            step_size=args.step_size,
            objective=args.objective,
        )
        counts = _flip_counts(model, inputs, adversarial)
        for key, value in counts.items():
            totals[key] += value
    denominator = max(totals["samples"], 1)
    result = {
        "samples": totals["samples"],
        "route_flip_rate": totals["route_flip"] / denominator,
        "prediction_flip_rate": totals["prediction_flip"] / denominator,
        "both_rate": totals["both"] / denominator,
        "route_only_rate": totals["route_only"] / denominator,
        "prediction_only_rate": totals["prediction_only"] / denominator,
    }
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure route flips versus prediction flips")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--objective", choices=("prediction", "route", "combined"), default="combined")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
