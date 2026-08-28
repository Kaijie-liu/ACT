# ===- act/pipeline/moe/route_flips.py - Route/Prediction Flip Study ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.train import _device, _load_dataset


SUMMARY_FIELDS = (
    "samples",
    "top1_flip_rate",
    "topk_set_flip_rate",
    "prediction_flip_rate",
    "both_rate",
    "route_only_rate",
    "prediction_only_rate",
    "topk_set_only_rate",
    "mean_changed_memberships",
    "mean_topk_jaccard",
)
RESULT_FIELDNAMES = (
    "samples",
    "route_flip_rate",
    *SUMMARY_FIELDS[1:],
    *(f"all_{field}" for field in SUMMARY_FIELDS),
    *(f"clean_correct_{field}" for field in SUMMARY_FIELDS),
)


def _route_margin(scores: torch.Tensor, original_route: torch.Tensor) -> torch.Tensor:
    """Margin for replacing at least one member of the original top-k set."""
    if original_route.ndim == 1:
        original_route = original_route[:, None]
    if original_route.shape[1] >= scores.shape[1]:
        return scores.new_zeros(scores.shape[0])
    selected_min = scores.gather(1, original_route).min(dim=1).values
    selected_mask = torch.zeros_like(scores, dtype=torch.bool)
    selected_mask.scatter_(1, original_route, True)
    outside_max = scores.masked_fill(selected_mask, float("-inf")).max(dim=1).values
    return outside_max - selected_min


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
        original_route = model.route(inputs).indices
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


def _membership_mask(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    mask = torch.zeros(
        (indices.shape[0], num_experts), dtype=torch.bool, device=indices.device
    )
    return mask.scatter_(1, indices, True)


def _group_totals(
    *,
    mask: torch.Tensor,
    top1_flip: torch.Tensor,
    topk_set_flip: torch.Tensor,
    prediction_flip: torch.Tensor,
    changed_memberships: torch.Tensor,
    topk_jaccard: torch.Tensor,
) -> dict[str, float | int]:
    return {
        "samples": int(mask.sum().item()),
        "top1_flip": int((top1_flip & mask).sum().item()),
        "topk_set_flip": int((topk_set_flip & mask).sum().item()),
        "prediction_flip": int((prediction_flip & mask).sum().item()),
        "both": int((top1_flip & prediction_flip & mask).sum().item()),
        "route_only": int((top1_flip & ~prediction_flip & mask).sum().item()),
        "prediction_only": int((~top1_flip & prediction_flip & mask).sum().item()),
        "topk_set_only": int(
            (topk_set_flip & ~prediction_flip & mask).sum().item()
        ),
        "changed_memberships": float(changed_memberships[mask].sum().item()),
        "topk_jaccard": float(topk_jaccard[mask].sum().item()),
    }


@torch.no_grad()
def _flip_totals(model, clean, labels, adversarial) -> dict[str, dict[str, float | int]]:
    clean_output, clean_route = model.forward_with_routing(clean)
    adv_output, adv_route = model.forward_with_routing(adversarial)
    top1_flip = clean_route.indices[:, 0] != adv_route.indices[:, 0]
    prediction_flip = clean_output.argmax(-1) != adv_output.argmax(-1)
    clean_membership = _membership_mask(clean_route.indices, model.spec.num_experts)
    adv_membership = _membership_mask(adv_route.indices, model.spec.num_experts)
    changed_memberships = (clean_membership ^ adv_membership).sum(dim=1)
    intersection = (clean_membership & adv_membership).sum(dim=1).float()
    union = (clean_membership | adv_membership).sum(dim=1).float()
    topk_jaccard = intersection / union.clamp_min(1.0)
    topk_set_flip = changed_memberships > 0
    clean_correct = clean_output.argmax(-1) == labels
    all_samples = torch.ones_like(clean_correct)
    return {
        "all": _group_totals(
            mask=all_samples,
            top1_flip=top1_flip,
            topk_set_flip=topk_set_flip,
            prediction_flip=prediction_flip,
            changed_memberships=changed_memberships,
            topk_jaccard=topk_jaccard,
        ),
        "clean_correct": _group_totals(
            mask=clean_correct,
            top1_flip=top1_flip,
            topk_set_flip=topk_set_flip,
            prediction_flip=prediction_flip,
            changed_memberships=changed_memberships,
            topk_jaccard=topk_jaccard,
        ),
    }


def _empty_totals() -> dict[str, float | int]:
    return {
        key: 0
        for key in (
            "samples",
            "top1_flip",
            "topk_set_flip",
            "prediction_flip",
            "both",
            "route_only",
            "prediction_only",
            "topk_set_only",
            "changed_memberships",
            "topk_jaccard",
        )
    }


def _summarize(totals: dict[str, float | int]) -> dict[str, float | int]:
    denominator = max(int(totals["samples"]), 1)
    return {
        "samples": int(totals["samples"]),
        "top1_flip_rate": totals["top1_flip"] / denominator,
        "topk_set_flip_rate": totals["topk_set_flip"] / denominator,
        "prediction_flip_rate": totals["prediction_flip"] / denominator,
        "both_rate": totals["both"] / denominator,
        "route_only_rate": totals["route_only"] / denominator,
        "prediction_only_rate": totals["prediction_only"] / denominator,
        "topk_set_only_rate": totals["topk_set_only"] / denominator,
        "mean_changed_memberships": totals["changed_memberships"] / denominator,
        "mean_topk_jaccard": totals["topk_jaccard"] / denominator,
    }


def run(args) -> dict[str, Any]:
    device = _device(args.device)
    model, payload = load_output_moe_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()
    dataset = _load_dataset(payload["dataset"], False, download=args.download)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = Subset(dataset, range(args.max_samples))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    totals = {"all": _empty_totals(), "clean_correct": _empty_totals()}
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
        batch_totals = _flip_totals(model, inputs, labels, adversarial)
        for group in totals:
            for key, value in batch_totals[group].items():
                totals[group][key] += value
    summaries = {group: _summarize(values) for group, values in totals.items()}
    all_summary = summaries["all"]
    result = {
        "samples": all_summary["samples"],
        "route_flip_rate": all_summary["top1_flip_rate"],
        **{key: value for key, value in all_summary.items() if key != "samples"},
        **{
            f"{group}_{key}": value
            for group, summary in summaries.items()
            for key, value in summary.items()
        },
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
