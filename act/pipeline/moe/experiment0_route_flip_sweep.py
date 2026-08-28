"""Run the Experiment 0 CIFAR-10 route-flip sweep and write CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch

from act.pipeline.moe.route_flips import run


EPSILONS = (0.007843137, 0.015686275, 0.031372549)
OBJECTIVES = ("route", "prediction", "combined")
FIELDNAMES = (
    "epsilon",
    "objective",
    "step_size",
    "samples",
    "route_flip_rate",
    "prediction_flip_rate",
    "both_rate",
    "route_only_rate",
    "prediction_only_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        stream.flush()
        for epsilon in EPSILONS:
            for objective in OBJECTIVES:
                torch.manual_seed(args.seed)
                result = run(
                    SimpleNamespace(
                        checkpoint=args.checkpoint,
                        objective=objective,
                        epsilon=epsilon,
                        steps=20,
                        step_size=epsilon / 4.0,
                        batch_size=128,
                        max_samples=1000,
                        device=args.device,
                        download=False,
                    )
                )
                writer.writerow(
                    {
                        "epsilon": epsilon,
                        "objective": objective,
                        "step_size": epsilon / 4.0,
                        **result,
                    }
                )
                stream.flush()
                print(
                    f"sweep_complete epsilon={epsilon} objective={objective} "
                    f"csv={output}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
