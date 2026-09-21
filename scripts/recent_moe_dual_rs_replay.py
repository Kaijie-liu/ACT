"""Independent recount of author saved radii; no fresh randomized certification.

Uses the exact author strict radius > threshold convention, including at zero.
NPY rows have no embedded input IDs: only positional alignment can be checked.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def percentages(radii, thresholds):
    return [float(np.mean(np.asarray(radii) > r) * 100) for r in thresholds]


def composed(data, router, experts, num_experts):
    with (data / router).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expert = np.load(data / experts, allow_pickle=False)
    choice = np.array([int(r["predict"]) for r in rows])
    radius = np.array([float(r["radius"]) for r in rows])
    if expert.ndim != 2 or expert.shape[0] != len(rows) or expert.shape[1] < num_experts:
        raise ValueError("router/expert row alignment mismatch")
    # Author NPYs may include oracle/label columns after the expert columns.
    expert = expert[:, :num_experts]
    if not np.all((choice == -1) | ((choice >= 0) & (choice < num_experts))):
        raise ValueError("invalid expert choice")
    if not np.isfinite(expert).all() or not np.isfinite(radius).all():
        raise ValueError("nonfinite author radii")
    radius[choice == -1] = 0
    final = np.minimum(radius, expert[np.arange(len(rows)), np.maximum(choice, 0)])
    return final, expert


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    a = p.parse_args()
    d = a.repo / "reproduce/data"
    thresholds = [i / 4 for i in range(11)]
    cases = {
        "cifar10_off_the_shelf": (
            "sigma_est/100map/num_2/softce_con_lbd40.0_eta0.5/class_weights/cifar_resnet110/round0/noise_1.0.tsv",
            "sigma_label/base/0.250_0.500_1.000_test.npy"),
        "cifar10_finetuned": (
            "sigma_est/100map/num_2/softce_con_lbd40.0_eta0.5/class_weights/cifar_resnet110/round1/noise_1.0.tsv",
            "sigma_label/100map/0.250_0.500_1.000_test.npy"),
        "fig4_weak_experts": ("sigma_est/100map/router/worse/noise_1.0.tsv",
                              "sigma_label/base/0.250_1.000_test.npy"),
        "fig4_strong_experts": ("sigma_est/100map/router/better/noise_1.0.tsv",
                                "sigma_label/mix/0.250_1.000_test.npy"),
        "imagenet_finetuned": ("sigma_est/imagenet/0.500_1.000.tsv",
                                "sigma_label/imagenet/0.500_1.000_test_ft.npy"),
    }
    results = {}
    for name, (router, experts) in cases.items():
        num_experts = 3 if name.startswith("cifar10") else 2
        final, expert = composed(d, router, experts, num_experts)
        grid = [0, .5, 1, 1.5, 2] if name.startswith("imagenet") else thresholds
        results[name] = {"n": len(final), "radii_l2": grid,
                         "percent": percentages(final, grid),
                         "expert_percent": [percentages(expert[:, i], grid)
                                            for i in range(expert.shape[1])],
                         "router_file": router, "expert_file": experts}
    expected = {
        "cifar10_off_the_shelf": [68.34,55.25,41.28,29.01,19.85,12.73,7.62,4.73,3.54,2.62,1.83],
        "cifar10_finetuned": [70.53,57.48,45.27,34.15,24.68,17.84,12.46,8.83,6.65,4.73,3.14],
        "imagenet_finetuned": [74,60.6,48,33.6,17],
    }
    for name, values in expected.items():
        results[name]["paper_v3_percent"] = values
        results[name]["paper_rounded_match"] = [round(v, 2) for v in results[name]["percent"]] == values
    print(json.dumps({"grade": "AUTHOR_RESULT_REPLAY", "fresh_model_calls": 0,
                      "alignment": "author positional rows; no independent NPY input identity",
                      "results": results}, indent=2))
    if not all(results[name]["paper_rounded_match"] for name in expected):
        raise SystemExit("author result / paper mismatch retained; no threshold change")


if __name__ == "__main__":
    main()
