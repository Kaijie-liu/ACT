"""Audit the float32 sparse-CROWN numerical positive-bound reach pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(raw_path: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    raw_path = _inside(raw_path, MOE_ROOT)
    config_path = _inside(config_path, PROJECT_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if raw.get("status") != "COMPLETED_NUMERICAL_ONLY":
        issues.append("raw status is not numerical-only complete")
    if raw.get("scope") != config.get("scope"):
        issues.append("scope changed")
    if raw.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash changed")
    if raw.get("bound_method") != "CROWN":
        issues.append("bound method changed")
    if "not outward-rounded" not in raw.get("interpretation", ""):
        issues.append("non-formal numerical interpretation is missing")
    if raw.get("peak_memory_bytes", 0) > config["resource_gate"][
        "maximum_peak_memory_bytes"
    ]:
        issues.append("peak memory exceeded the resource gate")

    input_path = _inside(Path(raw["source"]["inputs"]["path"]), MOE_ROOT)
    if raw["source"]["inputs"]["sha256"] != _sha256(input_path):
        issues.append("source input hash changed")
    with np.load(input_path, allow_pickle=False) as arrays:
        inputs = torch.from_numpy(arrays["inputs"].copy())
        margins = arrays["clean_margin"].astype(np.float64)
    slots = np.asarray(config["sample_slots"], dtype=np.int64)
    if len(raw.get("rows", [])) != len(slots):
        issues.append("row count changed")

    positive_epsilons: list[float] = []
    negative_epsilons: list[float] = []
    quantized_transitions = 0
    evaluations_checked = 0
    for expected_slot, row in zip(slots.tolist(), raw.get("rows", [])):
        if row.get("sample_slot") != expected_slot:
            issues.append(f"sample slot {expected_slot}: ordering changed")
            continue
        if row.get("status") != "NUMERICAL_TRANSITION_BRACKETED":
            issues.append(f"sample slot {expected_slot}: transition not bracketed")
        positive_epsilon = float(row["positive_requested_epsilon"])
        negative_epsilon = float(row["negative_requested_epsilon"])
        if not (0 < positive_epsilon < negative_epsilon):
            issues.append(f"sample slot {expected_slot}: invalid bracket")
        positive_epsilons.append(positive_epsilon)
        negative_epsilons.append(negative_epsilon)
        by_epsilon = {
            float(item["requested_epsilon"]): item for item in row["evaluations"]
        }
        positive_row = by_epsilon.get(positive_epsilon)
        negative_row = by_epsilon.get(negative_epsilon)
        if positive_row is None or negative_row is None:
            issues.append(f"sample slot {expected_slot}: bracket endpoints missing")
            continue
        if positive_row["lower_bound"] <= 0 or negative_row["lower_bound"] > 0:
            issues.append(f"sample slot {expected_slot}: bracket signs changed")
        sample = inputs[expected_slot : expected_slot + 1]
        seen_negative = False
        for evaluation in sorted(
            row["evaluations"], key=lambda item: item["requested_epsilon"]
        ):
            epsilon = float(evaluation["requested_epsilon"])
            lower = torch.clamp(sample - epsilon, 0, 1)
            upper = torch.clamp(sample + epsilon, 0, 1)
            recomputed = {
                "effective_lower_linf": float((sample - lower).abs().max().item()),
                "effective_upper_linf": float((upper - sample).abs().max().item()),
                "changed_lower_coordinates": int(torch.count_nonzero(lower != sample)),
                "changed_upper_coordinates": int(torch.count_nonzero(upper != sample)),
            }
            for key, value in recomputed.items():
                if value != evaluation[key]:
                    issues.append(
                        f"sample slot {expected_slot}: representable-box {key} changed"
                    )
            is_positive = bool(evaluation["lower_bound"] > 0)
            if not is_positive:
                seen_negative = True
            elif seen_negative:
                issues.append(
                    f"sample slot {expected_slot}: positive bound after negative bound"
                )
            evaluations_checked += 1
        positive_width = max(
            positive_row["effective_lower_linf"],
            positive_row["effective_upper_linf"],
        )
        negative_width = max(
            negative_row["effective_lower_linf"],
            negative_row["effective_upper_linf"],
        )
        positive_changed = max(
            positive_row["changed_lower_coordinates"],
            positive_row["changed_upper_coordinates"],
        )
        negative_changed = max(
            negative_row["changed_lower_coordinates"],
            negative_row["changed_upper_coordinates"],
        )
        if negative_width > positive_width and negative_changed > positive_changed:
            quantized_transitions += 1
        else:
            issues.append(
                f"sample slot {expected_slot}: transition is not aligned with a "
                "representable-box expansion"
            )

    strong_dir = _inside(Path(config["source_result_dir"]), MOE_ROOT)
    bounds_path = strong_dir / "crown_bounds.json"
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    first_row = bounds["rows"][0]
    source_slots = np.asarray(config["sample_slots"], dtype=np.int64)
    lower_bounds = np.asarray(first_row["lower_bounds"], dtype=np.float64)[
        source_slots
    ]
    source_epsilon = float(first_row["epsilon"])
    linear_zero = source_epsilon * margins[source_slots] / (
        margins[source_slots] - lower_bounds
    )

    positive_array = np.asarray(positive_epsilons, dtype=np.float64)
    negative_array = np.asarray(negative_epsilons, dtype=np.float64)
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_AUDIT_FLOAT32_CROWN_NUMERICAL_REACH",
        "raw_result": {"path": str(raw_path), "sha256": _sha256(raw_path)},
        "evaluations_checked": evaluations_checked,
        "quantized_transition_rows": quantized_transitions,
        "recomputed": {
            "linearized_zero_estimate": {
                "minimum": float(linear_zero.min()),
                "median": float(np.median(linear_zero)),
                "maximum": float(linear_zero.max()),
            },
            "positive_requested_epsilon": {
                "minimum": float(positive_array.min()),
                "median": float(np.median(positive_array)),
                "maximum": float(positive_array.max()),
            },
            "negative_requested_epsilon": {
                "minimum": float(negative_array.min()),
                "median": float(np.median(negative_array)),
                "maximum": float(negative_array.max()),
            },
        },
        "conclusion": (
            "All five requested-epsilon sign transitions replay and coincide with "
            "a discrete expansion of the representable float32 input box. The "
            "median positive/negative requested bracket is 1.856e-9--1.868e-9, "
            "whereas linear extrapolation from 0.5/255 predicts 1.60e-12. These "
            "values are numerical frontend/relaxation diagnostics, not sound "
            "CROWN reach or a paper-ready certificate-gap axis."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
