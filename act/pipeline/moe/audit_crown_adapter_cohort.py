# ===- audit_crown_adapter_cohort.py - Independent P0b audit ------====#
"""Independently audit and summarize a completed P0b adapter cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from act.pipeline.moe.crown_adapter_cohort import (
    RESULTS_ROOT,
    VARIANTS,
    independently_summarize,
)
from act.pipeline.moe.experiment1 import PROJECT_ROOT, _inside, _sha256, _write_json


DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/crown_adapter_cohort_bal010_43_r2.json"
)


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], *, comparison_tolerance: float = 1e-6
) -> dict[str, Any]:
    """Compute expert-level coverage and paired CROWN bound comparisons."""

    valid = [row for row in rows if not row.get("error")]
    expert_status = {variant: Counter() for variant in VARIANTS}
    pair_status = {variant: Counter() for variant in VARIANTS}
    hz_completeness: Counter[str] = Counter()
    property_differences: list[float] = []
    expert_minimum_differences: list[float] = []
    guarded_better_experts = guarded_equal_experts = guarded_worse_experts = 0
    for row in valid:
        for variant in VARIANTS:
            pair_status[variant][row["variants"][variant]["status"]] += 1
        for expert in row["experts"]:
            for variant in VARIANTS:
                expert_status[variant][expert[variant]["status"]] += 1
            hz = expert["hz_retained_guard"]
            hz_completeness[
                "complete_exact"
                if hz.get("complete") and hz.get("exact")
                else "incomplete"
            ] += 1
            guarded = np.asarray(
                expert["crown_guarded_box"].get("lower_bounds", []),
                dtype=np.float64,
            )
            original = np.asarray(
                expert["crown_original_box"].get("lower_bounds", []),
                dtype=np.float64,
            )
            if guarded.shape != original.shape or guarded.size == 0:
                continue
            differences = guarded - original
            property_differences.extend(float(value) for value in differences)
            minimum_difference = float(guarded.min() - original.min())
            expert_minimum_differences.append(minimum_difference)
            if minimum_difference > comparison_tolerance:
                guarded_better_experts += 1
            elif minimum_difference < -comparison_tolerance:
                guarded_worse_experts += 1
            else:
                guarded_equal_experts += 1
    differences_array = np.asarray(property_differences, dtype=np.float64)
    minimum_array = np.asarray(expert_minimum_differences, dtype=np.float64)
    return {
        "valid_branches": len(valid),
        "valid_expert_obligations": sum(
            len(row.get("experts", [])) for row in valid
        ),
        "pair_status_counts": {
            variant: dict(sorted(counter.items()))
            for variant, counter in pair_status.items()
        },
        "expert_status_counts": {
            variant: dict(sorted(counter.items()))
            for variant, counter in expert_status.items()
        },
        "hz_expert_completeness": dict(sorted(hz_completeness.items())),
        "guarded_vs_original_crown": {
            "comparison_tolerance": float(comparison_tolerance),
            "property_rows_compared": int(differences_array.size),
            "expert_obligations_compared": int(minimum_array.size),
            "guarded_minimum_strictly_better_experts": guarded_better_experts,
            "guarded_minimum_equal_experts": guarded_equal_experts,
            "guarded_minimum_strictly_worse_experts": guarded_worse_experts,
            "property_lower_bound_difference_median": (
                float(np.median(differences_array))
                if differences_array.size
                else None
            ),
            "property_lower_bound_difference_minimum": (
                float(differences_array.min()) if differences_array.size else None
            ),
            "property_lower_bound_difference_maximum": (
                float(differences_array.max()) if differences_array.size else None
            ),
        },
        "interpretation_limits": {
            "crown_positive_filter_outward_rounded": False,
            "negative_relaxation_is_unsafe": False,
            "certificate_ordering_predeclared": False,
            "runtime_speedup_claimed": False,
        },
    }


def run(config_path: Path, raw_sha256: str, output_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_dir = _inside(Path(config["output_dir"]), RESULTS_ROOT)
    rows_path = result_dir / "branches.jsonl"
    if _sha256(rows_path) != raw_sha256:
        raise RuntimeError("completed P0b JSONL differs from frozen audit input")
    internal = independently_summarize(result_dir, config)
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
    ]
    branch_ids = [row.get("branch_id") for row in rows]
    issues: list[str] = []
    if len(rows) != int(config["expected_branches"]):
        issues.append("row_count")
    if len(set(branch_ids)) != len(branch_ids):
        issues.append("duplicate_branch")
    if not internal.get("passed"):
        issues.append("runner_independent_summary_failed")
    if any(row.get("unsafe_claimed") for row in rows):
        issues.append("unsafe_without_full_replay")
    detailed = summarize_rows(rows)
    result = {
        "schema_version": 1,
        "scope": "independent_adapter_consistency_audit",
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "raw_result_jsonl": str(rows_path),
        "raw_result_jsonl_sha256": raw_sha256,
        "runner_independent_summary_sha256": _sha256(
            result_dir / "independent_summary.json"
        ),
        "structural_issues": issues,
        "issue_count": len(issues),
        "passed": not issues,
        "detailed_summary": detailed,
    }
    output_path = _inside(output_path, RESULTS_ROOT)
    if output_path.exists():
        raise RuntimeError(f"independent audit refuses to overwrite {output_path}")
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.config, arguments.raw_sha256, arguments.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
