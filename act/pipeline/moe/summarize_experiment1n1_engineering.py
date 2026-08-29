"""Independently summarize paired N1 engineering outcomes for publication."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

from scipy.stats import binomtest

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json


SOLVED = {"SAFE", "UNSAFE"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def paired_solved_table(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        (
            row["baseline_status"] in SOLVED,
            row["status"] in SOLVED,
        )
        for row in rows
    )
    n00 = counts[(False, False)]
    n01 = counts[(False, True)]
    n10 = counts[(True, False)]
    n11 = counts[(True, True)]
    discordant = n01 + n10
    return {
        "baseline_unsolved_n1_unsolved_n00": n00,
        "baseline_unsolved_n1_solved_n01": n01,
        "baseline_solved_n1_unsolved_n10": n10,
        "baseline_solved_n1_solved_n11": n11,
        "net_solved_gain": n01 - n10,
        "exact_two_sided_mcnemar_p": (
            float(binomtest(n01, discordant, 0.5).pvalue) if discordant else 1.0
        ),
        "test": "exact paired binomial/McNemar on discordant solved indicators",
    }


def conditioned_support_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    properties = segments = tightened_properties = tightened_segments = 0
    for row in rows:
        for pair in (row.get("n1") or {}).get("pairs", []):
            for prop in pair.get("property_rows", []):
                if prop.get("reused_parent") or not prop.get("segments"):
                    continue
                unconditional = prop.get("unconditional_difference_bounds")
                if not unconditional or len(unconditional) != 2:
                    continue
                properties += 1
                lower, upper = map(float, unconditional)
                property_tightened = False
                for segment in prop["segments"]:
                    bounds = segment.get("difference_bounds")
                    if not bounds or len(bounds) != 2:
                        continue
                    segments += 1
                    seg_lower, seg_upper = map(float, bounds)
                    tightened = seg_lower > lower + 1e-12 or seg_upper < upper - 1e-12
                    tightened_segments += int(tightened)
                    property_tightened = property_tightened or tightened
                tightened_properties += int(property_tightened)
    return {
        "segmented_properties_with_recorded_unconditional_range": properties,
        "active_segments_with_recorded_range": segments,
        "properties_with_any_strict_difference_tightening": tightened_properties,
        "segments_with_strict_difference_tightening": tightened_segments,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    table = paired_solved_table(rows)
    baseline_solved = sum(row["baseline_status"] in SOLVED for row in rows)
    n1_solved = sum(row["status"] in SOLVED for row in rows)
    time_differences = [
        float(row["total_seconds"]) - float(row["baseline_seconds"])
        for row in rows
        if row.get("baseline_seconds") is not None
    ]
    time_ratios = [
        float(row["total_seconds"]) / float(row["baseline_seconds"])
        for row in rows
        if float(row.get("baseline_seconds") or 0.0) > 0.0
    ]
    return {
        "scope": "engineering_performance_rerun_not_confirmatory_overwrite",
        "rows": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "reason_counts": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "transition_counts": dict(
            sorted(Counter(row["paired_transition"] for row in rows).items())
        ),
        "baseline_solved": baseline_solved,
        "n1_solved": n1_solved,
        "baseline_solved_rate": baseline_solved / len(rows) if rows else None,
        "n1_solved_rate": n1_solved / len(rows) if rows else None,
        "paired_solved_table": table,
        "new_safe_ranks": [
            int(row["sample_rank"])
            for row in rows
            if row["baseline_status"] not in SOLVED and row["status"] == "SAFE"
        ],
        "new_full_forward_unsafe_ranks": [
            int(row["sample_rank"])
            for row in rows
            if row["baseline_status"] not in SOLVED and row["status"] == "UNSAFE"
        ],
        "all_unsafe_full_forward_validated": all(
            row["status"] != "UNSAFE" or row.get("full_model_witness_valid")
            for row in rows
        ),
        "runtime": {
            "paired_difference_median_seconds": statistics.median(time_differences)
            if time_differences
            else None,
            "paired_ratio_median": statistics.median(time_ratios) if time_ratios else None,
            "interpretation": "descriptive engineering runtime; not a confirmatory endpoint",
        },
        "conditioned_support": conditioned_support_counts(rows),
        "segmented_property_rows": sum(
            int(row.get("segmented_property_rows", 0)) for row in rows
        ),
        "active_segments": sum(int(row.get("active_segments", 0)) for row in rows),
        "original_confirmatory_overall_solved_rate_immutable": 0.56,
    }


def publish(result_dir: Path, audit_path: Path, output_path: Path) -> dict[str, Any]:
    result_dir = _inside(result_dir, WRITE_ROOT)
    audit_path = _inside(audit_path, WRITE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite published result {output_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("issue_count") != 0:
        raise RuntimeError("N1 result cannot be published with audit issues")
    rows = _jsonl(result_dir / "results.jsonl")
    result = {
        "schema_version": 1,
        "result": summarize(rows),
        "independent_audit": {
            "issue_count": audit["issue_count"],
            "unsafe_replayed": audit["unsafe_replayed"],
            "sha256": _sha256(audit_path),
        },
        "artifact_sha256": {
            "runtime_config": _sha256(result_dir / "config.json"),
            "results_jsonl": _sha256(result_dir / "results.jsonl"),
            "results_csv": _sha256(result_dir / "results.csv"),
            "runner_summary": _sha256(result_dir / "summary.json"),
        },
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.result_dir, args.audit, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
