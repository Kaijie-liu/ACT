"""Independent structural and pairing audit for lazy top-k scaling."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path = _inside(path, PROJECT_ROOT)
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_root = _inside(Path(config["output_dir"]), WRITE_ROOT)
    summary_path = result_root / "summary.json"
    rows_path = result_root / "rows.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    issues: list[str] = []
    expected_rows = len(config["num_experts"]) * len(config["families"]) * 2
    if len(rows) != expected_rows or summary.get("rows") != expected_rows:
        issues.append("scaling row count differs from the frozen Cartesian product")
    if _sha256(rows_path) != summary.get("rows_jsonl", {}).get("sha256"):
        issues.append("rows JSONL hash differs from the runner summary")
    keys = Counter(
        (int(row["experts"]), str(row["family"]), bool(row["submit_partial_mip_starts"]))
        for row in rows
    )
    if any(value != 1 for value in keys.values()) or len(keys) != expected_rows:
        issues.append("paired condition keys are missing or duplicated")
    complete_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    ratios: list[float] = []
    transitions: list[dict[str, Any]] = []
    for experts in map(int, config["num_experts"]):
        for family in config["families"]:
            pair = [
                row for row in rows if int(row["experts"]) == experts and row["family"] == family
            ]
            if len(pair) != 2:
                continue
            by_start = {bool(row["submit_partial_mip_starts"]): row for row in pair}
            if set(by_start) != {False, True}:
                issues.append(f"missing start/no-start pair for E={experts}, {family}")
                continue
            cold, warm = by_start[False], by_start[True]
            if cold["router_weight_sha256"] != warm["router_weight_sha256"]:
                issues.append(f"paired router weights differ for E={experts}, {family}")
            if cold["router_bias_sha256"] != warm["router_bias_sha256"]:
                issues.append(f"paired router biases differ for E={experts}, {family}")
            if cold["telemetry"]["partial_mip_start_attempts"] != 0:
                issues.append(f"no-start condition submitted a MIP start for E={experts}, {family}")
            if warm["telemetry"]["partial_mip_start_attempts"] != warm["no_good_cuts"]:
                issues.append(f"start attempts do not equal reusable incumbents for E={experts}, {family}")
            if warm["telemetry"]["partial_mip_starts_accepted"] > warm["telemetry"]["partial_mip_start_attempts"]:
                issues.append(f"accepted MIP starts exceed attempts for E={experts}, {family}")
            if warm["telemetry"].get("partial_mip_start_internal_use_claimed") is not False:
                issues.append(f"runner overclaims MIP-start internal use for E={experts}, {family}")
            for row in pair:
                status_counts[str(row["status"])] += 1
                complete_counts[str(bool(row["complete"]))] += 1
                if row["complete"] and int(row["solves"]) != int(row["route_set_count"]) + 1:
                    issues.append(f"complete solve accounting failed for E={experts}, {family}")
                if row["complete"] and family == "all_tied_worst_case":
                    expected = math.comb(experts, int(config["top_k"]))
                    if int(row["route_set_count"]) != expected:
                        issues.append(f"all-tied set count failed for E={experts}")
                if row["complete"] and family == "strictly_stable" and int(row["route_set_count"]) != 1:
                    issues.append(f"strictly stable family has !=1 set for E={experts}")
                if experts <= int(config["exhaustive_differential_max_experts"]):
                    if not row["complete"] or row["route_sets"] != row["exhaustive_sets"]:
                        issues.append(f"small-E exhaustive differential failed for E={experts}, {family}")
            sets_equal = cold["route_sets"] == warm["route_sets"]
            if cold["complete"] and warm["complete"] and not sets_equal:
                issues.append(f"complete paired sets differ for E={experts}, {family}")
            ratio = float(warm["elapsed_seconds"]) / float(cold["elapsed_seconds"])
            ratios.append(ratio)
            transitions.append(
                {
                    "experts": experts,
                    "family": family,
                    "cold_status": cold["status"],
                    "warm_status": warm["status"],
                    "sets_equal": sets_equal,
                    "no_start_seconds": cold["elapsed_seconds"],
                    "with_start_seconds": warm["elapsed_seconds"],
                    "with_over_without_ratio": ratio,
                }
            )
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": "INDEPENDENT_LAZY_TOPK_SCALING_AND_MIP_START_AUDIT_R1",
        "issue_count": len(issues),
        "issues": issues,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "raw": {
            "summary": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "rows": str(rows_path),
            "rows_sha256": _sha256(rows_path),
        },
        "execution": {
            "rows": len(rows),
            "complete_counts": dict(complete_counts),
            "status_counts": dict(status_counts),
            "paired_conditions": transitions,
        },
        "paired_timing": {
            "median_with_over_without_ratio": statistics.median(ratios) if ratios else None,
            "minimum_ratio": min(ratios) if ratios else None,
            "maximum_ratio": max(ratios) if ratios else None,
            "claim": "measured effect only; solver-internal use is unobservable and not claimed",
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write(output, result)
    if issues:
        raise RuntimeError(f"lazy top-k scaling audit failed: {issues}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
