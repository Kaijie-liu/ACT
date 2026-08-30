"""Independently audit the frozen E=8 lazy top-k correctness artifact."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path

from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(config_path: Path, raw_path: Path, output_path: Path) -> dict[str, object]:
    config_path = _inside(config_path, PROJECT_ROOT)
    raw_path = _inside(raw_path, MOE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if raw.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw status changed")
    if raw.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash changed")
    experts = int(config["num_experts"])
    top_k = int(config["top_k"])
    expected = [list(value) for value in combinations(range(experts), top_k)]
    enumeration = raw.get("enumeration", {})
    if enumeration.get("route_sets") != expected:
        issues.append("lazy route sets differ from the closed-form all-tie oracle")
    if enumeration.get("exhaustive_set_count") != len(expected):
        issues.append("exhaustive set count changed")
    if enumeration.get("lazy_set_count") != len(expected):
        issues.append("lazy set count changed")
    if enumeration.get("lazy_complete") is not True:
        issues.append("lazy enumeration was not proved complete")
    if enumeration.get("lazy_no_good_cuts") != len(expected):
        issues.append("no-good-cut count does not equal enumerated set count")
    if enumeration.get("lazy_solves") != len(expected) + 1:
        issues.append("lazy solve count is not one per set plus infeasibility proof")
    if enumeration.get("lazy_model_builds") != 1:
        issues.append("lazy enumeration rebuilt the base model")
    telemetry = enumeration.get("telemetry", {})
    if telemetry.get("partial_mip_start_attempts") != len(expected):
        issues.append("partial MIP start was not attempted after every solution")
    if telemetry.get("partial_mip_starts_accepted") != len(expected):
        issues.append("HiGHS did not accept every submitted partial MIP start")
    if telemetry.get("partial_mip_start_internal_use_claimed") is not False:
        issues.append("artifact overclaims solver-internal MIP-start use")
    big_m = raw.get("big_m", {})
    if big_m.get("fast_selection_binaries") != 2:
        issues.append("fast big-M control no longer allocates two binaries")
    if big_m.get("exact_selection_binaries") != 0:
        issues.append("exact-support big-M failed to eliminate control binaries")
    if big_m.get("exact_support_complete") is not True:
        issues.append("exact-support control did not complete")

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "config_sha256": _sha256(config_path),
        "raw_sha256": _sha256(raw_path),
        "closed_form_expected_sets": len(expected),
        "lazy_sets": enumeration.get("lazy_set_count"),
        "lazy_solves": enumeration.get("lazy_solves"),
        "model_builds": enumeration.get("lazy_model_builds"),
        "partial_mip_start_attempts": telemetry.get("partial_mip_start_attempts"),
        "partial_mip_starts_accepted": telemetry.get(
            "partial_mip_starts_accepted"
        ),
        "partial_mip_start_internal_use_claimed": telemetry.get(
            "partial_mip_start_internal_use_claimed"
        ),
        "fast_to_exact_selection_binaries": [
            big_m.get("fast_selection_binaries"),
            big_m.get("exact_selection_binaries"),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.raw, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
