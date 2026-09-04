"""CROWN-environment worker for prepared official RT-ER B3 branches."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from act.util.typing_compat import install_typing_override

install_typing_override()

import numpy as np
import torch

from act.pipeline.moe.crown_adapter_cohort import _crown_bounds
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_b3 import (
    _nvidia_driver_version,
    _package_version_or_unavailable,
    normalize_unit_pixel_box,
)
from act.pipeline.moe.icml2025_route_telemetry import (
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _load_official_model,
)


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def _gpu_free_gib() -> float:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    values = [float(row.strip()) / 1024.0 for row in completed.stdout.splitlines()]
    if len(values) != 1:
        raise RuntimeError("B3 requires exactly one visible GPU")
    return values[0]


def _wait_for_gpu_memory(
    *, minimum_free_gib: float, poll_seconds: float, max_wait_seconds: float
) -> dict[str, Any]:
    started = time.monotonic()
    observations = 0
    minimum_observed = float("inf")
    while True:
        free_gib = _gpu_free_gib()
        observations += 1
        minimum_observed = min(minimum_observed, free_gib)
        elapsed = time.monotonic() - started
        if free_gib >= minimum_free_gib:
            return {
                "status": "PASSED",
                "minimum_free_gib": minimum_free_gib,
                "free_gib_at_release": free_gib,
                "minimum_observed_free_gib": minimum_observed,
                "observations": observations,
                "wait_seconds": elapsed,
            }
        if elapsed >= max_wait_seconds:
            raise RuntimeError(
                f"GPU resource gate timed out with {free_gib:.3f} GiB free; "
                f"required {minimum_free_gib:.3f} GiB"
            )
        time.sleep(poll_seconds)


def _top1_property_rows(prediction: int, classes: int = 10):
    rows = []
    for competitor in range(classes):
        if competitor == int(prediction):
            continue
        row = np.zeros(classes, dtype=np.float64)
        row[int(prediction)] = 1.0
        row[competitor] = -1.0
        rows.append((row, 0.0))
    return tuple(rows)


def run(prepare_path: Path, output_path: Path) -> dict[str, Any]:
    prepare_path = _inside(prepare_path, WRITE_ROOT)
    output_path = _inside(output_path, WRITE_ROOT)
    if output_path.exists():
        raise RuntimeError(f"B3 CROWN worker refuses to overwrite {output_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepare.get("status") != "PREPARED_CROWN_NOT_RUN":
        raise RuntimeError("B3 prepare artifact is not ready for CROWN")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")
    checkpoint = _inside(Path(prepare["checkpoint"]["path"]), WRITE_ROOT)
    if _sha256(checkpoint) != prepare["checkpoint"]["sha256"]:
        raise RuntimeError("prepared checkpoint identity changed")
    artifact = _inside(Path(prepare["artifact"]["path"]), WRITE_ROOT)
    if _sha256(artifact) != prepare["artifact"]["sha256"]:
        raise RuntimeError("prepared branch-box artifact changed")
    with np.load(artifact, allow_pickle=False) as cohort_arrays:
        required = {"dataset_indices", "centers", "lower", "upper", "epsilons"}
        if not required.issubset(cohort_arrays.files):
            raise RuntimeError("prepared cohort artifact is incomplete")
    sys.dont_write_bytecode = True
    model, payload = _load_official_model(checkpoint, torch.device("cpu"))
    if int(payload.get("epoch", -1)) + 1 != int(prepare["checkpoint"]["epoch"]):
        raise RuntimeError("checkpoint epoch changed")
    config = prepare["config"]["value"]
    method = str(config["crown"]["method"])
    device = str(config["crown"]["device"])
    tolerance = float(config["numerical"]["safe_positive_margin"])
    gate_config = config["crown"].get("resource_gate", {})
    resource_gate = _wait_for_gpu_memory(
        minimum_free_gib=float(gate_config.get("minimum_free_gib", 36.0)),
        poll_seconds=float(gate_config.get("poll_seconds", 300.0)),
        max_wait_seconds=float(gate_config.get("max_wait_hours", 24.0)) * 3600.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path = output_path.with_suffix(output_path.suffix + ".rows.jsonl")
    started = time.monotonic()
    branch_results: list[dict[str, Any]] = []
    by_row: dict[str, list[dict[str, Any]]] = {}
    with rows_path.open("x", encoding="utf-8") as rows_handle:
        for branch in prepare["branches"]:
            if branch["feasibility"] == "infeasible":
                continue
            slot = int(branch["sample_slot"])
            expert_index = int(branch["expert"])
            cohort = str(branch.get("cohort", "boundary_adaptive"))
            row_id = str(branch.get("row_id", f"adaptive:{slot}"))
            hull_artifact = _inside(Path(branch["hull_artifact"]), WRITE_ROOT)
            if _sha256(hull_artifact) != branch["hull_artifact_sha256"]:
                raise RuntimeError("prepared guarded-hull artifact changed")
            with np.load(hull_artifact, allow_pickle=False) as hull_arrays:
                lower = torch.from_numpy(hull_arrays["lower"].copy()).unsqueeze(0)
                upper = torch.from_numpy(hull_arrays["upper"].copy()).unsqueeze(0)
            center, lower, upper = normalize_unit_pixel_box(lower, upper)
            prediction = int(prepare["rows"][slot]["hard_prediction"])
            expert = copy.deepcopy(model.experts[expert_index])
            crown = _crown_bounds(
                expert,
                center,
                lower,
                upper,
                property_rows=_top1_property_rows(prediction),
                device=device,
                tolerance=tolerance,
                method=method,
            )
            record = {
                "cohort": cohort,
                "row_id": row_id,
                "sample_slot": slot,
                "dataset_index": int(branch["dataset_index"]),
                "expert": expert_index,
                "feasibility": branch["feasibility"],
                "hull_complete": bool(branch["hull_complete"]),
                "hull_exact": bool(branch["hull_exact"]),
                "crown": crown,
                "formal_status": (
                    "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
                    if crown["status"] == "CERTIFIED_MARGIN_FILTER"
                    else "UNKNOWN"
                ),
            }
            branch_results.append(record)
            by_row.setdefault(row_id, []).append(record)
            _append_json(rows_handle, record)
            del expert
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def aggregate_route_a(prepared_row: dict[str, Any]) -> tuple[str, bool, set[int], set[int]]:
        records = by_row.get(str(prepared_row["row_id"]), [])
        expected = set(int(value) for value in prepared_row["candidate_experts"])
        observed = {int(record["expert"]) for record in records}
        all_filtered = bool(records) and all(
            record["crown"]["status"] == "CERTIFIED_MARGIN_FILTER"
            for record in records
        )
        complete = bool(prepared_row["candidate_set_exact"]) and observed == expected
        route_a = (
            "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
            if complete and all_filtered
            else "UNKNOWN"
        )
        return route_a, complete, expected, observed

    sample_results: list[dict[str, Any]] = []
    for slot, prepared_row in enumerate(prepare["rows"]):
        prepared_row = {"row_id": f"adaptive:{slot}", **prepared_row}
        route_a, complete, expected, observed = aggregate_route_a(prepared_row)
        sample_results.append(
            {
                "sample_slot": slot,
                "dataset_index": int(prepared_row["dataset_index"]),
                "epsilon": float(prepared_row["epsilon"]),
                "candidate_experts": sorted(expected),
                "candidate_set_exact": bool(prepared_row["candidate_set_exact"]),
                "route_invariance_baseline": prepared_row["route_invariance_baseline"],
                "route_a_crown": route_a,
                "route_a_formal_safe": False,
                "reason": (
                    "CROWN positive margins pass the frozen filter but are not "
                    "outward-rounded formal SAFE results"
                    if route_a != "UNKNOWN"
                    else "one or more feasible expert properties remain unresolved"
                ),
            }
        )
    fixed_radius_results: list[dict[str, Any]] = []
    for prepared_row in prepare.get("fixed_radius_rows", []):
        route_a, complete, expected, observed = aggregate_route_a(prepared_row)
        route_status = str(prepared_row["route_status"])
        if route_status == "PROVEN_ROUTE_STABLE":
            clean_route = int(prepared_row["clean_route"])
            clean_records = [
                record
                for record in by_row.get(str(prepared_row["row_id"]), [])
                if int(record["expert"]) == clean_route
            ]
            route_invariance = (
                "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
                if complete
                and len(clean_records) == 1
                and clean_records[0]["crown"]["status"]
                == "CERTIFIED_MARGIN_FILTER"
                else "UNKNOWN_EXPERT_PROPERTY"
            )
        elif route_status == "PROVEN_ROUTE_UNSTABLE":
            route_invariance = "NOT_APPLICABLE_ROUTE_UNSTABLE"
        else:
            route_invariance = "UNKNOWN_ROUTE_STABILITY"
        fixed_radius_results.append(
            {
                **prepared_row,
                "candidate_experts": sorted(expected),
                "observed_experts": sorted(observed),
                "candidate_coverage_complete": complete,
                "route_invariance_crown": route_invariance,
                "route_invariance_formal_safe": False,
                "route_a_crown": route_a,
                "route_a_formal_safe": False,
                "numerical_scope": (
                    "positive-margin filter only; no outward-rounded formal SAFE"
                ),
            }
        )
    fixed_table: dict[str, Any] = {}
    for numerator in config["primary_table_epsilon_over_255"]:
        rows = [
            row
            for row in fixed_radius_results
            if float(row["epsilon_over_255"]) == float(numerator)
        ]
        fixed_table[str(numerator)] = {
            "samples": len(rows),
            "route_status_counts": dict(Counter(row["route_status"] for row in rows)),
            "route_invariance_status_counts": dict(
                Counter(row["route_invariance_crown"] for row in rows)
            ),
            "route_a_status_counts": dict(
                Counter(row["route_a_crown"] for row in rows)
            ),
            "formal_safe_count": 0,
        }
    backend_error_count = sum(
        record["crown"]["status"] == "ERROR" for record in branch_results
    )
    incomplete_bound_count = sum(
        not bool(record["crown"].get("complete", False)) for record in branch_results
    )
    summary = {
        "schema_version": 1,
        "status": (
            "COMPLETED_NUMERICAL_CONFORMANCE_ONLY"
            if backend_error_count == 0 and incomplete_bound_count == 0
            else "FAILED_BACKEND_INCOMPLETE"
        ),
        "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        "checkpoint": prepare["checkpoint"],
        "rows_artifact": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        "samples": sample_results,
        "fixed_radius_samples": fixed_radius_results,
        "fixed_radius_table": fixed_table,
        "branches": len(branch_results),
        "backend_error_count": backend_error_count,
        "incomplete_bound_count": incomplete_bound_count,
        "resource_gate": resource_gate,
        "branch_crown_status_counts": dict(
            Counter(record["crown"]["status"] for record in branch_results)
        ),
        "sample_route_a_status_counts": dict(
            Counter(record["route_a_crown"] for record in sample_results)
        ),
        "formal_safe_count": 0,
        "runtime_seconds": time.monotonic() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "auto_lirpa": _package_version_or_unavailable("auto-lirpa"),
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": _nvidia_driver_version(),
            "device": device,
        },
        "certificate_identity": prepare["certificate_identity"],
        "numerical_scope": (
            "auto_LiRPA positive-margin conformance filter; no outward-rounded "
            "formal SAFE claim and no negative-bound UNSAFE claim"
        ),
    }
    _write_json(output_path, summary)
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official repository became dirty during CROWN worker")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prepare, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
