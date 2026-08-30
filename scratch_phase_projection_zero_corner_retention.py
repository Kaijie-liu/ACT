#!/usr/bin/env python3
"""Fresh-child retention audits for the sole zero-correction rule."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
DATA = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
SOURCE = ROOT / "artifacts/hybridz_largecls_gates/phase_projection_single_stream_float64_candidate_20260813.json"
PROBE = ROOT / "scratch_phase_projection_zero_corner_probe.py"


def _instance(case: str) -> tuple[str, Path, Path]:
    category = "tinyimagenet_2024" if case.startswith("tinyimagenet_") else "cifar100_2024"
    index = int(case.rsplit("iid", 1)[1])
    with (DATA / category / "instances.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    model_rel, spec_rel, _timeout = rows[index]
    expected = "large" if "_large_" in case else "medium"
    if expected not in Path(model_rel).name.lower():
        raise RuntimeError(f"{case} disagrees with instances.csv model")
    return category, DATA / category / model_rel, DATA / category / spec_rel


def _cases(mode: str) -> tuple[str, ...]:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if mode == "fixed14":
        section = data["fixed_14"]
        values = section["candidate_terminal_verified"] + section["remaining_unknown"]
    elif mode == "retained59":
        values = list(data["retention_43"]["retained_cases"])
        values.extend(
            item["case"]
            for item in data["official_sat_current_unknown_cases"]
            if item["status"] == "FALSIFIED"
        )
    else:
        raise RuntimeError("ACT_ZERO_CORNER_SET must be fixed14 or retained59")
    return tuple(dict.fromkeys(values))


def _run(case: str) -> dict[str, object]:
    category, model, spec = _instance(case)
    env = dict(os.environ)
    env.update(
        {
            "ACT_PHASE_PROJECTION_ONNX": str(model),
            "ACT_PHASE_PROJECTION_VNNLIB": str(spec),
            "ACT_PHASE_PROJECTION_CATEGORY": category,
            "ACT_PHASE_PROJECTION_CASE": case,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(PROBE)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except subprocess.TimeoutExpired:
        return {
            "case": case,
            "status": "ERROR",
            "reason": "fresh child exceeded 60 seconds",
            "fresh_child_seconds": time.monotonic() - started,
        }
    records = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    if completed.returncode != 0 or not records:
        return {
            "case": case,
            "status": "ERROR",
            "reason": "fresh child failed or emitted no JSON",
            "returncode": completed.returncode,
            "fresh_child_seconds": time.monotonic() - started,
        }
    record = records[-1]
    timing = record.get("timing") or {}
    return {
        "case": case,
        "status": (
            "FALSIFIED"
            if record.get("status") == "TERMINAL_VERIFIED"
            else "UNKNOWN"
        ),
        "raw_box_verified": record.get("raw_box_verified"),
        "selected_property_row": record.get("selected_property_row"),
        "first_cell_affine_margin": record.get("first_cell_affine_margin"),
        "terminal_exact_margin_lower": record.get("terminal_exact_margin_lower"),
        "corner_path_seconds_instrumented": timing.get(
            "corner_path_seconds_instrumented"
        ),
        "terminal_seconds": timing.get("terminal_seconds"),
        "candidate_plus_terminal_seconds_instrumented": timing.get(
            "candidate_plus_terminal_seconds_instrumented"
        ),
        "fresh_child_seconds": time.monotonic() - started,
    }


def main() -> None:
    mode = os.environ.get("ACT_ZERO_CORNER_SET", "fixed14")
    cases = _cases(mode)
    results = []
    started = time.monotonic()
    for index, case in enumerate(cases, start=1):
        result = _run(case)
        results.append(result)
        print(
            json.dumps(
                {"index": index, "total": len(cases), **result},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    summary = {
        "schema": "act.scratch.phase_projection_zero_corner_retention.v1",
        "mode": mode,
        "attempted": len(results),
        "falsified": sum(item["status"] == "FALSIFIED" for item in results),
        "unknown": sum(item["status"] == "UNKNOWN" for item in results),
        "errors": sum(item["status"] == "ERROR" for item in results),
        "elapsed_seconds": time.monotonic() - started,
        "falsified_cases": [
            item["case"] for item in results if item["status"] == "FALSIFIED"
        ],
        "unknown_cases": [
            item["case"] for item in results if item["status"] == "UNKNOWN"
        ],
        "results": results,
        "scope": {
            "rule": "unique_first_stream_analytic_inward_box_corner",
            "candidate_authority": False,
            "terminal_authority": (
                "raw_BOX;verifier_owned_zero_width_interval;"
                "stored_binary64_Fraction_property"
            ),
            "input_sampling_used": False,
            "onnx_point_execution_used": False,
            "pgd_used": False,
            "bab_or_split_used": False,
            "backward_bounds_used": False,
            "dual_tightening_used": False,
            "target_cell_or_delta_expansion_used": False,
            "lp_model_or_solve_used": False,
            "fallbacks_or_retries": 0,
            "production_modified": False,
            "workers": 1,
        },
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
