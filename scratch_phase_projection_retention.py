#!/usr/bin/env python3
"""Fresh-child retention gates for the float-only candidate."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
DATA = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
ARTIFACT = ROOT / "artifacts/hybridz_largecls_gates/phase_projection_single_stream_float64_candidate_20260813.json"


def _instance(case):
    category = "tinyimagenet_2024" if case.startswith("tinyimagenet_") else "cifar100_2024"
    index = int(case.rsplit("iid", 1)[1])
    with (DATA / category / "instances.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    model_rel, spec_rel, _timeout = rows[index]
    expected = "large" if "_large_" in case else "medium"
    if expected not in Path(model_rel).name.lower():
        raise RuntimeError(f"{case} disagrees with instances.csv model")
    return category, DATA / category / model_rel, DATA / category / spec_rel


def main():
    mode = os.environ.get("ACT_RETENTION_SET", "fixed14")
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if mode == "fixed14":
        section = data["fixed_14"]
        cases = (
            section["candidate_terminal_verified"]
            + section["remaining_unknown"]
        )
    elif mode == "retained59":
        cases = list(data["retention_43"]["retained_cases"])
        cases.extend(
            item["case"]
            for item in data["official_sat_current_unknown_cases"]
            if item["status"] == "FALSIFIED"
        )
    elif mode == "fixed400":
        cases = [f"cifar100_{'medium' if index < 100 else 'large'}_iid{index}" for index in range(200)]
        cases.extend(f"tinyimagenet_medium_iid{index}" for index in range(200))
    else:
        raise RuntimeError("ACT_RETENTION_SET must be fixed14, retained59, or fixed400")
    cases = tuple(dict.fromkeys(cases))
    results = []
    started = time.monotonic()

    def run_case(case):
        category, model, spec = _instance(case)
        env = dict(os.environ)
        env.update({
            "ACT_PHASE_PROJECTION_ONNX": str(model),
            "ACT_PHASE_PROJECTION_VNNLIB": str(spec),
            "ACT_PHASE_PROJECTION_CATEGORY": category,
            "ACT_FLOAT_PROBE_REPEATS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        child_started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scratch_phase_projection_float64_probe.py")],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=60.0,
            )
            records = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.startswith("{")
            ]
            record = records[-1] if completed.returncode == 0 and records else {
                "status": "ERROR",
                "reason": "fresh child failed or emitted no JSON",
                "returncode": completed.returncode,
            }
        except subprocess.TimeoutExpired:
            record = {"status": "TIMEOUT", "reason": "fresh child exceeded 60 seconds"}
        receipt = record.get("receipt") or {}
        result = {
            "case": case,
            "status": record.get("status"),
            "reason": record.get("reason"),
            "float_margin": receipt.get("float_margin"),
            "singleton_margin_lower": receipt.get("singleton_margin_lower"),
            "candidate_total_seconds": receipt.get("total_seconds"),
            "fresh_child_seconds": time.monotonic() - child_started,
        }
        return result

    workers = 4 if mode == "fixed400" else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_case, cases):
            results.append(result)
            print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    summary = {
        "schema": "act.hybridz.phase_projection_retention.v1",
        "mode": mode,
        "attempted": len(results),
        "falsified": sum(item["status"] == "FALSIFIED" for item in results),
        "unknown": sum(item["status"] == "UNKNOWN" for item in results),
        "errors": sum(item["status"] not in {"FALSIFIED", "UNKNOWN"} for item in results),
        "elapsed_seconds": time.monotonic() - started,
        "workers": workers,
        "falsified_cases": [
            item["case"] for item in results if item["status"] == "FALSIFIED"
        ],
        "results": results,
        "scope": {
            "candidate_authority": False,
            "terminal_proof_unchanged": True,
            "input_sampling_used": False,
            "onnx_input_execution_used": False,
            "pgd_used": False,
            "bab_or_split_used": False,
            "backward_used": False,
            "dual_tightening_used": False,
            "runtime_fallbacks": 0,
        },
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
