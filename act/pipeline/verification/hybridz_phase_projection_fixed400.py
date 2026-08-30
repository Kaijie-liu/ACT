#!/usr/bin/env python3
"""Fixed-400 formal gate for the single-stream phase-projection verifier path.

This is an offline gate runner, not a production dispatch.  It enumerates the
complete registered CIFAR100-medium, CIFAR100-large, and TinyImageNet-medium
CSV ranges in fixed order.  Each case runs ``verify_once`` in a fresh process
with the single exact phase-projection path enabled.  There is no input
sampling, concrete ONNX execution, PGD, BaB, backward pass, dual tightening,
or solver fallback.

Every completed result is fsync'd to JSONL and the summary is atomically
replaced, so an interrupted run can be resumed without losing prior cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
)
OUTPUT_JSONL = (
    REPO_ROOT
    / "artifacts/hybridz_largecls_gates/phase_projection_gpu_csr_fixed400_20260814.jsonl"
)
SUMMARY_JSON = (
    REPO_ROOT
    / "artifacts/hybridz_largecls_gates/phase_projection_gpu_csr_fixed400_20260814.summary.json"
)
WORKER_TIMEOUT_SECONDS = 45.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cases() -> tuple[dict[str, Any], ...]:
    families = (
        ("cifar100_medium", "cifar100_2024", 0, 99),
        ("cifar100_large", "cifar100_2024", 100, 199),
        ("tinyimagenet_medium", "tinyimagenet_2024", 0, 199),
    )
    result = []
    for family, benchmark, first, last in families:
        root = BENCHMARK_ROOT / benchmark
        with (root / "instances.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.reader(handle))
        if last >= len(rows):
            raise RuntimeError(
                f"{benchmark} instances.csv has only {len(rows)} rows"
            )
        for iid in range(first, last + 1):
            row = rows[iid]
            if len(row) != 3:
                raise RuntimeError(
                    f"{benchmark} instances.csv row {iid} is malformed"
                )
            model = (root / row[0]).resolve()
            spec = (root / row[1]).resolve()
            expected_model = (
                "CIFAR100_resnet_medium.onnx"
                if family == "cifar100_medium"
                else "CIFAR100_resnet_large.onnx"
                if family == "cifar100_large"
                else "TinyImageNet_resnet_medium.onnx"
            )
            if model.name != expected_model or not model.is_file() or not spec.is_file():
                raise RuntimeError(
                    f"fixed400 row {family}/iid{iid} has unexpected artifacts"
                )
            result.append(
                {
                    "case": f"{family}_iid{iid}",
                    "family": family,
                    "benchmark": benchmark,
                    "iid": iid,
                    "onnx": str(model),
                    "vnnlib": str(spec),
                }
            )
    if len(result) != 400 or len({item["case"] for item in result}) != 400:
        raise RuntimeError("fixed400 manifest is not exactly 400 unique cases")
    return tuple(result)


def _validated_status(record: dict[str, Any]) -> str:
    status = record.get("status")
    projection = record.get("phase_projection")
    if not isinstance(projection, dict):
        return "ERROR"
    forbidden = (
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
    )
    if any(projection.get(key) is not False for key in forbidden):
        return "ERROR"
    if status == "VerifyStatus.FALSIFIED":
        candidate = projection.get("candidate_receipt")
        if not isinstance(candidate, dict):
            return "ERROR"
        if not (
            record.get("has_counterexample") is True
            and projection.get("status") == "FALSIFIED"
            and projection.get("verifier_owned_proof_authority") is True
            and candidate.get("status") == "singleton_verified"
            and candidate.get("all_unstable_exact") is True
            and candidate.get("triangle_rows") == 0
            and candidate.get("phase_retries") == 0
            and candidate.get("property_row_retries") == 0
            and candidate.get("proof_authority") is False
            and candidate.get("verdict_authority") is False
        ):
            return "ERROR"
        return "FALSIFIED"
    if status == "VerifyStatus.UNKNOWN":
        if not (
            record.get("has_counterexample") is False
            and projection.get("status") == "UNKNOWN"
            and projection.get("verifier_owned_proof_authority") is False
            and "candidate_receipt" not in projection
        ):
            return "ERROR"
        return "UNKNOWN"
    return "ERROR"


def _worker(case: dict[str, Any]) -> int:
    import torch

    from act.back_end.config import BackendConfig, HybridZConfig
    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )
    from act.back_end.verifier import verify_once
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
    from act.pipeline.verification.torch2act import TorchToACT
    from act.util.device_manager import initialize_device

    onnx = Path(case["onnx"])
    vnnlib = Path(case["vnnlib"])
    before = {"onnx": _sha256(onnx), "vnnlib": _sha256(vnnlib)}
    started = time.monotonic()
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    specs = create_specs_from_paths(
        str(onnx), str(vnnlib), category=str(case["benchmark"])
    )
    wrapped = synthesize_models_from_specs([specs])
    if len(wrapped) != 1:
        raise RuntimeError("fixed400 worker requires exactly one wrapped model")
    model = next(iter(wrapped.values())).to(
        device=torch.device("cuda"), dtype=torch.float64
    )
    net = TorchToACT(model).run()
    result = verify_once(
        net,
        backend_cfg=BackendConfig(
            solver="hybridz",
            device="cuda",
            dtype="float64",
            timeout=30.0,
            hybridz=HybridZConfig(
                timeout=20.0,
                engine="operator_hz_objbound",
                operator_exact_budget=-1,
                operator_phase_projection_time_limit=10.0,
                operator_materialize_add=True,
            ),
        ),
    )[0]
    after = {"onnx": _sha256(onnx), "vnnlib": _sha256(vnnlib)}
    if after != before:
        raise RuntimeError("fixed400 input artifact changed during verification")
    projection = result.metadata.get("operator_phase_projection", {})
    record = {
        "schema": "act.hybridz.phase_projection_fixed400.worker.v1",
        "case": case["case"],
        "family": case["family"],
        "iid": case["iid"],
        "onnx": str(onnx),
        "vnnlib": str(vnnlib),
        "input_sha256": before,
        "status": str(result.status),
        "has_counterexample": result.counterexample is not None,
        "phase_projection": projection,
        "elapsed_seconds": time.monotonic() - started,
    }
    record["validated_status"] = _validated_status(record)
    print(_canonical_json(record), flush=True)
    return 0 if record["validated_status"] != "ERROR" else 2


def _existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            case = value.get("case")
            if type(case) is not str or case in records:
                raise RuntimeError(
                    f"resume JSONL has invalid/duplicate case on line {number}"
                )
            if _validated_status(value) != value.get("validated_status"):
                raise RuntimeError(
                    f"resume JSONL case {case} failed strict revalidation"
                )
            records[case] = value
    return records


def _summary(
    manifest: Iterable[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    manifest = tuple(manifest)
    family_counts: dict[str, dict[str, int]] = {}
    for item in manifest:
        family = str(item["family"])
        counts = family_counts.setdefault(
            family,
            {"completed": 0, "FALSIFIED": 0, "UNKNOWN": 0, "ERROR": 0},
        )
        record = records.get(str(item["case"]))
        if record is not None:
            counts["completed"] += 1
            counts[str(record["validated_status"])] += 1
    totals = {
        key: sum(value[key] for value in family_counts.values())
        for key in ("completed", "FALSIFIED", "UNKNOWN", "ERROR")
    }
    source_paths = (
        Path(__file__),
        REPO_ROOT / "act/back_end/config.py",
        REPO_ROOT / "act/back_end/verifier.py",
        REPO_ROOT
        / "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py",
        REPO_ROOT
        / "act/back_end/hybridz_tf/forward_exact_relu_live_row_stream_candidate.py",
    )
    return {
        "schema": "act.hybridz.phase_projection_fixed400.summary.v1",
        "status": (
            "COMPLETE"
            if totals["completed"] == 400 and totals["ERROR"] == 0
            else "IN_PROGRESS"
        ),
        "manifest_cases": len(manifest),
        "totals": totals,
        "families": family_counts,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in source_paths
        },
        "prohibitions": {
            "input_sampling_used": False,
            "pgd_used": False,
            "concrete_onnx_execution_used": False,
            "act_bab_used": False,
            "backward_used": False,
            "dual_tightening_used": False,
            "root_solver_fallback_used": False,
        },
        "checkpoint_jsonl": str(OUTPUT_JSONL.relative_to(REPO_ROOT)),
    }


def _parent() -> int:
    manifest = _cases()
    records = _existing(OUTPUT_JSONL)
    expected = {str(case["case"]) for case in manifest}
    if not set(records).issubset(expected):
        raise RuntimeError("resume JSONL contains a case outside fixed400")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(SUMMARY_JSON, _summary(manifest, records))
    for case in manifest:
        name = str(case["case"])
        if name in records:
            continue
        command = [
            sys.executable,
            "-m",
            "act.pipeline.verification.hybridz_phase_projection_fixed400",
            "--worker-json",
            _canonical_json(case),
        ]
        env = dict(os.environ)
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            record = {
                "schema": "act.hybridz.phase_projection_fixed400.worker.v1",
                "case": name,
                "family": case["family"],
                "iid": case["iid"],
                "status": "worker_timeout",
                "has_counterexample": False,
                "phase_projection": {},
                "elapsed_seconds": time.monotonic() - started,
                "validated_status": "ERROR",
                "error": type(exc).__name__,
            }
        else:
            parsed = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.startswith("{")
            ]
            if completed.returncode == 0 and len(parsed) == 1:
                record = parsed[0]
            else:
                record = {
                    "schema": "act.hybridz.phase_projection_fixed400.worker.v1",
                    "case": name,
                    "family": case["family"],
                    "iid": case["iid"],
                    "status": "worker_error",
                    "has_counterexample": False,
                    "phase_projection": {},
                    "elapsed_seconds": time.monotonic() - started,
                    "validated_status": "ERROR",
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                }
        if record.get("case") != name:
            raise RuntimeError("worker result case does not match request")
        with OUTPUT_JSONL.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        records[name] = record
        _atomic_json(SUMMARY_JSON, _summary(manifest, records))
        print(
            _canonical_json(
                {
                    "case": name,
                    "completed": len(records),
                    "validated_status": record["validated_status"],
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    summary = _summary(manifest, records)
    _atomic_json(SUMMARY_JSON, summary)
    print(_canonical_json(summary), flush=True)
    return 0 if summary["status"] == "COMPLETE" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_json is not None:
        value = json.loads(args.worker_json)
        if not isinstance(value, dict):
            raise RuntimeError("worker payload must be a mapping")
        return _worker(value)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
