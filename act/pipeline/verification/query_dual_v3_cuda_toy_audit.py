#!/usr/bin/env python3
"""Controlled real-CUDA audit for sealed sparse query-dual V3.

This command uses the fixed ADD residual-DAG toy and exact transaction
configuration that produced the pre-registered V2 timing comparator.  It
never loads ONNX/VNNLIB, calls Operator-HZ, invokes a verifier verdict, or
touches a benchmark dataset.  The candidate path is the production CUDA-f64
``DualSolver`` path; authority still comes from sealed independent CPU-f64
replay and live full-object validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from act.back_end.hybridz_tf.query_dual_pipeline import (
    validate_verified_query_dual_feedback,
)
from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
    build_verified_query_dual_feedback_v3,
)
from act.back_end.hybridz_tf.test_query_dual_operator_integration import (
    _residual_two_relu_toy,
)
from act.util.device_manager import initialize_device


_SCHEMA = "act.query_dual_v3_cuda_toy_audit.v1"
_HISTORICAL_V2_SECONDS = 0.7203334719
_REGRESSION_FACTOR = 1.10
_CUDA_MEMORY_GATE_BYTES = int(2.5 * 1024**3)
_WORKSPACE_GATE_BYTES = 512 * 1024 * 1024
_TARGETS = (7,)
_QUOTAS = (1,)
_SOURCE_FILES = (
    "act/pipeline/verification/query_dual_v3_cuda_toy_audit.py",
    "act/back_end/hybridz_tf/query_dual_box_certifier.py",
    "act/back_end/hybridz_tf/query_dual_replay.py",
    "act/back_end/hybridz_tf/property_residual_targets.py",
    "act/back_end/hybridz_tf/query_dual_candidates.py",
    "act/back_end/hybridz_tf/query_dual_pipeline.py",
    "act/back_end/hybridz_tf/query_dual_pipeline_v3.py",
    "act/back_end/solver/solver_dual.py",
    "act/util/device_manager.py",
    "act/back_end/hybridz_tf/test_query_dual_operator_integration.py",
    "act/back_end/hybridz_tf/test_query_dual_pipeline_v3.py",
    "act/back_end/hybridz_tf/test_query_dual_replay_v3.py",
    "act/back_end/hybridz_tf/test_property_residual_targets.py",
    "act/back_end/hybridz_tf/test_query_dual_candidates.py",
    "act/pipeline/verification/test_query_dual_v3_cuda_toy_audit.py",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> Dict[str, str]:
    return {
        path: _file_sha256(Path(path).resolve()) for path in _SOURCE_FILES
    }


def _atomic_json(
    path: Path, value: Mapping[str, Any], *, overwrite: bool
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"refusing to overwrite existing audit {path}"
                ) from exc
            temporary.unlink()
            temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cuda_toy() -> Any:
    toy = _residual_two_relu_toy()
    device = torch.device("cuda")
    for layer in toy.net.layers:
        for name, value in tuple(layer.params.items()):
            if isinstance(value, np.ndarray):
                layer.params[name] = torch.as_tensor(
                    value, device=device, dtype=torch.float64
                ).contiguous()
            elif isinstance(value, torch.Tensor):
                dtype = torch.float64 if value.is_floating_point() else value.dtype
                layer.params[name] = value.to(
                    device=device, dtype=dtype
                ).contiguous()
    return toy


def _run_transaction() -> Dict[str, Any]:
    toy = _cuda_toy()
    net = toy.net
    torch.cuda.synchronize()
    started = time.monotonic()
    deadline = started + 10.0
    bundle = build_verified_query_dual_feedback_v3(
        net,
        toy.C,
        toy.thresholds,
        target_relu_ids=_TARGETS,
        stage_quotas=_QUOTAS,
        steps=1,
        block_size=1,
        replay_chunk_size=16,
        replay_max_workspace_bytes=_WORKSPACE_GATE_BYTES,
        candidate_device="cuda",
        selector_time_limit=1.0,
        selector_max_adjoint_cells=30_000_000,
        selector_pool_per_rival=64,
        deadline=deadline,
    )
    torch.cuda.synchronize()
    build_seconds = time.monotonic() - started
    validation_started = time.monotonic()
    valid = validate_verified_query_dual_feedback(
        bundle,
        net=net,
        property_rows=toy.C,
        thresholds=toy.thresholds,
        expected_target_relu_ids=_TARGETS,
        require_live_provenance=True,
    )
    torch.cuda.synchronize()
    validation_seconds = time.monotonic() - validation_started
    strict = [int(stage.strict_improvements) for stage in bundle.stages]
    selected = [
        int(stage.candidate_receipt["selected_target_count"])
        for stage in bundle.stages
    ]
    if valid is not True:
        raise RuntimeError("fresh V3 bundle failed live validation")
    if any(count <= 0 for count in selected) or any(
        count <= 0 for count in strict
    ):
        raise RuntimeError(
            "every nonempty controlled target stage must strictly improve"
        )
    if (
        not np.all(np.isfinite(bundle.property_upper))
        or bundle.receipt.get("schema")
        != "act.verified_query_dual_feedback.v3"
        or bundle.receipt.get("candidate_device") != "cuda"
        or bundle.receipt.get("candidate_device_fallback") is not False
    ):
        raise RuntimeError("V3 CUDA transaction receipt binding failed")
    return {
        "build_seconds": build_seconds,
        "validation_seconds": validation_seconds,
        "build_plus_validation_seconds": (
            build_seconds + validation_seconds
        ),
        "strict_improvements": strict,
        "selected_target_counts": selected,
        "property_upper_hex": [
            float(value).hex() for value in bundle.property_upper
        ],
        "pipeline_receipt_sha256": bundle.receipt["receipt_sha256"],
        "candidate_device": bundle.receipt["candidate_device"],
        "candidate_device_fallback": bundle.receipt[
            "candidate_device_fallback"
        ],
    }


def run_audit(*, warmups: int = 1, repetitions: int = 3) -> Dict[str, Any]:
    started = time.monotonic()
    source_before: Dict[str, str] = {}
    body: Dict[str, Any]
    try:
        source_before = _source_hashes()
        if (
            isinstance(warmups, bool)
            or isinstance(repetitions, bool)
            or not isinstance(warmups, int)
            or not isinstance(repetitions, int)
            or int(warmups) < 0
            or int(repetitions) <= 0
            or int(repetitions) > 7
        ):
            raise ValueError("warmups must be >=0 and repetitions in [1,7]")
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
        torch.cuda.set_device(0)
        initialize_device(device="cuda", dtype="float64")
        if (
            torch.get_default_device().type != "cuda"
            or torch.get_default_dtype() != torch.float64
        ):
            raise RuntimeError("device manager did not retain CUDA float64")
        properties = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        )
        warmup_rows = [_run_transaction() for _ in range(int(warmups))]
        del warmup_rows
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        for _ in range(int(repetitions)):
            measurements.append(_run_transaction())
            gc.collect()
            torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        median_build = float(
            statistics.median(
                row["build_seconds"] for row in measurements
            )
        )
        median_total = float(
            statistics.median(
                row["build_plus_validation_seconds"]
                for row in measurements
            )
        )
        limit = _HISTORICAL_V2_SECONDS * _REGRESSION_FACTOR
        timing_pass = median_build <= limit
        memory_pass = (
            peak_allocated <= _CUDA_MEMORY_GATE_BYTES
            and peak_reserved <= _CUDA_MEMORY_GATE_BYTES
        )
        workspace_pass = (
            _WORKSPACE_GATE_BYTES <= 512 * 1024 * 1024
        )
        source_after = _source_hashes()
        source_stable = source_after == source_before
        passed = bool(
            timing_pass
            and memory_pass
            and workspace_pass
            and source_stable
        )
        body = {
            "schema": _SCHEMA,
            "status": "pass" if passed else "fail",
            "proof_authority": False,
            "controlled_toy_only": True,
            "real_model_run": False,
            "onnx_loaded": False,
            "vnnlib_loaded": False,
            "operator_hz_called": False,
            "solver_verdict_called": False,
            "production_cuda_candidate_path": True,
            "production_default_selector": True,
            "production_default_solver_factory": True,
            "independent_cpu_replay_authority": True,
            "config": {
                "toy": (
                    "residual_two_relu_add_dag"
                ),
                "baseline_config_provenance": (
                    "test_query_dual_operator_integration."
                    "_residual_two_relu_toy plus documented V2 helper "
                    "steps=1/block=1/replay_chunk=16"
                ),
                "targets": list(_TARGETS),
                "stage_quotas": list(_QUOTAS),
                "steps": 1,
                "block_size": 1,
                "replay_chunk_size": 16,
                "replay_max_workspace_bytes": _WORKSPACE_GATE_BYTES,
                "selector_time_limit_seconds": 1.0,
                "warmups": int(warmups),
                "repetitions": int(repetitions),
                "dtype": "float64",
                "device": "cuda:0",
            },
            "cpu_parallelism": {
                "transaction_workers": 1,
                "worker_policy": (
                    "single_sequential_proof_worker_with_internal_"
                    "tensor_parallelism"
                ),
                "torch_intraop_threads": int(torch.get_num_threads()),
                "torch_interop_threads": int(
                    torch.get_num_interop_threads()
                ),
                "logical_cpu_count": os.cpu_count(),
            },
            "cuda": {
                "device_index": int(torch.cuda.current_device()),
                "device_name": str(properties.name),
                "cuda_build": str(torch.version.cuda),
                "total_bytes": int(properties.total_memory),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "maximum_peak_bytes": _CUDA_MEMORY_GATE_BYTES,
                "memory_pass": memory_pass,
            },
            "workspace": {
                "configured_max_bytes": _WORKSPACE_GATE_BYTES,
                "maximum_bytes": 512 * 1024 * 1024,
                "pass": workspace_pass,
            },
            "timing": {
                "historical_v2_seconds": _HISTORICAL_V2_SECONDS,
                "maximum_regression_factor": _REGRESSION_FACTOR,
                "maximum_v3_seconds": limit,
                "median_v3_build_seconds": median_build,
                "median_v3_build_plus_validation_seconds": median_total,
                "ratio_to_historical_v2": (
                    median_build / _HISTORICAL_V2_SECONDS
                ),
                "measurement_scope": (
                    "cuda_synchronized_builder_through_atomic_commit;"
                    "external_live_validation_reported_separately"
                ),
                "comparator_policy": (
                    "fixed_pre_registered_historical_v2_transaction"
                ),
                "pass": timing_pass,
                "measurements": measurements,
            },
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "source_integrity_stable": source_stable,
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:
        try:
            source_after = _source_hashes()
        except Exception:
            source_after = {}
        body = {
            "schema": _SCHEMA,
            "status": "fail",
            "proof_authority": False,
            "controlled_toy_only": True,
            "real_model_run": False,
            "onnx_loaded": False,
            "vnnlib_loaded": False,
            "operator_hz_called": False,
            "solver_verdict_called": False,
            "production_cuda_candidate_path": True,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            },
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "source_integrity_stable": bool(
                source_before
                and source_after
                and source_before == source_after
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
    body["receipt_sha256"] = _canonical_sha256(body)
    if not math.isfinite(float(body["elapsed_seconds"])):
        raise RuntimeError("non-finite audit duration")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_audit(
        warmups=int(args.warmups), repetitions=int(args.repetitions)
    )
    _atomic_json(args.output, result, overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_sha256": result["receipt_sha256"],
                "timing": result.get("timing"),
                "cuda": result.get("cuda"),
                "error": result.get("error"),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
