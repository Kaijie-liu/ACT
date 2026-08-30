#!/usr/bin/env python3
# ===- query_dual_probe.py - strict query-dual Gate-1 probe ----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Strict, verdict-free Gate-1 probe for query-dual transactions.

This entry point deliberately stops after construction and live validation of
``VerifiedQueryDualFeedback``.  It never imports or calls Operator-HZ, an HZ
solver, or a verdict routine.  The production path is exactly:

``create_specs_from_paths -> synthesize_models_from_specs -> TorchToACT
-> an explicitly selected V2/V3 feedback builder -> live full-object
validation``.

Both inputs and every local proof-path source are hashed before and after the
transaction.  CUDA binary64 is mandatory; CPU fallback is an error.  Output is
atomically created without clobbering by default; ``--overwrite`` is explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


_SCHEMA_V2 = "act.query_dual_gate1_probe.v1"
_SCHEMA_V3 = "act.query_dual_gate1_probe.v2"
_UNSET = object()
_TRANSACTION_HARD_LIMIT_SECONDS = 12.0
_TRANSACTION_STOP_LOSS_SECONDS = 10.5
_REPLAY_MAX_WORKSPACE_BYTES = 512 * 1024 * 1024
_V3_CUDA_MEMORY_STOP_LOSS_BYTES = int(2.5 * 1024**3)
_V3_CPU_RSS_INCREMENT_STOP_LOSS_BYTES = 2 * 1024**3
_PIPELINE_V2 = "full_descriptor_replay_v2"
_PIPELINE_V3 = "property_sparse_sealed_replay_v3"
_SOURCE_MODULES_COMMON = (
    "act.front_end.vnnlib_loader.create_specs",
    "act.front_end.model_synthesis",
    "act.pipeline.verification.torch2act",
    "act.util.device_manager",
    "act.back_end.hybridz_tf.query_dual_box_certifier",
    "act.back_end.hybridz_tf.query_dual_candidates",
    "act.back_end.hybridz_tf.query_dual_replay",
    "act.back_end.solver.solver_dual",
)
_SOURCE_MODULES_V2 = _SOURCE_MODULES_COMMON + (
    "act.back_end.hybridz_tf.query_dual_pipeline",
)
_SOURCE_MODULES_V3 = _SOURCE_MODULES_COMMON + (
    "act.back_end.hybridz_tf.property_residual_targets",
    "act.back_end.hybridz_tf.query_dual_pipeline",
    "act.back_end.hybridz_tf.query_dual_pipeline_v3",
)


class QueryDualProbeError(RuntimeError):
    """Fail-closed probe error."""


@dataclass(frozen=True)
class QueryDualProbeConfig:
    onnx_path: Path
    vnnlib_path: Path
    target_relu_ids: Tuple[int, ...]
    steps: int
    time_limit: float
    block_size: int
    device: str
    output_path: Path
    overwrite: bool = False
    stage_quotas: Optional[Tuple[int, ...]] = None
    selector_time_limit: float = 1.0
    selector_max_adjoint_cells: int = 30_000_000
    selector_pool_per_rival: int = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _finalize_receipt(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _paths_alias(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    if left == right:
        return True
    try:
        return bool(left.exists() and right.exists() and left.samefile(right))
    except OSError:
        return False


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
    forbidden_paths: Sequence[Path] = (),
) -> None:
    path = path.expanduser().resolve()
    if any(_paths_alias(path, forbidden) for forbidden in forbidden_paths):
        raise QueryDualProbeError(
            "OUTPUT_ALIAS: output must not alias an ONNX/VNNLIB input"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary: Optional[Path] = None
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
                # Same-directory hard-link publication is an atomic
                # create-if-absent operation; unlike a preflight exists()
                # check, it closes the output-name race.
                os.link(temporary, path)
            except FileExistsError as exc:
                raise QueryDualProbeError(
                    f"OUTPUT_EXISTS: refusing to overwrite {path}"
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


def _array_digest(value: Any) -> str:
    """Use the query-dual pipeline's canonical little-endian f64 digest."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<f8\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _parse_targets(value: str) -> Tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("targets must be a nonempty CSV")
    try:
        targets = tuple(int(token.strip()) for token in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "targets must be comma-separated integer layer ids"
        ) from exc
    if (
        not targets
        or any(value < 0 for value in targets)
        or len(set(targets)) != len(targets)
    ):
        raise argparse.ArgumentTypeError(
            "targets must be unique nonnegative layer ids"
        )
    return targets


def _parse_stage_quotas(value: str) -> Tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("V3 quotas must be a nonempty CSV")
    try:
        quotas = tuple(int(token.strip()) for token in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "V3 quotas must be comma-separated integers"
        ) from exc
    if not quotas or any(quota < 0 or quota > 64 for quota in quotas):
        raise argparse.ArgumentTypeError(
            "each V3 quota must be an integer in [0,64]"
        )
    return quotas


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def _positive_seconds(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected positive finite seconds"
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("expected positive finite seconds")
    return result


def _validate_config(config: QueryDualProbeConfig) -> QueryDualProbeConfig:
    if config.device != "cuda":
        raise QueryDualProbeError(
            "DEVICE_POLICY: only --device cuda is allowed; CPU fallback is "
            "forbidden"
        )
    if (
        not config.target_relu_ids
        or len(set(config.target_relu_ids)) != len(config.target_relu_ids)
        or any(value < 0 for value in config.target_relu_ids)
    ):
        raise QueryDualProbeError(
            "INVALID_TARGETS: targets must be unique nonnegative ids"
        )
    if isinstance(config.steps, bool) or int(config.steps) <= 0:
        raise QueryDualProbeError("INVALID_STEPS: steps must be positive")
    if isinstance(config.block_size, bool) or int(config.block_size) <= 0:
        raise QueryDualProbeError("INVALID_BLOCK: block must be positive")
    if (
        not math.isfinite(float(config.time_limit))
        or float(config.time_limit) <= 0.0
    ):
        raise QueryDualProbeError(
            "INVALID_TIME: time must be positive and finite"
        )
    if float(config.time_limit) > _TRANSACTION_HARD_LIMIT_SECONDS:
        raise QueryDualProbeError(
            "INVALID_TIME: transaction deadline must not exceed the "
            f"pre-registered {_TRANSACTION_HARD_LIMIT_SECONDS:.0f}s hard "
            "limit"
        )
    quotas: Optional[Tuple[int, ...]]
    if config.stage_quotas is None:
        quotas = None
    else:
        if isinstance(config.stage_quotas, (str, bytes)):
            raise QueryDualProbeError(
                "INVALID_V3_QUOTAS: expected an explicit integer sequence"
            )
        try:
            quotas = tuple(config.stage_quotas)
        except TypeError as exc:
            raise QueryDualProbeError(
                "INVALID_V3_QUOTAS: expected an explicit integer sequence"
            ) from exc
        if len(quotas) != len(config.target_relu_ids):
            raise QueryDualProbeError(
                "INVALID_V3_QUOTAS: quota count must equal target count"
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            or int(value) > 64
            for value in quotas
        ):
            raise QueryDualProbeError(
                "INVALID_V3_QUOTAS: each quota must be an integer in [0,64]"
            )
        quotas = tuple(int(value) for value in quotas)
        if not any(quotas):
            raise QueryDualProbeError(
                "INVALID_V3_QUOTAS: at least one target quota must be nonzero"
            )
    if (
        isinstance(config.selector_time_limit, bool)
        or not math.isfinite(float(config.selector_time_limit))
        or float(config.selector_time_limit) <= 0.0
    ):
        raise QueryDualProbeError(
            "INVALID_SELECTOR_TIME: expected positive finite seconds"
        )
    for name, value in (
        ("selector_max_adjoint_cells", config.selector_max_adjoint_cells),
        ("selector_pool_per_rival", config.selector_pool_per_rival),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise QueryDualProbeError(
                f"INVALID_SELECTOR_CONFIG: {name} must be positive"
            )
    if quotas and max(quotas) > int(config.selector_pool_per_rival):
        raise QueryDualProbeError(
            "INVALID_SELECTOR_CONFIG: selector pool must cover every quota"
        )
    onnx = config.onnx_path.expanduser().resolve(strict=True)
    vnnlib = config.vnnlib_path.expanduser().resolve(strict=True)
    if not onnx.is_file() or onnx.suffix.lower() != ".onnx":
        raise QueryDualProbeError("INVALID_ONNX: expected one .onnx file")
    if not vnnlib.is_file() or vnnlib.suffix.lower() != ".vnnlib":
        raise QueryDualProbeError(
            "INVALID_VNNLIB: expected one .vnnlib file"
        )
    if not isinstance(config.overwrite, bool):
        raise QueryDualProbeError("INVALID_OVERWRITE: overwrite must be boolean")
    output = config.output_path.expanduser().resolve()
    if _paths_alias(output, onnx) or _paths_alias(output, vnnlib):
        raise QueryDualProbeError(
            "OUTPUT_ALIAS: output must not alias an ONNX/VNNLIB input"
        )
    if output.exists() and not config.overwrite:
        raise QueryDualProbeError(
            f"OUTPUT_EXISTS: refusing to overwrite {output}"
        )
    return QueryDualProbeConfig(
        onnx_path=onnx,
        vnnlib_path=vnnlib,
        target_relu_ids=tuple(int(value) for value in config.target_relu_ids),
        steps=int(config.steps),
        time_limit=float(config.time_limit),
        block_size=int(config.block_size),
        device="cuda",
        output_path=output,
        overwrite=config.overwrite,
        stage_quotas=quotas,
        selector_time_limit=float(config.selector_time_limit),
        selector_max_adjoint_cells=int(config.selector_max_adjoint_cells),
        selector_pool_per_rival=int(config.selector_pool_per_rival),
    )


def _initialize_cuda_f64(device: str) -> Dict[str, Any]:
    if device != "cuda":
        raise QueryDualProbeError(
            "DEVICE_POLICY: CUDA is mandatory and fallback is forbidden"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise QueryDualProbeError(
            "CUDA_UNAVAILABLE: CPU fallback is forbidden"
        )
    from act.util.device_manager import initialize_device

    torch.cuda.set_device(0)
    initialize_device("cuda", "float64")
    default_device = torch.get_default_device()
    if default_device.type != "cuda":
        raise QueryDualProbeError(
            "CUDA_FALLBACK: device manager did not retain CUDA"
        )
    if torch.get_default_dtype() != torch.float64:
        raise QueryDualProbeError(
            "DTYPE_FALLBACK: device manager did not retain float64"
        )
    probe = torch.empty(1, device=torch.device("cuda"), dtype=torch.float64)
    if probe.device.type != "cuda" or probe.dtype != torch.float64:
        raise QueryDualProbeError(
            "CUDA_PROBE_FAILED: allocation did not use CUDA float64"
        )
    del probe
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "status": "tracking",
        "device": "cuda",
        "device_index": int(torch.cuda.current_device()),
        "device_name": str(properties.name),
        "dtype": "float64",
        "cuda_build": str(torch.version.cuda),
    }


def _cuda_memory_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "status": "unavailable",
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "total_bytes": None,
        }
    torch.cuda.synchronize()
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    return {
        "status": "measured",
        "device_index": index,
        "device_name": str(properties.name),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        "total_bytes": int(properties.total_memory),
    }


def _cpu_parallelism_snapshot() -> Dict[str, Any]:
    return {
        "transaction_workers": 1,
        "worker_policy": (
            "single_sequential_proof_worker_with_internal_tensor_parallelism"
        ),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "logical_cpu_count": os.cpu_count(),
    }


def _peak_rss_bytes() -> int:
    # Linux reports KiB.  This repository's supported benchmark runners are
    # Linux; binding the raw unit here keeps receipts comparable.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _validate_cuda_memory(
    memory: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> None:
    integer_fields = (
        "device_index",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "total_bytes",
    )
    if memory.get("status") != "measured":
        raise QueryDualProbeError(
            "CUDA_MEMORY: peak memory measurement is unavailable"
        )
    for name in integer_fields:
        value = memory.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QueryDualProbeError(
                f"CUDA_MEMORY: {name} must be a nonnegative integer"
            )
    environment_index = environment.get("device_index")
    if (
        isinstance(environment_index, bool)
        or not isinstance(environment_index, int)
        or environment_index < 0
        or memory["device_index"] != environment_index
    ):
        raise QueryDualProbeError(
            "CUDA_MEMORY: device index differs from CUDA initialization"
        )
    environment_name = environment.get("device_name")
    if (
        not isinstance(environment_name, str)
        or not environment_name
        or not isinstance(memory.get("device_name"), str)
        or memory["device_name"] != environment_name
    ):
        raise QueryDualProbeError(
            "CUDA_MEMORY: device name differs from CUDA initialization"
        )
    allocated = memory["peak_allocated_bytes"]
    reserved = memory["peak_reserved_bytes"]
    total = memory["total_bytes"]
    if not allocated <= reserved <= total:
        raise QueryDualProbeError(
            "CUDA_MEMORY: expected allocated <= reserved <= total"
        )


def _production_dependency_policy(*hooks: Any) -> bool:
    """Only the private all-default sentinel path may retain authority."""

    return all(value is _UNSET for value in hooks)


def _move_model_cuda_f64(model: Any) -> Any:
    move = getattr(model, "to", None)
    if not callable(move):
        raise QueryDualProbeError(
            "SYNTHESIS_MODEL: synthesized object has no .to()"
        )
    moved = move(device=torch.device("cuda"), dtype=torch.float64)
    if moved is not None:
        model = moved
    tensors = []
    parameters = getattr(model, "parameters", None)
    buffers = getattr(model, "buffers", None)
    if callable(parameters):
        tensors.extend(list(parameters()))
    if callable(buffers):
        tensors.extend(list(buffers()))
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            raise QueryDualProbeError(
                "MODEL_DEVICE_FALLBACK: a parameter/buffer is not on CUDA"
            )
        if tensor.is_floating_point() and tensor.dtype != torch.float64:
            raise QueryDualProbeError(
                "MODEL_DTYPE_FALLBACK: a floating parameter/buffer is not f64"
            )
    return model


def _extract_assert_property(net: Any) -> Tuple[np.ndarray, np.ndarray]:
    assertions = [
        layer
        for layer in net.layers
        if str(getattr(layer.kind, "value", layer.kind)).upper() == "ASSERT"
    ]
    if len(assertions) != 1:
        raise QueryDualProbeError(
            f"ASSERT_TOPOLOGY: expected one ASSERT, found {len(assertions)}"
        )
    params = assertions[0].params
    if not isinstance(params, Mapping):
        raise QueryDualProbeError("ASSERT_PARAMS: params is not a mapping")
    try:
        M = int(params["M"])
        C_tensor = torch.as_tensor(params["C"]).detach()
        threshold_tensor = torch.as_tensor(params["thresholds"]).detach()
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryDualProbeError(
            f"ASSERT_PARAMS: malformed C/thresholds/M: {exc}"
        ) from exc
    if (
        M <= 0
        or C_tensor.dim() != 2
        or int(C_tensor.shape[0]) != M
        or threshold_tensor.dim() != 2
        or tuple(threshold_tensor.shape) != (1, M)
    ):
        raise QueryDualProbeError(
            "ASSERT_SHAPE: Gate-1 requires one lane with C=[M,n_out] and "
            "thresholds=[1,M]"
        )
    C = np.ascontiguousarray(
        C_tensor.to(device="cpu", dtype=torch.float64).numpy(),
        dtype=np.float64,
    )
    thresholds = np.ascontiguousarray(
        threshold_tensor.to(device="cpu", dtype=torch.float64)
        .numpy()
        .reshape(-1),
        dtype=np.float64,
    )
    if (
        C.shape[1] <= 0
        or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(thresholds))
    ):
        raise QueryDualProbeError(
            "ASSERT_NUMERIC: C/thresholds are empty or non-finite"
        )
    return C, thresholds


def _pipeline_protocol(config: QueryDualProbeConfig) -> str:
    return _PIPELINE_V3 if config.stage_quotas is not None else _PIPELINE_V2


def _quotas_for_receipt(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return {"invalid_repr": repr(value)[:500]}
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return {"invalid_repr": repr(value)[:500]}


def _source_hashes(protocol: str) -> Dict[str, str]:
    import importlib

    if protocol == _PIPELINE_V2:
        modules = _SOURCE_MODULES_V2
    elif protocol == _PIPELINE_V3:
        modules = _SOURCE_MODULES_V3
    else:
        raise QueryDualProbeError(
            f"SOURCE_HASH: unknown pipeline protocol {protocol!r}"
        )
    result = {"query_dual_probe": _sha256_file(Path(__file__).resolve())}
    for name in modules:
        module = importlib.import_module(name)
        source = getattr(module, "__file__", None)
        if not source:
            raise QueryDualProbeError(
                f"SOURCE_HASH: module {name} has no source path"
            )
        result[name] = _sha256_file(Path(source).resolve())
    return result


def _root_property_upper(
    bundle: Any,
    net: Any,
    rows: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Evaluate the independent outward-root interval property upper."""

    from act.back_end.hybridz_tf import query_dual_pipeline as pipeline

    by_id, predecessors = pipeline._layer_maps(net)
    output_lid = pipeline._assert_output_id(by_id, predecessors)
    lower, upper = pipeline._flat_box(
        bundle.root_certificate.bounds[output_lid],
        lid=output_lid,
    )
    if rows.shape[1] != lower.size:
        raise QueryDualProbeError(
            "PROPERTY_BASELINE: root output width differs from property"
        )
    baseline = (
        np.maximum(rows, 0.0) @ upper
        + np.minimum(rows, 0.0) @ lower
        - thresholds
    )
    baseline = np.ascontiguousarray(baseline, dtype=np.float64)
    if not np.all(np.isfinite(baseline)):
        raise QueryDualProbeError(
            "PROPERTY_BASELINE: non-finite root property upper"
        )
    return baseline


def _deadline_check(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise QueryDualProbeError(
            f"DEADLINE_EXPIRED: budget exhausted during {stage}"
        )


def _seconds_from_hex(value: Any) -> float:
    if not isinstance(value, str):
        raise QueryDualProbeError("RECEIPT_TIMING: missing hex duration")
    result = float.fromhex(value)
    if not math.isfinite(result) or result < 0.0:
        raise QueryDualProbeError("RECEIPT_TIMING: invalid duration")
    return result


def _stage_diagnostics(bundle: Any) -> Tuple[list, Dict[str, Any]]:
    stages = []
    for stage in bundle.stages:
        candidate = dict(stage.candidate_receipt)
        replay_receipts = [
            dict(block.replay_receipt) for block in stage.blocks
        ]
        stages.append(
            {
                "stage_index": int(stage.stage_index),
                "target_relu_id": int(stage.target_relu_lid),
                "status": str(stage.status),
                "strict_improvements": int(stage.strict_improvements),
                "candidate_seconds": float(
                    candidate.get("elapsed_seconds", 0.0)
                ),
                "candidate_block_timings": list(
                    candidate.get("timings", [])
                ),
                "independent_replay_seconds": float(
                    sum(
                        _seconds_from_hex(receipt["elapsed_s_hex"])
                        for receipt in replay_receipts
                    )
                ),
                "stage_receipt": dict(stage.receipt),
                "candidate_receipt": candidate,
                "replay_receipts": replay_receipts,
            }
        )
    property_stage = bundle.property_stage
    property_candidate = dict(property_stage.candidate_receipt)
    property_replays = [
        dict(block.replay_receipt) for block in property_stage.blocks
    ]
    property_diagnostic = {
        "candidate_seconds": float(
            property_candidate.get("elapsed_seconds", 0.0)
        ),
        "candidate_block_timings": list(
            property_candidate.get("timings", [])
        ),
        "independent_replay_seconds": float(
            sum(
                _seconds_from_hex(receipt["elapsed_s_hex"])
                for receipt in property_replays
            )
        ),
        "property_receipt": dict(property_stage.receipt),
        "candidate_receipt": property_candidate,
        "replay_receipts": property_replays,
    }
    return stages, property_diagnostic


def run_query_dual_probe(
    config: QueryDualProbeConfig,
    *,
    spec_loader: Any = _UNSET,
    synthesizer: Any = _UNSET,
    converter_factory: Any = _UNSET,
    feedback_builder: Any = _UNSET,
    feedback_validator: Any = _UNSET,
    cuda_initializer: Any = _UNSET,
    cuda_memory_reader: Any = _UNSET,
) -> Dict[str, Any]:
    """Run the strict probe and always attempt an atomic JSON receipt."""

    started = time.monotonic()
    injected_hooks = (
        spec_loader,
        synthesizer,
        converter_factory,
        feedback_builder,
        feedback_validator,
        cuda_initializer,
        cuda_memory_reader,
    )
    production_dependencies = _production_dependency_policy(*injected_hooks)
    pipeline_protocol = _pipeline_protocol(config)
    stage = "config"
    phase_times: Dict[str, float] = {}
    input_before: Dict[str, str] = {}
    input_after: Dict[str, str] = {}
    source_before: Dict[str, str] = {}
    source_after: Dict[str, str] = {}
    cuda_environment: Dict[str, Any] = {}
    cuda_memory: Dict[str, Any] = {}
    normalized: Optional[QueryDualProbeConfig] = None
    setup_seconds: Optional[float] = None
    transaction_started: Optional[float] = None
    transaction_seconds: Optional[float] = None
    transaction_rss_before: Optional[int] = None
    rss_peak_bytes: Optional[int] = None
    payload: Dict[str, Any]
    try:
        normalized = _validate_config(config)
        stage = "input_hash_before"
        phase_started = time.monotonic()
        input_before = {
            "onnx": _sha256_file(normalized.onnx_path),
            "vnnlib": _sha256_file(normalized.vnnlib_path),
        }
        phase_times[stage] = time.monotonic() - phase_started

        stage = "source_hash_before"
        phase_started = time.monotonic()
        pipeline_protocol = _pipeline_protocol(normalized)
        source_before = _source_hashes(pipeline_protocol)
        phase_times[stage] = time.monotonic() - phase_started

        if cuda_initializer is _UNSET:
            cuda_initializer = _initialize_cuda_f64
        if cuda_memory_reader is _UNSET:
            cuda_memory_reader = _cuda_memory_snapshot
        stage = "cuda_initialize"
        phase_started = time.monotonic()
        cuda_environment = dict(cuda_initializer(normalized.device))
        phase_times[stage] = time.monotonic() - phase_started
        if (
            cuda_environment.get("status") != "tracking"
            or cuda_environment.get("device") != "cuda"
            or cuda_environment.get("dtype") != "float64"
        ):
            raise QueryDualProbeError(
                "CUDA_TRACKING: initialization did not establish CUDA f64"
            )

        if any(
            value is _UNSET
            for value in (spec_loader, synthesizer, converter_factory)
        ):
            from act.front_end.model_synthesis import (
                synthesize_models_from_specs,
            )
            from act.front_end.vnnlib_loader.create_specs import (
                create_specs_from_paths,
            )
            from act.pipeline.verification.torch2act import TorchToACT

            if spec_loader is _UNSET:
                spec_loader = create_specs_from_paths
            if synthesizer is _UNSET:
                synthesizer = synthesize_models_from_specs
            if converter_factory is _UNSET:
                converter_factory = lambda model: TorchToACT(model)
        if any(
            value is _UNSET
            for value in (feedback_builder, feedback_validator)
        ):
            from act.back_end.hybridz_tf.query_dual_pipeline import (
                build_verified_query_dual_feedback,
                validate_verified_query_dual_feedback,
            )

            if feedback_builder is _UNSET:
                if pipeline_protocol == _PIPELINE_V3:
                    from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
                        build_verified_query_dual_feedback_v3,
                    )

                    feedback_builder = build_verified_query_dual_feedback_v3
                else:
                    feedback_builder = build_verified_query_dual_feedback
            if feedback_validator is _UNSET:
                feedback_validator = validate_verified_query_dual_feedback

        stage = "parse_convert_specs"
        phase_started = time.monotonic()
        spec_result = spec_loader(
            str(normalized.onnx_path),
            str(normalized.vnnlib_path),
            category="query_dual_gate1_probe",
        )
        phase_times[stage] = time.monotonic() - phase_started

        stage = "synthesis"
        phase_started = time.monotonic()
        wrapped = synthesizer([spec_result])
        if not isinstance(wrapped, Mapping) or len(wrapped) != 1:
            raise QueryDualProbeError(
                "SYNTHESIS_DISJUNCTION: expected exactly one wrapped model"
            )
        model = _move_model_cuda_f64(next(iter(wrapped.values())))
        phase_times[stage] = time.monotonic() - phase_started

        stage = "torch2act"
        phase_started = time.monotonic()
        converter = converter_factory(model)
        run = getattr(converter, "run", None)
        if not callable(run):
            raise QueryDualProbeError(
                "TORCH2ACT_FACTORY: converter has no run()"
            )
        net = run()
        phase_times[stage] = time.monotonic() - phase_started

        stage = "assert_extract"
        phase_started = time.monotonic()
        C, thresholds = _extract_assert_property(net)
        phase_times[stage] = time.monotonic() - phase_started

        stage = "query_dual_transaction"
        # --time applies only to the authority transaction builder.  Parsing,
        # synthesis, conversion, external validation, CUDA diagnostics, and
        # hashing cannot consume query-transaction budget.
        transaction_started = time.monotonic()
        transaction_rss_before = _peak_rss_bytes()
        setup_seconds = transaction_started - started
        deadline = transaction_started + normalized.time_limit
        phase_started = transaction_started
        builder_kwargs: Dict[str, Any] = {
            "target_relu_ids": normalized.target_relu_ids,
            "steps": normalized.steps,
            "block_size": normalized.block_size,
            "replay_chunk_size": normalized.block_size,
            "replay_max_workspace_bytes": _REPLAY_MAX_WORKSPACE_BYTES,
            "candidate_device": "cuda",
            "deadline": deadline,
        }
        if pipeline_protocol == _PIPELINE_V3:
            builder_kwargs.update(
                {
                    "stage_quotas": normalized.stage_quotas,
                    "selector_time_limit": normalized.selector_time_limit,
                    "selector_max_adjoint_cells": (
                        normalized.selector_max_adjoint_cells
                    ),
                    "selector_pool_per_rival": (
                        normalized.selector_pool_per_rival
                    ),
                }
            )
        bundle = feedback_builder(net, C, thresholds, **builder_kwargs)
        phase_times[stage] = time.monotonic() - phase_started
        _deadline_check(deadline, stage)
        transaction_seconds = phase_times[stage]

        stage = "pipeline_protocol_binding"
        expected_pipeline_schema = (
            "act.verified_query_dual_feedback.v3"
            if pipeline_protocol == _PIPELINE_V3
            else "act.verified_query_dual_feedback.v2"
        )
        returned_receipt = getattr(bundle, "receipt", None)
        returned_schema = (
            returned_receipt.get("schema")
            if isinstance(returned_receipt, Mapping)
            else None
        )
        if (
            not isinstance(returned_receipt, Mapping)
            or returned_schema != expected_pipeline_schema
        ):
            raise QueryDualProbeError(
                "PIPELINE_PROTOCOL_BINDING: builder returned "
                f"{returned_schema!r}, expected "
                f"{expected_pipeline_schema!r}"
            )

        stage = "live_validation"
        phase_started = time.monotonic()
        valid = feedback_validator(
            bundle,
            net=net,
            property_rows=C,
            thresholds=thresholds,
            expected_target_relu_ids=normalized.target_relu_ids,
            require_live_provenance=True,
        )
        phase_times[stage] = time.monotonic() - phase_started
        if valid is not True:
            raise QueryDualProbeError(
                "LIVE_VALIDATION: completed bundle failed live validation"
            )
        stage = "transaction_stop_loss"
        # V3 has two deliberately distinct limits: a 12-second hard
        # transaction deadline and a 10.5-second promotion margin.  A sound
        # V3 transaction in the interval (10.5, 12] remains useful
        # diagnostic evidence, but must be reported as not promoted.  Keep
        # the legacy V2 stop-loss behavior unchanged.
        if (
            pipeline_protocol == _PIPELINE_V2
            and transaction_seconds > _TRANSACTION_STOP_LOSS_SECONDS
        ):
            raise QueryDualProbeError(
                "TRANSACTION_STOP_LOSS: query-dual builder took "
                f"{transaction_seconds:.9f}s, exceeding "
                f"{_TRANSACTION_STOP_LOSS_SECONDS:.1f}s"
            )

        stage = "input_hash_after"
        phase_started = time.monotonic()
        input_after = {
            "onnx": _sha256_file(normalized.onnx_path),
            "vnnlib": _sha256_file(normalized.vnnlib_path),
        }
        phase_times[stage] = time.monotonic() - phase_started
        if input_after != input_before:
            raise QueryDualProbeError(
                "INPUT_TOCTOU: ONNX/VNNLIB hashes changed during probe"
            )

        stage = "source_hash_after"
        phase_started = time.monotonic()
        source_after = _source_hashes(pipeline_protocol)
        phase_times[stage] = time.monotonic() - phase_started
        if source_after != source_before:
            raise QueryDualProbeError(
                "SOURCE_TOCTOU: proof-path source hashes changed during probe"
            )

        stage = "cuda_memory"
        phase_started = time.monotonic()
        cuda_memory = dict(cuda_memory_reader())
        phase_times[stage] = time.monotonic() - phase_started
        _validate_cuda_memory(cuda_memory, cuda_environment)
        if pipeline_protocol == _PIPELINE_V3 and (
            cuda_memory["peak_allocated_bytes"]
            > _V3_CUDA_MEMORY_STOP_LOSS_BYTES
            or cuda_memory["peak_reserved_bytes"]
            > _V3_CUDA_MEMORY_STOP_LOSS_BYTES
        ):
            raise QueryDualProbeError(
                "CUDA_MEMORY_STOP_LOSS: V3 peak allocation/reservation "
                f"exceeded {_V3_CUDA_MEMORY_STOP_LOSS_BYTES} bytes"
            )
        rss_peak_bytes = _peak_rss_bytes()
        rss_increment = max(
            0, rss_peak_bytes - int(transaction_rss_before)
        )
        if (
            pipeline_protocol == _PIPELINE_V3
            and rss_increment > _V3_CPU_RSS_INCREMENT_STOP_LOSS_BYTES
        ):
            raise QueryDualProbeError(
                "CPU_RSS_STOP_LOSS: V3 transaction increment exceeded "
                f"{_V3_CPU_RSS_INCREMENT_STOP_LOSS_BYTES} bytes"
            )

        stage = "receipt_assembly"
        phase_started = time.monotonic()
        stage_rows, property_diagnostic = _stage_diagnostics(bundle)
        property_upper = np.ascontiguousarray(
            np.asarray(bundle.property_upper, dtype=np.float64).reshape(-1)
        )
        property_hash = _array_digest(property_upper)
        if property_hash != bundle.receipt.get("property_upper_sha256"):
            raise QueryDualProbeError(
                "PROPERTY_HASH: bundle property upper hash mismatch"
            )
        root_property_upper = _root_property_upper(
            bundle, net, C, thresholds
        )
        property_strict_improvements = int(
            np.count_nonzero(property_upper < root_property_upper)
        )
        property_strict_regressions = int(
            np.count_nonzero(property_upper > root_property_upper)
        )
        property_equal = int(
            property_upper.size
            - property_strict_improvements
            - property_strict_regressions
        )
        target_strict_improvements = int(
            sum(int(row["strict_improvements"]) for row in stage_rows)
        )
        nonempty_stage_results = []
        for row in stage_rows:
            selected_count = int(
                row["candidate_receipt"].get(
                    "selected_target_count",
                    row["candidate_receipt"].get("target_rows", 0),
                )
            )
            nonempty_stage_results.append(
                {
                    "stage_index": int(row["stage_index"]),
                    "target_relu_id": int(row["target_relu_id"]),
                    "selected_target_count": selected_count,
                    "strict_improvements": int(
                        row["strict_improvements"]
                    ),
                    "required": selected_count > 0,
                    "pass": (
                        selected_count == 0
                        or int(row["strict_improvements"]) >= 1
                    ),
                }
            )
        promotion_applicable = pipeline_protocol == _PIPELINE_V3
        promotion_pass = bool(
            not promotion_applicable
            or (
                transaction_seconds <= _TRANSACTION_STOP_LOSS_SECONDS
                and target_strict_improvements >= 20
                and all(
                    row["pass"] for row in nonempty_stage_results
                )
                and property_strict_improvements >= 20
            )
        )
        promotion_gate = {
            "applicable": promotion_applicable,
            "pass": promotion_pass,
            "transaction_seconds": float(transaction_seconds),
            "hard_limit_seconds": _TRANSACTION_HARD_LIMIT_SECONDS,
            "hard_limit_pass": (
                transaction_seconds <= _TRANSACTION_HARD_LIMIT_SECONDS
            ),
            "maximum_transaction_seconds": (
                _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "maximum_promotion_seconds": (
                _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "target_strict_improvements": target_strict_improvements,
            "minimum_target_strict_improvements": (
                20 if promotion_applicable else None
            ),
            "nonempty_target_stages": nonempty_stage_results,
            "property_strict_improvements": (
                property_strict_improvements
            ),
            "minimum_property_strict_improvements": (
                20 if promotion_applicable else None
            ),
        }
        phase_times[stage] = time.monotonic() - phase_started
        payload = {
            "schema": (
                _SCHEMA_V3
                if pipeline_protocol == _PIPELINE_V3
                else _SCHEMA_V2
            ),
            "status": (
                "verified"
                if promotion_pass
                else "verified_not_promoted"
            ),
            "proof_authority": production_dependencies,
            "production_dependencies": production_dependencies,
            "test_only": not production_dependencies,
            "diagnostic_only": True,
            "produces_verdict": False,
            "operator_hz_called": False,
            "hz_solver_called": False,
            "config": {
                "onnx_path": str(normalized.onnx_path),
                "vnnlib_path": str(normalized.vnnlib_path),
                "pipeline_protocol": pipeline_protocol,
                "target_relu_ids": list(normalized.target_relu_ids),
                "stage_quotas": (
                    None
                    if normalized.stage_quotas is None
                    else list(normalized.stage_quotas)
                ),
                "steps": normalized.steps,
                "time_semantics": "query_transaction_builder_only",
                "transaction_time_limit_seconds": normalized.time_limit,
                "time_limit_seconds": normalized.time_limit,
                "block_size": normalized.block_size,
                "replay_max_workspace_bytes": (
                    _REPLAY_MAX_WORKSPACE_BYTES
                ),
                "device": normalized.device,
                "dtype": "float64",
                "selector_time_limit": (
                    normalized.selector_time_limit
                ),
                "selector_max_adjoint_cells": (
                    normalized.selector_max_adjoint_cells
                ),
                "selector_pool_per_rival": (
                    normalized.selector_pool_per_rival
                ),
                "overwrite": normalized.overwrite,
            },
            "input_sha256_before": input_before,
            "input_sha256_after": input_after,
            "input_integrity_stable": True,
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "source_sha256": source_after,
            "source_integrity_stable": True,
            "cuda_environment": cuda_environment,
            "cuda_memory": cuda_memory,
            "cuda_memory_stop_loss_bytes": (
                _V3_CUDA_MEMORY_STOP_LOSS_BYTES
                if pipeline_protocol == _PIPELINE_V3
                else None
            ),
            "cpu_parallelism": _cpu_parallelism_snapshot(),
            "cpu_rss": {
                "transaction_before_bytes": transaction_rss_before,
                "process_peak_bytes": rss_peak_bytes,
                "transaction_increment_bytes": rss_increment,
                "maximum_transaction_increment_bytes": (
                    _V3_CPU_RSS_INCREMENT_STOP_LOSS_BYTES
                    if pipeline_protocol == _PIPELINE_V3
                    else None
                ),
                "pass": True,
            },
            "phase_seconds": phase_times,
            "setup_seconds": float(setup_seconds),
            "transaction_seconds": float(transaction_seconds),
            "transaction_hard_limit_seconds": (
                _TRANSACTION_HARD_LIMIT_SECONDS
            ),
            "transaction_hard_limit_pass": (
                transaction_seconds <= _TRANSACTION_HARD_LIMIT_SECONDS
            ),
            "transaction_stop_loss_seconds": (
                _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "transaction_promotion_margin_seconds": (
                _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "transaction_stop_loss_pass": (
                transaction_seconds <= _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "promotion_gate": promotion_gate,
            "validation_seconds": float(phase_times["live_validation"]),
            "stages": stage_rows,
            "property": {
                **property_diagnostic,
                "root_interval_upper_hex": [
                    float(value).hex() for value in root_property_upper
                ],
                "root_interval_upper_sha256": _array_digest(
                    root_property_upper
                ),
                "strict_improvements_from_root": (
                    property_strict_improvements
                ),
                "strict_regressions_from_root": (
                    property_strict_regressions
                ),
                "numerically_equal_to_root": property_equal,
                "upper_hex": [
                    float(value).hex() for value in property_upper
                ],
                "upper_sha256": property_hash,
            },
            "pipeline_receipt": dict(bundle.receipt),
        }
    except Exception as exc:
        if transaction_started is not None and transaction_seconds is None:
            transaction_seconds = time.monotonic() - transaction_started
        if normalized is not None:
            try:
                input_after = {
                    "onnx": _sha256_file(normalized.onnx_path),
                    "vnnlib": _sha256_file(normalized.vnnlib_path),
                }
            except Exception:
                input_after = {}
        if not source_after:
            try:
                phase_started = time.monotonic()
                source_after = _source_hashes(pipeline_protocol)
                phase_times["source_hash_after"] = (
                    time.monotonic() - phase_started
                )
            except Exception as source_exc:
                phase_times.setdefault("source_hash_after", 0.0)
                source_hash_error = {
                    "type": type(source_exc).__name__,
                    "message": str(source_exc)[:1000],
                }
            else:
                source_hash_error = None
        else:
            source_hash_error = None
        try:
            phase_started = time.monotonic()
            if cuda_memory_reader is _UNSET:
                cuda_memory_reader = _cuda_memory_snapshot
            cuda_memory = dict(cuda_memory_reader())
            phase_times["cuda_memory"] = time.monotonic() - phase_started
        except Exception as memory_exc:
            cuda_memory = {
                "status": "error",
                "error_type": type(memory_exc).__name__,
                "error": str(memory_exc)[:1000],
            }
        try:
            rss_peak_bytes = _peak_rss_bytes()
        except Exception:
            rss_peak_bytes = None
        rss_increment = (
            None
            if transaction_rss_before is None or rss_peak_bytes is None
            else max(0, rss_peak_bytes - transaction_rss_before)
        )
        payload = {
            "schema": (
                _SCHEMA_V3
                if pipeline_protocol == _PIPELINE_V3
                else _SCHEMA_V2
            ),
            "status": "error",
            "proof_authority": False,
            "production_dependencies": production_dependencies,
            "test_only": not production_dependencies,
            "diagnostic_only": True,
            "produces_verdict": False,
            "operator_hz_called": False,
            "hz_solver_called": False,
            "failed_stage": stage,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            },
            "config": {
                "onnx_path": str(config.onnx_path),
                "vnnlib_path": str(config.vnnlib_path),
                "pipeline_protocol": pipeline_protocol,
                "target_relu_ids": [
                    int(value) for value in config.target_relu_ids
                ],
                "stage_quotas": (
                    _quotas_for_receipt(config.stage_quotas)
                ),
                "steps": int(config.steps),
                "time_semantics": "query_transaction_builder_only",
                "transaction_time_limit_seconds": float(config.time_limit),
                "time_limit_seconds": float(config.time_limit),
                "block_size": int(config.block_size),
                "replay_max_workspace_bytes": (
                    _REPLAY_MAX_WORKSPACE_BYTES
                ),
                "device": str(config.device),
                "dtype": "float64",
                "selector_time_limit": float(
                    config.selector_time_limit
                ),
                "selector_max_adjoint_cells": int(
                    config.selector_max_adjoint_cells
                ),
                "selector_pool_per_rival": int(
                    config.selector_pool_per_rival
                ),
                "overwrite": bool(config.overwrite),
            },
            "input_sha256_before": input_before,
            "input_sha256_after": input_after,
            "input_integrity_stable": bool(
                input_before
                and input_after
                and input_before == input_after
            ),
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "source_integrity_stable": bool(
                source_before
                and source_after
                and source_before == source_after
            ),
            "cuda_environment": cuda_environment,
            "cuda_memory": cuda_memory,
            "cuda_memory_stop_loss_bytes": (
                _V3_CUDA_MEMORY_STOP_LOSS_BYTES
                if pipeline_protocol == _PIPELINE_V3
                else None
            ),
            "cpu_parallelism": _cpu_parallelism_snapshot(),
            "cpu_rss": {
                "transaction_before_bytes": transaction_rss_before,
                "process_peak_bytes": rss_peak_bytes,
                "transaction_increment_bytes": rss_increment,
                "maximum_transaction_increment_bytes": (
                    _V3_CPU_RSS_INCREMENT_STOP_LOSS_BYTES
                    if pipeline_protocol == _PIPELINE_V3
                    else None
                ),
                "pass": bool(
                    pipeline_protocol != _PIPELINE_V3
                    or (
                        rss_increment is not None
                        and rss_increment
                        <= _V3_CPU_RSS_INCREMENT_STOP_LOSS_BYTES
                    )
                ),
            },
            "phase_seconds": phase_times,
            "setup_seconds": (
                None if setup_seconds is None else float(setup_seconds)
            ),
            "transaction_seconds": (
                None
                if transaction_seconds is None
                else float(transaction_seconds)
            ),
            "transaction_hard_limit_seconds": (
                _TRANSACTION_HARD_LIMIT_SECONDS
            ),
            "transaction_hard_limit_pass": bool(
                transaction_seconds is not None
                and transaction_seconds
                <= _TRANSACTION_HARD_LIMIT_SECONDS
            ),
            "transaction_stop_loss_seconds": (
                _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "transaction_promotion_margin_seconds": (
                _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "transaction_stop_loss_pass": bool(
                transaction_seconds is not None
                and transaction_seconds <= _TRANSACTION_STOP_LOSS_SECONDS
            ),
            "validation_seconds": phase_times.get("live_validation"),
        }
        if source_before and source_after and source_before == source_after:
            payload["source_sha256"] = source_after
        if source_hash_error is not None:
            payload["source_sha256_error"] = {
                **source_hash_error,
            }

    # A provisional canonicalization makes the measured total include one full
    # receipt-finalization pass.  The final pass binds those timing fields.
    payload["receipt_finalization_seconds"] = 0.0
    payload["total_seconds"] = 0.0
    finalization_started = time.monotonic()
    _finalize_receipt(payload)
    finalization_seconds = time.monotonic() - finalization_started
    phase_times["receipt_finalization"] = finalization_seconds
    payload["receipt_finalization_seconds"] = finalization_seconds
    payload["total_seconds"] = float(time.monotonic() - started)
    result = _finalize_receipt(payload)
    _atomic_json(
        normalized.output_path if normalized is not None else config.output_path,
        result,
        overwrite=(
            normalized.overwrite
            if normalized is not None
            else config.overwrite is True
        ),
        forbidden_paths=(config.onnx_path, config.vnnlib_path),
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict CUDA-f64 query-dual Gate-1 diagnostic; produces no "
            "verification verdict."
        )
    )
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--vnnlib", required=True, type=Path)
    parser.add_argument(
        "--targets",
        required=True,
        type=_parse_targets,
        help="comma-separated target ReLU layer ids, in transaction order",
    )
    parser.add_argument(
        "--v3-quotas",
        dest="stage_quotas",
        type=_parse_stage_quotas,
        default=None,
        help=(
            "explicit sparse quota per target; its presence selects sealed "
            "property-sparse V3 (for CIFAR100-medium: 16,8,24,16)"
        ),
    )
    parser.add_argument("--steps", type=_positive_int, default=8)
    parser.add_argument(
        "--time",
        dest="time_limit",
        type=_positive_seconds,
        default=12.0,
        help=(
            "hard seconds budget for feedback_builder only (maximum 12); "
            "setup and external live validation are separately timed "
            "(default: 12)"
        ),
    )
    parser.add_argument(
        "--block",
        dest="block_size",
        type=_positive_int,
        default=1024,
    )
    parser.add_argument(
        "--selector-time",
        dest="selector_time_limit",
        type=_positive_seconds,
        default=1.0,
    )
    parser.add_argument(
        "--selector-max-adjoint-cells",
        type=_positive_int,
        default=30_000_000,
    )
    parser.add_argument(
        "--selector-pool-per-rival",
        type=_positive_int,
        default=64,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing non-input output path",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> QueryDualProbeConfig:
    return QueryDualProbeConfig(
        onnx_path=args.onnx,
        vnnlib_path=args.vnnlib,
        target_relu_ids=tuple(args.targets),
        steps=int(args.steps),
        time_limit=float(args.time_limit),
        block_size=int(args.block_size),
        device=str(args.device),
        output_path=args.output,
        overwrite=bool(args.overwrite),
        stage_quotas=(
            None
            if args.stage_quotas is None
            else tuple(args.stage_quotas)
        ),
        selector_time_limit=float(args.selector_time_limit),
        selector_max_adjoint_cells=int(args.selector_max_adjoint_cells),
        selector_pool_per_rival=int(args.selector_pool_per_rival),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_query_dual_probe(_config_from_args(args))
    except Exception as exc:
        # Atomic-write failures are the only errors which cannot be represented
        # in the requested output file.
        parser.exit(
            2,
            f"query-dual probe could not write its receipt: "
            f"{type(exc).__name__}: {exc}\n",
        )
    return 0 if result.get("status") == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
