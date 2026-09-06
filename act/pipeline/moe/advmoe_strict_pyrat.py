"""Strict directed-rounding verification of specialized AdvMoE paths.

This module is deliberately small and backend-facing.  It removes dynamic
dispatch by exporting each globally specialized AdvMoE path, checks the ONNX
artifact against the frozen PyTorch path, and asks PyRAT to prove the same
top-1 property for every feasible path.  A dynamic-model SAFE result is only
formed when every path returns ``SAFE`` under PyRAT's CPU directed-rounding
mode.  UNKNOWN, timeout, parser errors, and incomplete path coverage all fail
closed.

PyRAT is an independently installed verifier and is not copied into ACT.  The
portable evidence consists of the checkpoint/config identities, ONNX files,
VNN-LIB properties, complete command lines, logs, and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoePath,
    path_adapter_equivalence,
    specialize_advmoe_path,
)
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.advmoe_two_path import _load_model, _predict
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


_RESULT_PATTERN = re.compile(
    r"\bResult\s*=\s*(SAFE|UNSAFE|UNKNOWN|ERROR|TIMEOUT|INFEASIBLE)\b",
    re.IGNORECASE,
)


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def classification_vnnlib(
    lower: np.ndarray,
    upper: np.ndarray,
    prediction: int,
    *,
    classes: int = 10,
) -> str:
    """Return a VNN-LIB safety property for one CHW input box.

    PyRAT's property parser consumes the desired output specification and
    internally negates it for verification.  We therefore assert every clean
    class margin directly.  All decimal bounds are emitted from binary64 with
    enough significant digits for round-trip recovery.
    """

    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    if lo.shape != hi.shape or lo.size == 0:
        raise ValueError("input bounds must be non-empty and shape matched")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("input bounds must be finite")
    if np.any(lo > hi):
        raise ValueError("input lower bound exceeds upper bound")
    if not 0 <= int(prediction) < int(classes):
        raise ValueError("prediction outside output range")

    lines = [f"(declare-const X_{index} Real)" for index in range(lo.size)]
    lines.extend(f"(declare-const Y_{index} Real)" for index in range(classes))
    for index, (minimum, maximum) in enumerate(zip(lo.tolist(), hi.tolist())):
        lines.append(f"(assert (>= X_{index} {minimum:.17g}))")
        lines.append(f"(assert (<= X_{index} {maximum:.17g}))")
    for competitor in range(classes):
        if competitor == int(prediction):
            continue
        lines.append(
            f"(assert (>= (- Y_{int(prediction)} Y_{competitor}) 0))"
        )
    return "\n".join(lines) + "\n"


def parse_pyrat_status(output: str, *, returncode: int, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    matches = [value.upper() for value in _RESULT_PATTERN.findall(output)]
    if returncode != 0:
        return "ERROR"
    if not matches:
        return "ERROR"
    if len(set(matches)) != 1:
        return "ERROR"
    return matches[0]


def aggregate_strict_paths(path_statuses: Iterable[str]) -> str:
    statuses = tuple(str(value).upper() for value in path_statuses)
    if len(statuses) != 2:
        return "UNKNOWN_INCOMPLETE_PATH_COVERAGE"
    if all(value == "SAFE" for value in statuses):
        return "SAFE_ALL_PATHS_DIRECTED_ROUNDING"
    if any(value == "UNSAFE" for value in statuses):
        # A static-path counterexample is not automatically reachable through
        # the dynamic router.  It cannot be promoted without full-model replay.
        return "UNKNOWN_STATIC_PATH_COUNTEREXAMPLE_NOT_LIFTED"
    if any(value == "TIMEOUT" for value in statuses):
        return "TIMEOUT"
    return "UNKNOWN"


def _export_path(
    path: torch.nn.Module,
    probes: torch.Tensor,
    output: Path,
    *,
    opset: int,
    semantic_atol: float,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    path = path.float().eval()
    probes = probes.float()
    with torch.no_grad():
        expected = path(probes).cpu().numpy()
    torch.onnx.export(
        path,
        probes[:1],
        str(output),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=int(opset),
        dynamo=False,
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    observed = np.concatenate(
        [
            session.run(
                None,
                {session.get_inputs()[0].name: probe.cpu().numpy()},
            )[0]
            for probe in probes.split(1, dim=0)
        ],
        axis=0,
    )
    maximum_error = float(np.max(np.abs(observed - expected)))
    equivalent = bool(
        np.allclose(observed, expected, atol=float(semantic_atol), rtol=0.0)
    )
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "nodes": len(graph.graph.node),
        "operators": sorted({node.op_type for node in graph.graph.node}),
        "dynamic_dispatch_present": any(
            node.op_type in {"ArgMax", "TopK", "Gather", "GatherElements"}
            for node in graph.graph.node
        ),
        "onnxruntime_semantic_equivalent": equivalent,
        "onnxruntime_maximum_abs_error": maximum_error,
    }


def _pyrat_call(
    executable: Path,
    onnx_path: Path,
    property_path: Path,
    *,
    domains: list[str],
    timeout_seconds: float,
    environment: dict[str, str],
) -> dict[str, Any]:
    command = [
        str(executable),
        "--model_path",
        str(onnx_path),
        "--property_path",
        str(property_path),
        "--domains",
        *domains,
        "--dtype",
        "64",
        "--library",
        "numpy",
        "--device",
        "cpu",
        "--sound",
        "true",
        "--check",
        "skip",
        "--verbose",
        "false",
        "--timeout",
        str(float(timeout_seconds)),
    ]
    started = time.monotonic()
    timed_out = False
    try:
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_seconds) + 30.0,
            env=environment,
            check=False,
        )
        output = process.stdout
        returncode = int(process.returncode)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        output = (error.stdout or "") if isinstance(error.stdout, str) else ""
        returncode = -1
    return {
        "status": parse_pyrat_status(
            output, returncode=returncode, timed_out=timed_out
        ),
        "returncode": returncode,
        "timed_out": timed_out,
        "seconds": time.monotonic() - started,
        "command": command,
        "stdout": output,
        "soundness_contract": {
            "sound": True,
            "dtype": 64,
            "library": "numpy",
            "device": "cpu",
            "directed_rounding": True,
        },
    }


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace_boundary"])
    repository = _inside(Path(config["act_repository"]), workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)
    archive = _inside(Path(config["dataset_archive"]), workspace)
    executable = _inside(Path(config["pyrat"]["executable"]), Path("/data1/Kane"))
    config_path = _inside(config_path, workspace)
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("configuration is not frozen")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if _git(repository, "branch", "--show-current") != config["required_branch"]:
        raise RuntimeError("branch gate failed")
    if _git(repository, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree must be clean")
    required_head = str(config["required_head"])
    ancestry = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", required_head, "HEAD"],
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("required implementation commit is not an ancestor of HEAD")
    if _sha256(archive) != config["dataset_archive_sha256"]:
        raise RuntimeError("dataset archive hash mismatch")
    if not executable.is_file():
        raise FileNotFoundError(executable)

    output_dir.mkdir(parents=True)
    rows_path = output_dir / "rows.jsonl"
    model, _router, moe_type, checkpoint = _load_model(config, workspace)
    inputs, labels = load_cifar10_test_archive(archive)
    predictions = _predict(model, inputs, int(config["batch_size"]))
    selected = np.asarray(config["selection"]["ordered_dataset_indices"], dtype=np.int64)
    if selected.size == 0 or len(set(selected.tolist())) != len(selected):
        raise ValueError("selection must be non-empty and unique")
    if not np.all(predictions[selected] == labels[selected]):
        raise RuntimeError("selection contains a non-clean-correct input")

    specialized: list[torch.nn.Module] = []
    for route in range(2):
        literal, _count = specialize_advmoe_path(model, route, moe_type)
        adapted = CrownCompatibleAdvMoePath(literal).eval()
        equivalence = path_adapter_equivalence(
            literal,
            torch.from_numpy(inputs[selected]),
            atol=float(config["export"]["semantic_atol"]),
            rtol=0.0,
        )
        if not equivalence["outputs_close"] or not equivalence["predictions_equal"]:
            raise RuntimeError("specialized path adapter mismatch")
        specialized.append(adapted)

    probe_tensor = torch.from_numpy(inputs[selected])
    exports = []
    for route, path in enumerate(specialized):
        record = _export_path(
            path,
            probe_tensor,
            output_dir / f"advmoe_path{route}.onnx",
            opset=int(config["export"]["opset"]),
            semantic_atol=float(config["export"]["semantic_atol"]),
        )
        if not record["onnxruntime_semantic_equivalent"]:
            raise RuntimeError("ONNX export semantic mismatch")
        if record["dynamic_dispatch_present"]:
            raise RuntimeError("specialized path retains dynamic dispatch")
        exports.append(record)

    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(int(config["pyrat"]["threads"]))
    environment["MKL_NUM_THREADS"] = str(int(config["pyrat"]["threads"]))
    started = time.monotonic()
    rows = []
    with rows_path.open("x", encoding="utf-8") as handle:
        for slot, index in enumerate(selected.tolist()):
            epsilon = float(config["epsilon_over_255"]) / 255.0
            center = inputs[index].astype(np.float64)
            lower = np.clip(center - epsilon, 0.0, 1.0)
            upper = np.clip(center + epsilon, 0.0, 1.0)
            property_path = output_dir / f"sample{slot}_eps{config['epsilon_over_255']}.vnnlib"
            _write_text(
                property_path,
                classification_vnnlib(lower, upper, int(predictions[index])),
            )
            path_records = []
            for route, export in enumerate(exports):
                call = _pyrat_call(
                    executable,
                    Path(export["path"]),
                    property_path,
                    domains=[str(value) for value in config["pyrat"]["domains"]],
                    timeout_seconds=float(config["pyrat"]["timeout_seconds"]),
                    environment=environment,
                )
                log_path = output_dir / f"sample{slot}_path{route}.log"
                _write_text(log_path, call.pop("stdout"))
                call.update(
                    {
                        "route": route,
                        "log": str(log_path),
                        "log_sha256": _sha256(log_path),
                        "onnx": export["path"],
                        "onnx_sha256": export["sha256"],
                    }
                )
                path_records.append(call)
            row = {
                "row_id": f"sample{slot}:eps{config['epsilon_over_255']}",
                "sample_slot": slot,
                "dataset_index": index,
                "label": int(labels[index]),
                "clean_prediction": int(predictions[index]),
                "epsilon_over_255": float(config["epsilon_over_255"]),
                "property": str(property_path),
                "property_sha256": _sha256(property_path),
                "path_results": path_records,
                "endpoint_status": aggregate_strict_paths(
                    value["status"] for value in path_records
                ),
                "negative_semantics": (
                    "UNSAFE from a static path is UNKNOWN until a full dynamic-model "
                    "witness is replayed"
                ),
            }
            _append_json(handle, row)
            rows.append(row)

    summary = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "repository": {
            "branch": _git(repository, "branch", "--show-current"),
            "head": _git(repository, "rev-parse", "HEAD"),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "dataset": {"path": str(archive), "sha256": _sha256(archive)},
        "model_clean_accuracy_percent": float(
            100.0 * np.mean(predictions == labels)
        ),
        "selection": selected.tolist(),
        "exports": exports,
        "rows": {
            "path": str(rows_path),
            "sha256": _sha256(rows_path),
            "count": len(rows),
        },
        "endpoint_counts": {
            key: sum(row["endpoint_status"] == key for row in rows)
            for key in sorted({row["endpoint_status"] for row in rows})
        },
        "pyrat": {
            "version_stdout": subprocess.run(
                [str(executable), "-V"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                env=environment,
            ).stdout.strip(),
            "executable": str(executable),
            "soundness_contract": (
                "CPU numpy float64 with PyRAT --sound true directed rounding"
            ),
        },
        "runtime_seconds": time.monotonic() - started,
        "interpretation": (
            "Only SAFE_ALL_PATHS_DIRECTED_ROUNDING is a strict dynamic-model "
            "certificate; every other status fails closed."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
