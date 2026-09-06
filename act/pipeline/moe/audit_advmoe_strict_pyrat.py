"""Independently audit the directed-rounding AdvMoE PyRAT pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoePath,
    specialize_advmoe_path,
)
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.advmoe_strict_pyrat import (
    aggregate_strict_paths,
    classification_vnnlib,
    parse_pyrat_status,
)
from act.pipeline.moe.advmoe_two_path import _load_model, _predict
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/advmoe_strict_pyrat_seed0_compat_pilot_r3.json"
)


def _issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _command_value(command: list[str], option: str) -> str | None:
    try:
        position = command.index(option)
    except ValueError:
        return None
    if position + 1 >= len(command):
        return None
    return command[position + 1]


def _semantic_pyrat_version(output: str) -> str | None:
    match = re.search(r"\bPyRAT\s+\d+(?:\.\d+)*\b", output)
    return match.group(0) if match is not None else None


def _replay_export(
    onnx_path: Path,
    path: torch.nn.Module,
    probes: torch.Tensor,
    tolerance: float,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = path.float().eval()(probes.float()).cpu().numpy()
    observed = np.concatenate(
        [
            session.run(
                None,
                {session.get_inputs()[0].name: probe.cpu().numpy()},
            )[0]
            for probe in probes.float().split(1, dim=0)
        ],
        axis=0,
    )
    difference = np.abs(observed - expected)
    operators = [node.op_type for node in graph.graph.node]
    return {
        "nodes": len(operators),
        "operators": sorted(set(operators)),
        "batchnormalization_nodes": operators.count("BatchNormalization"),
        "dynamic_dispatch_present": any(
            value in {"ArgMax", "TopK", "Gather", "GatherElements"}
            for value in operators
        ),
        "maximum_abs_error": float(difference.max()),
        "equivalent": bool(
            np.allclose(observed, expected, atol=float(tolerance), rtol=0.0)
        ),
        "predictions_equal": bool(
            np.array_equal(observed.argmax(axis=1), expected.argmax(axis=1))
        ),
    }


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    issues: list[str] = []
    rows_path = output / "rows.jsonl"
    summary_path = output / "summary.json"
    _issue(issues, rows_path.is_file(), "rows.jsonl is missing")
    _issue(issues, summary_path.is_file(), "summary.json is missing")
    if issues:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "issue_count": len(issues),
            "issues": issues,
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(rows_path)
    selected = [int(value) for value in config["selection"]["ordered_dataset_indices"]]
    _issue(issues, len(rows) == len(selected), "row count differs from frozen selection")
    _issue(
        issues,
        summary.get("config", {}).get("sha256") == _sha256(config_path),
        "summary config hash differs",
    )
    _issue(
        issues,
        summary.get("rows", {}).get("sha256") == _sha256(rows_path),
        "summary rows hash differs",
    )
    _issue(
        issues,
        summary.get("selection") == selected,
        "summary selection differs",
    )

    pyrat = config["pyrat"]
    expected_contract = {
        "sound": True,
        "dtype": 64,
        "library": "numpy",
        "device": "cpu",
        "directed_rounding": True,
    }
    _issue(issues, pyrat.get("sound") is True, "PyRAT sound mode is not frozen true")
    _issue(issues, int(pyrat.get("dtype", 0)) == 64, "PyRAT dtype is not 64")
    _issue(issues, pyrat.get("library") == "numpy", "PyRAT library is not numpy")
    _issue(issues, pyrat.get("device") == "cpu", "PyRAT device is not CPU")
    _issue(
        issues,
        pyrat.get("directed_rounding") is True,
        "directed-rounding contract is absent",
    )
    version = subprocess.run(
        [str(Path(pyrat["executable"])), "-V"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    version_semantic = _semantic_pyrat_version(version)
    runtime_version_semantic = _semantic_pyrat_version(
        str(summary.get("pyrat", {}).get("version_stdout", ""))
    )
    _issue(
        issues,
        version_semantic == str(pyrat["version_expected"]),
        f"PyRAT version differs: {version!r}",
    )
    _issue(
        issues,
        runtime_version_semantic == version_semantic,
        "runtime and audit PyRAT versions differ",
    )

    checkpoint = _inside(Path(config["checkpoint"]["path"]), WRITE_ROOT)
    archive = _inside(Path(config["dataset_archive"]), WRITE_ROOT)
    _issue(
        issues,
        _sha256(checkpoint) == config["checkpoint"]["sha256"],
        "checkpoint hash differs",
    )
    _issue(
        issues,
        _sha256(archive) == config["dataset_archive_sha256"],
        "dataset archive hash differs",
    )
    _issue(
        issues,
        summary.get("checkpoint", {}).get("sha256") == _sha256(checkpoint),
        "summary checkpoint hash differs",
    )
    _issue(
        issues,
        summary.get("dataset", {}).get("sha256") == _sha256(archive),
        "summary dataset hash differs",
    )

    model, _router, moe_type, _checkpoint = _load_model(config, WRITE_ROOT)
    inputs, labels = load_cifar10_test_archive(archive)
    predictions = _predict(model, inputs, int(config["batch_size"]))
    _issue(
        issues,
        bool(np.all(predictions[selected] == labels[selected])),
        "frozen selection is not clean-correct",
    )
    probes = torch.from_numpy(inputs[selected])
    paths: list[torch.nn.Module] = []
    export_replays: list[dict[str, Any]] = []
    for route in range(2):
        literal, _ = specialize_advmoe_path(model, route, moe_type)
        path = CrownCompatibleAdvMoePath(literal).float().eval()
        paths.append(path)
        onnx_path = output / f"advmoe_path{route}.onnx"
        _issue(issues, onnx_path.is_file(), f"route {route} ONNX is missing")
        if not onnx_path.is_file():
            continue
        replay = _replay_export(
            onnx_path,
            path,
            probes,
            float(config["export"]["semantic_atol"]),
        )
        replay.update({"route": route, "path": str(onnx_path), "sha256": _sha256(onnx_path)})
        export_replays.append(replay)
        _issue(issues, replay["equivalent"], f"route {route} ONNX replay differs")
        _issue(
            issues,
            replay["predictions_equal"],
            f"route {route} ONNX prediction differs",
        )
        _issue(
            issues,
            not replay["dynamic_dispatch_present"],
            f"route {route} ONNX retains dynamic dispatch",
        )
        _issue(
            issues,
            replay["batchnormalization_nodes"] > 0,
            f"route {route} ONNX lost BatchNormalization",
        )
        registered_exports = summary.get("exports", [])
        if route < len(registered_exports):
            _issue(
                issues,
                registered_exports[route].get("sha256") == replay["sha256"],
                f"route {route} ONNX hash differs from summary",
            )

    independently_aggregated: list[str] = []
    for slot, (index, row) in enumerate(zip(selected, rows)):
        prefix = f"row {slot}"
        _issue(issues, int(row.get("sample_slot", -1)) == slot, f"{prefix}: slot differs")
        _issue(issues, int(row.get("dataset_index", -1)) == index, f"{prefix}: index differs")
        _issue(
            issues,
            int(row.get("clean_prediction", -1)) == int(predictions[index]),
            f"{prefix}: clean prediction differs",
        )
        epsilon = float(config["epsilon_over_255"]) / 255.0
        center = inputs[index].astype(np.float64)
        lower = np.clip(center - epsilon, 0.0, 1.0)
        upper = np.clip(center + epsilon, 0.0, 1.0)
        expected_property = classification_vnnlib(
            lower, upper, int(predictions[index])
        )
        property_path = _inside(Path(row["property"]), WRITE_ROOT)
        _issue(issues, property_path.is_file(), f"{prefix}: property is missing")
        if property_path.is_file():
            observed_property = property_path.read_text(encoding="utf-8")
            _issue(
                issues,
                observed_property == expected_property,
                f"{prefix}: property does not reconstruct",
            )
            _issue(
                issues,
                row.get("property_sha256") == _sha256(property_path),
                f"{prefix}: property hash differs",
            )
        records = row.get("path_results", [])
        _issue(issues, len(records) == 2, f"{prefix}: path coverage is incomplete")
        statuses: list[str] = []
        for route, record in enumerate(records):
            command = [str(value) for value in record.get("command", [])]
            log_path = _inside(Path(record["log"]), WRITE_ROOT)
            _issue(issues, log_path.is_file(), f"{prefix}/path{route}: log is missing")
            output_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
            parsed = parse_pyrat_status(
                output_text,
                returncode=int(record["returncode"]),
                timed_out=bool(record["timed_out"]),
            )
            statuses.append(parsed)
            _issue(
                issues,
                parsed == record.get("status"),
                f"{prefix}/path{route}: status does not parse",
            )
            _issue(
                issues,
                record.get("soundness_contract") == expected_contract,
                f"{prefix}/path{route}: soundness contract differs",
            )
            expected_options = {
                "--dtype": "64",
                "--library": "numpy",
                "--device": "cpu",
                "--sound": "true",
                "--check": "skip",
            }
            for option, expected in expected_options.items():
                _issue(
                    issues,
                    _command_value(command, option) == expected,
                    f"{prefix}/path{route}: {option} differs",
                )
            _issue(
                issues,
                _command_value(command, "--model_path") == record.get("onnx"),
                f"{prefix}/path{route}: model path differs",
            )
            _issue(
                issues,
                _command_value(command, "--property_path") == row.get("property"),
                f"{prefix}/path{route}: property path differs",
            )
            _issue(
                issues,
                record.get("log_sha256") == _sha256(log_path),
                f"{prefix}/path{route}: log hash differs",
            )
            onnx_path = _inside(Path(record["onnx"]), WRITE_ROOT)
            _issue(
                issues,
                record.get("onnx_sha256") == _sha256(onnx_path),
                f"{prefix}/path{route}: ONNX hash differs",
            )
        endpoint = aggregate_strict_paths(statuses)
        independently_aggregated.append(endpoint)
        _issue(
            issues,
            row.get("endpoint_status") == endpoint,
            f"{prefix}: endpoint aggregation differs",
        )

    counts = dict(sorted(Counter(independently_aggregated).items()))
    _issue(
        issues,
        summary.get("endpoint_counts") == counts,
        "summary endpoint counts do not recompute",
    )
    strict_safe = sum(value == "SAFE_ALL_PATHS_DIRECTED_ROUNDING" for value in independently_aggregated)
    result = {
        "schema_version": 1,
        "classification": "INDEPENDENT_STRICT_BACKEND_PILOT_AUDIT",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "pyrat_version": version_semantic,
        "pyrat_version_raw": version,
        "selection_clean_correct": bool(np.all(predictions[selected] == labels[selected])),
        "exports_replayed": export_replays,
        "rows_audited": len(rows),
        "endpoint_counts": counts,
        "strict_dynamic_safe": strict_safe,
        "backend_feasibility_gate_met": strict_safe >= 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "claim_boundary": (
            "PASS audits identities and fail-closed aggregation. Only "
            "SAFE_ALL_PATHS_DIRECTED_ROUNDING is a strict dynamic-model "
            "certificate; this backend-control cohort is not a prevalence estimate."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-name", default="independent_audit.json")
    args = parser.parse_args()
    result = audit(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    path = output / args.output_name
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    _write_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
