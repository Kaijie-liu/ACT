"""Independently audit the frozen four-cell AdvMoE backend check."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.advmoe_backend_consistency import compare_cells
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


WORKSPACE = Path("/data1/Kane/MOE")


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(WORKSPACE)
    return resolved


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            rtol=0.0,
            atol=float(tolerance),
        )
    )


def _graph_issues(cell: dict[str, Any]) -> list[str]:
    issues = []
    graph = cell["bound"].get("graph_metadata")
    if not isinstance(graph, dict):
        if cell["bound"].get("status") == "ERROR":
            return issues
        return [f"missing graph metadata: {cell['cell_id']}"]
    nodes = graph.get("nodes", [])
    if int(graph.get("node_count", -1)) != len(nodes):
        issues.append(f"node count mismatch: {cell['cell_id']}")
    if sum(graph.get("operation_histogram", {}).values()) != len(nodes):
        issues.append(f"operation histogram mismatch: {cell['cell_id']}")
    expected_router = cell["graph_form"] == "compiled_mu0"
    if bool(graph.get("router_parameters_present_in_lowered_graph")) != expected_router:
        issues.append(f"lowered router-presence mismatch: {cell['cell_id']}")
    names = graph.get("source_router_parameter_names", [])
    if bool(names) != expected_router:
        issues.append(f"source router-presence mismatch: {cell['cell_id']}")
    return issues


def _trajectory_issues(cell: dict[str, Any]) -> list[str]:
    issues = []
    bound = cell["bound"]
    trajectory = bound.get("optimization_trajectory")
    if not isinstance(trajectory, dict):
        if bound.get("status") == "ERROR":
            return issues
        return [f"missing optimization trajectory: {cell['cell_id']}"]
    returned = bound.get("lower_bounds", [])
    if not _close(trajectory.get("returned_lower_bounds", []), returned):
        issues.append(f"trajectory returned-bound mismatch: {cell['cell_id']}")
    if cell["backend"] == "plain_crown":
        if bool(trajectory.get("trace_available")) or int(
            trajectory.get("trace_points", -1)
        ) != 0:
            issues.append(f"unexpected plain-CROWN trace: {cell['cell_id']}")
        for name in (
            "initial_lower_bounds",
            "best_observed_lower_bounds",
            "last_iteration_lower_bounds",
        ):
            if not _close(trajectory.get(name, []), returned):
                issues.append(f"plain-CROWN {name} mismatch: {cell['cell_id']}")
    else:
        trace = trajectory.get("lower_bound_trace", [])
        if not bool(trajectory.get("trace_available")) or len(trace) != int(
            trajectory.get("trace_points", -1)
        ):
            issues.append(f"optimized trace identity mismatch: {cell['cell_id']}")
        elif trace:
            values = np.asarray(trace, dtype=np.float64)
            if not _close(trajectory["initial_lower_bounds"], values[0]):
                issues.append(f"optimized initial bound mismatch: {cell['cell_id']}")
            if not _close(trajectory["last_iteration_lower_bounds"], values[-1]):
                issues.append(f"optimized final-iterate mismatch: {cell['cell_id']}")
            if not _close(trajectory["best_observed_lower_bounds"], values.max(axis=0)):
                issues.append(f"optimized best-observed mismatch: {cell['cell_id']}")
            keep_best = (
                bound.get("backend_configuration", {}).get("optimized_method")
                and cell["backend"] == "sparse_alpha"
            )
            if keep_best and len(returned) == 1 and not _close(
                returned, trajectory["best_observed_lower_bounds"]
            ):
                issues.append(f"returned bound did not preserve observed best: {cell['cell_id']}")
    return issues


def audit(config_path: Path, summary_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path)
    summary_path = _inside(summary_path)
    config = _json(config_path)
    summary = _json(summary_path)
    issues: list[str] = []
    if summary.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("summary/config hash mismatch")
    for name, identity in config["inputs"].items():
        path = _inside(Path(identity["path"]))
        if _sha256(path) != identity["sha256"]:
            issues.append(f"input identity mismatch: {name}")
    if summary.get("scope") != config.get("scope"):
        issues.append("scope mismatch")
    if summary.get("holdout") != "LOCKED_NOT_ACCESSED":
        issues.append("holdout gate mismatch")
    if bool(summary.get("formal_safe_enabled")):
        issues.append("formal SAFE must remain disabled")
    if summary.get("negative_bound_semantics") != "UNKNOWN_NEVER_UNSAFE":
        issues.append("negative-bound semantics mismatch")
    obligation = summary.get("obligation", {})
    for name, expected in config["obligation"].items():
        if obligation.get(name) != expected:
            issues.append(f"obligation mismatch: {name}")
    cells = summary.get("cells", [])
    if [cell.get("cell_id") for cell in cells] != config["execution_order"]:
        issues.append("four-cell execution order mismatch")
    for cell in cells:
        backend = config["backends"].get(cell.get("backend"))
        if backend is None:
            issues.append(f"unknown backend: {cell.get('cell_id')}")
            continue
        bound = cell.get("bound", {})
        if bound.get("method") != backend["method"]:
            issues.append(f"backend method mismatch: {cell['cell_id']}")
        if bool(bound.get("gradient_tracking_enabled")) != bool(
            backend["gradient_tracking"]
        ):
            issues.append(f"gradient setting mismatch: {cell['cell_id']}")
        if cell.get("intermediate_bound_strategy") != backend[
            "intermediate_bound_strategy"
        ]:
            issues.append(f"intermediate strategy mismatch: {cell['cell_id']}")
        if bound.get("negative_bound_semantics") != "UNKNOWN_NEVER_UNSAFE":
            issues.append(f"cell negative semantics mismatch: {cell['cell_id']}")
        if bound.get("status") == "CERTIFIED_MARGIN_FILTER":
            lower = np.asarray(bound.get("lower_bounds", []), dtype=np.float64)
            if not bool(
                lower.size == 1
                and np.isfinite(lower).all()
                and lower[0] >= float(config["numerical"]["safe_positive_margin"])
            ):
                issues.append(f"positive-filter rule mismatch: {cell['cell_id']}")
        issues.extend(_graph_issues(cell))
        issues.extend(_trajectory_issues(cell))
    if summary.get("semantic_equivalence", {}).get("max_abs_error", math.inf) > float(
        config["semantic_equivalence"]["absolute_tolerance"]
    ):
        issues.append("concrete pure/compiler equivalence failed")
    expected = compare_cells(
        cells, float(config["numerical"]["comparison_tolerance"])
    )
    if expected != summary.get("comparisons"):
        issues.append("independent four-cell comparison mismatch")
    if summary.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("unexpected runner status")
    return {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": f"{config['scope']}_AUDIT",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "result": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "classification": summary.get("comparisons", {}).get("classification"),
        "cell_lower_bounds": summary.get("comparisons", {}).get("lower_bounds"),
        "issues": issues,
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path = _inside(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.summary)
    _write(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
