# ===- audit_experiment1n2_topk3_external.py ------------------------====#
"""External artifact, route-set, and witness audit for Experiment 1N2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import torch

from act.back_end.moe import (
    analyze_topk_sets,
    build_act_moe_program,
    load_output_moe_checkpoint,
)
from act.back_end.solver.solver_hz import SparseHZono
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _inside,
    _propagate_component,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1n2_topk3 import DEFAULT_CONFIG, _load_cifar10


SEMANTIC_SOURCE_PATHS = (
    "act/pipeline/moe/experiment1n2_topk3.py",
    "act/back_end/moe/weighted_topk.py",
    "act/back_end/moe/topk_routes.py",
    "act/back_end/moe/model.py",
    "act/back_end/solver/solver_hz.py",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _semantic_source_drift(launch_head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", launch_head, "--", *SEMANTIC_SOURCE_PATHS],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def route_structure_issues(row: dict[str, Any]) -> list[str]:
    """Audit top-3 aggregation without trusting the stored summary."""

    rank = int(row["sample_rank"])
    issues: list[str] = []
    routes = row.get("route_sets", [])
    recorded = [tuple(int(value) for value in route.get("route_set", [])) for route in routes]
    if not routes:
        issues.append(f"rank {rank}: no evaluated route set")
    if len(set(recorded)) != len(recorded):
        issues.append(f"rank {rank}: duplicate route set")
    if any(len(route) != 3 or tuple(sorted(route)) != route for route in recorded):
        issues.append(f"rank {rank}: route set is not canonical unordered top-3")
    if int(row.get("exact_feasible_unordered_top3_set_count", -1)) != len(routes):
        issues.append(f"rank {rank}: feasible-set count differs from evaluated sets")
    if row.get("route_set_enumeration_exact") is not True:
        issues.append(f"rank {rank}: route-set enumeration is not exact")
    if row.get("unresolved_route_sets"):
        issues.append(f"rank {rank}: exact row retains unresolved route sets")
    if row.get("products_per_property") != 2:
        issues.append(f"rank {rank}: declared product count is not k-1=2")

    for route in routes:
        selected = tuple(int(value) for value in route["route_set"])
        properties = route.get("property_rows", [])
        if route.get("reason") == "SAFE_GATE_ELIMINATION":
            obligations = route.get("gate_elimination", [])
            if {int(item.get("expert", -1)) for item in obligations} != set(selected):
                issues.append(f"rank {rank}: gate SAFE misses a selected expert")
            if any(
                item.get("status") != "SAFE"
                or item.get("solver_status") != "certified"
                or item.get("solver_reason") != "expanded_violations_infeasible"
                for item in obligations
            ):
                issues.append(f"rank {rank}: gate SAFE has an uncertified expert")
            if properties:
                issues.append(f"rank {rank}: gate SAFE unexpectedly uses fallback")
        else:
            indices = [int(prop.get("property_index", -1)) for prop in properties]
            replayed_unsafe = (
                route.get("status") == "UNSAFE"
                and route.get("full_model_witness_valid") is True
            )
            if replayed_unsafe:
                if not properties or len(set(indices)) != len(indices):
                    issues.append(
                        f"rank {rank}: validated UNSAFE route has invalid property prefix"
                    )
                if not any(
                    prop.get("status") == "UNSAFE"
                    and prop.get("full_model_witness_valid") is True
                    for prop in properties
                ):
                    issues.append(
                        f"rank {rank}: validated UNSAFE route lacks replayed property"
                    )
            else:
                if len(properties) != 9:
                    issues.append(
                        f"rank {rank}: fallback route {selected} lacks nine properties"
                    )
                if set(indices) != set(range(9)) or len(indices) != 9:
                    issues.append(
                        f"rank {rank}: fallback route {selected} property coverage differs"
                    )
            if any(int(prop.get("product_count", -1)) != 2 for prop in properties):
                issues.append(
                    f"rank {rank}: fallback route {selected} does not use two products"
                )
            if any(
                prop.get("status") == "UNSAFE"
                and prop.get("full_model_witness_valid") is not True
                for prop in properties
            ):
                issues.append(
                    f"rank {rank}: fallback route {selected} promotes relaxation UNSAFE"
                )
        if route.get("status") == "SAFE":
            if route.get("reason") == "SAFE_WEIGHTED_RANGE" and any(
                prop.get("status") != "SAFE" for prop in properties
            ):
                issues.append(f"rank {rank}: weighted SAFE has unresolved property")

    if row.get("status") == "SAFE" and any(
        route.get("status") != "SAFE" for route in routes
    ):
        issues.append(f"rank {rank}: row SAFE without all routes SAFE")
    if row.get("status") == "UNSAFE" and row.get("full_model_witness_valid") is not True:
        issues.append(f"rank {rank}: UNSAFE lacks full-model witness flag")
    return issues


def _recompute_route_sets(model, dataset, row: dict[str, Any], config: dict[str, Any]):
    image, label = dataset[int(row["dataset_index"])]
    clean = image.unsqueeze(0).double()
    epsilon = float(row["epsilon"])
    lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
    with torch.no_grad():
        output, decision = model.forward_with_routing(clean)
    prediction = int(output.argmax(dim=1).item())
    if prediction != int(label) or prediction != int(row["clean_prediction"]):
        return None, "sample is not the recorded clean-correct input"
    actual_clean_set = sorted(int(value) for value in decision.indices[0].tolist())
    if actual_clean_set != sorted(int(value) for value in row["clean_topk_set"]):
        return None, "clean top-3 set differs during replay"
    program = build_act_moe_program(
        model,
        center=clean,
        lower=lower,
        upper=upper,
        output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[prediction]),
    )
    router = _propagate_component(program.router)
    if not isinstance(router.output_hz, SparseHZono) or not router.output_hz.exact:
        return None, "router replay is not an exact sparse HZ"
    route_sets = analyze_topk_sets(
        router.output_hz,
        model.spec.top_k,
        time_limit_per_set=float(config["solver"]["route_set_seconds"]),
        router_exact=True,
    )
    if not route_sets.exact:
        return None, "route-set replay did not complete exactly"
    return [tuple(int(value) for value in values) for values in route_sets.feasible], None


def _replay_witness(
    model, dataset, row: dict[str, Any], output_dir: Path
) -> list[str]:
    rank = int(row["sample_rank"])
    relative = row.get("witness_path")
    if not relative:
        return [f"rank {rank}: UNSAFE witness path missing"]
    path = _inside(output_dir / relative, output_dir)
    if not path.is_file() or _sha256(path) != row.get("witness_sha256"):
        return [f"rank {rank}: UNSAFE witness path/hash mismatch"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidate = payload.get("input")
    if not isinstance(candidate, torch.Tensor):
        return [f"rank {rank}: witness payload lacks tensor input"]
    image, _ = dataset[int(row["dataset_index"])]
    clean = image.unsqueeze(0).double()
    epsilon = float(row["epsilon"])
    checked = _forward_validate(
        model,
        candidate,
        lower=(clean - epsilon).clamp(0, 1),
        upper=(clean + epsilon).clamp(0, 1),
        clean_prediction=int(row["clean_prediction"]),
    )
    issues: list[str] = []
    if not checked["valid"]:
        issues.append(f"rank {rank}: full-model witness replay failed")
    metadata = payload.get("metadata") or {}
    if int(metadata.get("clean_prediction", -1)) != int(row["clean_prediction"]):
        issues.append(f"rank {rank}: witness clean prediction metadata differs")
    if int(metadata.get("counterexample_prediction", -1)) != int(
        row["counterexample_prediction"]
    ):
        issues.append(f"rank {rank}: witness counterexample metadata differs")
    if sorted(metadata.get("counterexample_topk_set", [])) != sorted(
        row["counterexample_topk_set"]
    ):
        issues.append(f"rank {rank}: witness route metadata differs")
    return issues


def audit(config_path: Path, result_dir: Path | None = None) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(
        result_dir if result_dir is not None else Path(config["output_dir"]), WRITE_ROOT
    )
    output_path = output_dir / "external_audit_r2.json"
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    runtime = json.loads((output_dir / "runtime_config.json").read_text(encoding="utf-8"))
    selection = json.loads((output_dir / "sample_indices.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    rows = _load_jsonl(output_dir / "results.jsonl")
    issues: list[str] = []
    if runtime.get("source_config_sha256") != _sha256(config_path):
        issues.append("runtime source config hash mismatch")
    if runtime.get("config") != config:
        issues.append("embedded runtime config differs")
    if runtime.get("checkpoint_sha256") != _sha256(Path(config["checkpoint"])):
        issues.append("checkpoint hash mismatch")
    launch_head = str(runtime.get("git_head", ""))
    try:
        drift = _semantic_source_drift(launch_head)
    except (subprocess.CalledProcessError, ValueError):
        drift = ["UNABLE_TO_COMPARE_LAUNCH_HEAD"]
    if drift:
        issues.append(f"semantic source drift after launch: {drift}")
    expected_ranks = [int(value) for value in config["cohort"]["clean_correct_ranks"]]
    if [int(row["sample_rank"]) for row in rows] != expected_ranks:
        issues.append("result ranks differ from frozen cohort")
    if [int(row["sample_rank"]) for row in selection] != expected_ranks:
        issues.append("selection ranks differ from frozen cohort")
    selected = {int(row["sample_rank"]): int(row["dataset_index"]) for row in selection}

    model, payload = load_output_moe_checkpoint(Path(config["checkpoint"]), map_location="cpu")
    model.double().eval()
    dataset = _load_cifar10(Path(config["dataset_root"]))
    recomputed_sets: dict[str, list[list[int]]] = {}
    replayed = 0
    for row in rows:
        rank = int(row["sample_rank"])
        if int(row["dataset_index"]) != selected.get(rank):
            issues.append(f"rank {rank}: dataset index differs from selection")
        if not math.isclose(
            float(row["epsilon"]),
            float(config["verification"]["epsilon"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            issues.append(f"rank {rank}: epsilon differs from frozen config")
        issues.extend(route_structure_issues(row))
        exact_sets, error = _recompute_route_sets(model, dataset, row, config)
        if error:
            issues.append(f"rank {rank}: {error}")
        else:
            recomputed_sets[str(rank)] = [list(values) for values in exact_sets]
            recorded = [
                tuple(int(value) for value in values)
                for values in row["exact_feasible_unordered_top3_sets"]
            ]
            if exact_sets != recorded:
                issues.append(f"rank {rank}: independently recomputed sets differ")
        if row["status"] == "UNSAFE":
            witness_issues = _replay_witness(model, dataset, row, output_dir)
            issues.extend(witness_issues)
            replayed += int(not witness_issues)

    stored_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    independent_counts = {
        "rows": len(rows),
        "route_sets": sum(len(row.get("route_sets", [])) for row in rows),
        "fallback_property_rows": sum(
            len(route.get("property_rows", []))
            for row in rows
            for route in row.get("route_sets", [])
        ),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
    }
    for key in independent_counts:
        if independent_counts[key] != stored_summary.get(key):
            issues.append(f"independent summary field differs: {key}")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_issues = []
    for relative, expected in manifest.items():
        if relative == "manifest_payload_sha256":
            continue
        path = output_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            manifest_issues.append(relative)
    if manifest_issues:
        issues.append(f"runner manifest hash mismatch: {manifest_issues}")

    result = {
        "schema_version": 1,
        "scope": "external_route_set_and_witness_audit",
        "excluded_external_audit_r1": {
            "path": str(output_dir / "external_audit.json"),
            "issue_count": 2,
            "cause": (
                "audit implementation required nine property rows after an "
                "already full-forward-validated early UNSAFE witness"
            ),
        },
        "issue_count": len(issues),
        "issues": issues,
        "passed": not issues,
        "rows": len(rows),
        "unsafe_rows": sum(row["status"] == "UNSAFE" for row in rows),
        "unsafe_witnesses_replayed": replayed,
        "recomputed_exact_route_sets": recomputed_sets,
        "independent_counts": independent_counts,
        "runner_manifest_files_checked": len(manifest) - 1,
        "result_jsonl_sha256": _sha256(output_dir / "results.jsonl"),
        "selection_sha256": _sha256(output_dir / "sample_indices.json"),
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--result-dir")
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(
                Path(arguments.config),
                Path(arguments.result_dir) if arguments.result_dir else None,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
