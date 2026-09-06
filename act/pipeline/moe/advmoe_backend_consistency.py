"""Four-cell backend consistency check for one frozen AdvMoE obligation.

This is a bounded development diagnostic, not another multiplier search.  It
compares one router-free static expert graph and the graph-matched zero-
multiplier compiler under plain CROWN and the already frozen sparse-alpha
configuration.  Negative bounds are always UNKNOWN.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from act.back_end.moe.tie_safe_implication import LagrangianTop1GuardedProperty
from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoePath,
    CrownCompatibleAdvMoeRouter,
    specialize_advmoe_path,
)
from act.pipeline.moe.advmoe_lagrangian_family_diagnosis import (
    select_mu0_blocking_obligation,
)
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.advmoe_two_path import (
    _cleanup_cuda,
    _load_model,
    top1_property_rows,
)
from act.pipeline.moe.crown_adapter_cohort import (
    _crown_bounds,
    validate_crown_configuration,
)
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_hashed_inputs(
    config: dict[str, Any], workspace: Path
) -> dict[str, Path]:
    paths = {}
    for name, identity in config["inputs"].items():
        path = _inside(Path(identity["path"]), workspace)
        if _sha256(path) != identity["sha256"]:
            raise RuntimeError(f"input identity mismatch: {name}")
        paths[name] = path
    return paths


def semantic_equivalence_points(
    pure_expert: torch.nn.Module,
    compiled_mu0: torch.nn.Module,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    property_row: np.ndarray,
    *,
    random_points: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    points = [center, lower, upper]
    for _index in range(int(random_points)):
        fraction = torch.rand(center.shape, generator=generator, dtype=center.dtype)
        points.append(lower + fraction * (upper - lower))
    inputs = torch.cat(points, dim=0)
    pure_expert = pure_expert.cpu().eval()
    compiled_mu0 = compiled_mu0.cpu().eval()
    with torch.no_grad():
        row = torch.as_tensor(property_row, dtype=inputs.dtype)
        pure = pure_expert(inputs) @ row
        compiled = compiled_mu0(inputs).reshape(-1)
    difference = (pure - compiled).abs()
    return {
        "points": int(inputs.shape[0]),
        "random_points": int(random_points),
        "seed": int(seed),
        "bit_exact": bool(torch.equal(pure, compiled)),
        "max_abs_error": float(difference.max().item()),
        "pure_values_sha256": _sha256_bytes(pure.numpy().tobytes()),
        "compiled_values_sha256": _sha256_bytes(compiled.numpy().tobytes()),
    }


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def compare_cells(cells: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    by_id = {cell["cell_id"]: cell for cell in cells}
    required = {
        "pure_expert__plain_crown",
        "compiled_mu0__plain_crown",
        "pure_expert__sparse_alpha",
        "compiled_mu0__sparse_alpha",
    }
    if set(by_id) != required:
        raise ValueError("four-cell result identity mismatch")

    def lower(cell_id: str) -> float | None:
        record = by_id[cell_id]["bound"]
        values = record.get("lower_bounds", [])
        if not bool(record.get("complete")) or len(values) != 1:
            return None
        value = float(values[0])
        return value if math.isfinite(value) else None

    pure_plain = lower("pure_expert__plain_crown")
    compiled_plain = lower("compiled_mu0__plain_crown")
    pure_alpha = lower("pure_expert__sparse_alpha")
    compiled_alpha = lower("compiled_mu0__sparse_alpha")
    complete = all(
        value is not None
        for value in (pure_plain, compiled_plain, pure_alpha, compiled_alpha)
    )
    if not complete:
        conclusion = "INCOMPLETE_RETAIN_AND_STOP"
    elif pure_plain > compiled_plain + tolerance:
        conclusion = "COMPILED_MU0_GRAPH_IS_WEAKER_THAN_ROUTER_FREE_EXPERT"
    elif (
        pure_alpha < pure_plain - tolerance
        and compiled_alpha < compiled_plain - tolerance
    ):
        conclusion = "SPARSE_ALPHA_CONFIGURATION_DEGRADES_BOTH_GRAPH_FORMS"
    elif pure_plain < tolerance and compiled_plain < tolerance:
        conclusion = "PLAIN_CROWN_DOES_NOT_CLOSE_EITHER_GRAPH_FORM"
    else:
        conclusion = "NO_PREREGISTERED_SINGLE_CAUSE_IDENTIFIED"
    return {
        "all_cells_complete": complete,
        "lower_bounds": {
            "pure_expert_plain_crown": pure_plain,
            "compiled_mu0_plain_crown": compiled_plain,
            "pure_expert_sparse_alpha": pure_alpha,
            "compiled_mu0_sparse_alpha": compiled_alpha,
        },
        "plain_graph_delta_pure_minus_compiled": (
            None
            if pure_plain is None or compiled_plain is None
            else pure_plain - compiled_plain
        ),
        "sparse_alpha_delta_from_plain": {
            "pure_expert": (
                None if pure_alpha is None or pure_plain is None else pure_alpha - pure_plain
            ),
            "compiled_mu0": (
                None
                if compiled_alpha is None or compiled_plain is None
                else compiled_alpha - compiled_plain
            ),
        },
        "classification": conclusion,
        "claim_limit": (
            "one frozen development obligation; this check does not identify a "
            "global cause or establish guarded safety"
        ),
    }


def _cell_module(
    graph_form: str,
    router: torch.nn.Module,
    expert: torch.nn.Module,
    route: int,
    property_row: np.ndarray,
) -> tuple[torch.nn.Module, tuple[tuple[np.ndarray, float], ...] | None]:
    if graph_form == "pure_expert":
        return copy.deepcopy(expert), ((np.asarray(property_row), 0.0),)
    if graph_form == "compiled_mu0":
        module = LagrangianTop1GuardedProperty(
            copy.deepcopy(router),
            copy.deepcopy(expert),
            int(route),
            np.asarray(property_row, dtype=np.float64)[None, :],
            np.asarray([0.0], dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
        )
        return module, None
    raise ValueError(f"unknown graph form: {graph_form}")


def run(config_path: Path) -> dict[str, Any]:
    config = _json(config_path)
    workspace = Path(config["workspace_boundary"])
    config_path = _inside(config_path, workspace)
    repository = _inside(Path(config["act_repository"]), workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("backend-consistency configuration is not frozen")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if _git(repository, "branch", "--show-current") != config["required_branch"]:
        raise RuntimeError("ACT branch gate failed")
    if _git(repository, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree is dirty")
    paths = _load_hashed_inputs(config, workspace)

    parent_config = _json(paths["parent_config"])
    parent_rows = _jsonl(paths["parent_rows"])
    parent_by_id = {row["row_id"]: row for row in parent_rows}
    stage_a_rows = _jsonl(paths["stage_a_rows"])
    selected = [row for row in stage_a_rows if row["row_id"] == config["obligation"]["row_id"]]
    if len(selected) != 1:
        raise RuntimeError("frozen Stage-A row identity changed")
    observed_obligation = {
        "row_id": selected[0]["row_id"],
        **select_mu0_blocking_obligation(
            selected[0], float(config["numerical"]["safe_positive_margin"])
        ),
    }
    if observed_obligation != config["obligation"]:
        raise RuntimeError("frozen obligation changed")
    parent = parent_by_id[observed_obligation["row_id"]]

    source = _inside(Path(parent_config["official_source"]["repository"]), workspace)
    if _git(source, "rev-parse", "HEAD") != parent_config["official_source"]["commit"]:
        raise RuntimeError("official source commit mismatch")
    if _git(source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")

    model, router, moe_type, checkpoint = _load_model(parent_config, workspace)
    route = int(observed_obligation["route"])
    specialized = specialize_advmoe_path(model, route, moe_type)[0].eval()
    expert = CrownCompatibleAdvMoePath(specialized).eval()
    router_adapter = CrownCompatibleAdvMoeRouter(router).eval()
    properties = top1_property_rows(int(parent["clean_prediction"]))
    property_row, offset = properties[int(observed_obligation["property_index"])]
    if float(offset) != 0.0:
        raise RuntimeError("frozen property offset is not zero")

    archive = _inside(Path(parent_config["dataset_archive"]), workspace)
    inputs, _labels = load_cifar10_test_archive(archive)
    index = int(parent["dataset_index"])
    center = torch.from_numpy(inputs[index : index + 1])
    epsilon = float(parent["epsilon"])
    lower = torch.clamp(center - epsilon, 0.0, 1.0)
    upper = torch.clamp(center + epsilon, 0.0, 1.0)
    compiled_for_equivalence, _unused = _cell_module(
        "compiled_mu0", router_adapter, expert, route, property_row
    )
    equivalence = semantic_equivalence_points(
        expert,
        compiled_for_equivalence,
        center,
        lower,
        upper,
        property_row,
        random_points=int(config["semantic_equivalence"]["random_points"]),
        seed=int(config["semantic_equivalence"]["seed"]),
    )
    if equivalence["max_abs_error"] > float(
        config["semantic_equivalence"]["absolute_tolerance"]
    ):
        raise RuntimeError("pure and compiled mu=0 concrete semantics differ")

    for backend_name, backend in config["backends"].items():
        validate_crown_configuration(
            method=backend["method"],
            track_gradients=bool(backend["gradient_tracking"]),
            bound_options=backend.get("bound_options"),
        )
    free, total = torch.cuda.mem_get_info(config["device"])
    if free / 1024**3 < float(config["minimum_free_gpu_memory_gib"]):
        raise RuntimeError("GPU memory gate failed")

    output_dir.mkdir(parents=True)
    cells = []
    started = time.monotonic()
    for cell_id in config["execution_order"]:
        graph_form, backend_name = cell_id.split("__", maxsplit=1)
        backend = config["backends"][backend_name]
        module, property_rows = _cell_module(
            graph_form, router_adapter, expert, route, property_row
        )
        _cleanup_cuda()
        free_before, _total = torch.cuda.mem_get_info(config["device"])
        bound = _crown_bounds(
            module,
            center,
            lower,
            upper,
            property_rows=property_rows,
            device=config["device"],
            tolerance=float(config["numerical"]["safe_positive_margin"]),
            method=backend["method"],
            track_gradients=bool(backend["gradient_tracking"]),
            bound_options=backend.get("bound_options"),
            capture_graph_metadata=True,
            capture_optimization_trace=True,
        )
        cells.append(
            {
                "cell_id": cell_id,
                "graph_form": graph_form,
                "backend": backend_name,
                "free_gpu_gib_before": free_before / 1024**3,
                "intermediate_bound_strategy": backend[
                    "intermediate_bound_strategy"
                ],
                "bound": bound,
            }
        )
        _cleanup_cuda()

    comparisons = compare_cells(
        cells, float(config["numerical"]["comparison_tolerance"])
    )
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "dataset_archive": {"path": str(archive), "sha256": _sha256(archive)},
        "obligation": {
            **observed_obligation,
            "sample_slot": int(parent["sample_slot"]),
            "dataset_index": index,
            "epsilon": epsilon,
            "epsilon_over_255": float(parent["epsilon_over_255"]),
            "clean_prediction": int(parent["clean_prediction"]),
            "clean_route": int(parent["clean_route"]),
            "property_row": np.asarray(property_row).tolist(),
        },
        "semantic_equivalence": equivalence,
        "cells": cells,
        "comparisons": comparisons,
        "runtime_seconds": time.monotonic() - started,
        "gpu": {"free_gib_before": free / 1024**3, "total_gib": total / 1024**3},
        "formal_safe_enabled": False,
        "negative_bound_semantics": "UNKNOWN_NEVER_UNSAFE",
        "holdout": "LOCKED_NOT_ACCESSED",
        "official_source_clean_after": not bool(
            _git(source, "status", "--porcelain=v1")
        ),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    print(json.dumps(result["comparisons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
