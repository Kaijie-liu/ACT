# ===- benchmark_guarded_box_hull.py - Paired guarded-hull benchmark -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Paired guarded-box-hull engineering benchmark on frozen route branches.

This runner rematerializes every exact feasible unordered top-2 route set for
the frozen twenty-row Experiment 1D selection.  It then evaluates the same
guarded input HZ with the incremental highspy implementation and the SciPy
reference implementation.  Odd/even sample ranks reverse backend order to
make a global order drift visible instead of silently favouring one backend.

The benchmark is an engineering rerun.  It never overwrites confirmatory or
Experiment 1D artifacts.  If either backend is incomplete, its sound fast HZ
fallback is retained and the branch is excluded from every speed conclusion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np

from act.back_end.moe import (
    guarded_hz_box_hull_highs,
    guarded_hz_box_hull_scipy,
    guarded_input_topk_set,
    load_output_moe_checkpoint,
)
from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_fast_bounds
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _git_value,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1d import _load_frozen_selection, _row_context
from act.pipeline.moe.train import _load_dataset
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/benchmark_guarded_box_hull_experiment1d.json"
)
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
RESULTS_ROOT = PROJECT_ROOT / "data/moe/results"
_BACKEND_NAMES = ("highspy", "scipy")
_COUNTER_FIELDS = (
    "model_builds",
    "objective_update_calls",
    "objective_coefficients_changed",
    "solves",
    "cold_start_solves",
    "basis_submission_attempts",
    "basis_submissions_accepted",
    "basis_valid_after_solve",
    "simplex_iterations",
    "ipm_iterations",
)
_TIME_FIELDS = (
    "model_build_seconds",
    "objective_update_seconds",
    "solve_seconds",
    "total_seconds",
)


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _append_log(handle, value: str) -> None:
    handle.write(value.rstrip() + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _backend_order(sample_rank: int) -> tuple[str, str]:
    """Freeze an alternating order without depending on traversal position."""

    if int(sample_rank) % 2 == 0:
        return ("highspy", "scipy")
    return ("scipy", "highspy")


def _bounds_arrays(bounds) -> tuple[np.ndarray, np.ndarray]:
    lower = bounds.lb.detach().cpu().double().numpy().reshape(-1)
    upper = bounds.ub.detach().cpu().double().numpy().reshape(-1)
    return lower, upper


def _bounds_sha256(lower: np.ndarray, upper: np.ndarray) -> str:
    values = np.stack((lower, upper), axis=0).astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _max_abs_bound_difference(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> float | None:
    left_lower, left_upper = left
    right_lower, right_upper = right
    if left_lower.shape != right_lower.shape or left_upper.shape != right_upper.shape:
        return None
    if not all(
        np.all(np.isfinite(value))
        for value in (left_lower, left_upper, right_lower, right_upper)
    ):
        return None
    if left_lower.size == 0:
        return 0.0
    return float(
        max(
            np.max(np.abs(left_lower - right_lower)),
            np.max(np.abs(left_upper - right_upper)),
        )
    )


def _empty_telemetry(backend: str, elapsed: float) -> dict[str, Any]:
    return {
        "backend": backend,
        **{key: 0 for key in _COUNTER_FIELDS},
        **{key: 0.0 for key in _TIME_FIELDS},
        "total_seconds": float(elapsed),
        "status_counts": {"runner_error": 1},
        "basis_semantics": "no basis submitted after runner error",
        "warm_start_claimed": False,
    }


def _result_record(result, wall_seconds: float) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    arrays = _bounds_arrays(result.bounds)
    fallback_sides = sum(
        status not in {"lp_optimal", "constant_exact"}
        for status in (*result.lower_status, *result.upper_status)
    )
    return (
        {
            "complete": bool(result.complete),
            "status": result.domain_status,
            "domain_status": result.domain_status,
            "exact": bool(result.exact),
            "relaxed_binaries": int(result.relaxed_binaries),
            "fallback_sides": int(fallback_sides),
            "bounds_sha256": _bounds_sha256(*arrays),
            "wall_seconds": float(wall_seconds),
            "telemetry": result.telemetry.as_dict(),
            "error": None,
        },
        arrays,
    )


def _fallback_record(
    backend: str,
    hz: SparseHZono,
    error: Exception,
    wall_seconds: float,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    """Preserve a sound unconditioned-generator fallback after runner errors."""

    arrays = _bounds_arrays(sparse_hz_fast_bounds(hz))
    return (
        {
            "complete": False,
            "status": "runner_error",
            "domain_status": "runner_error",
            "exact": False,
            "relaxed_binaries": int(hz.n_bin),
            "fallback_sides": int(2 * hz.n_out),
            "bounds_sha256": _bounds_sha256(*arrays),
            "wall_seconds": float(wall_seconds),
            "telemetry": _empty_telemetry(backend, wall_seconds),
            "error": f"{type(error).__name__}: {error}",
        },
        arrays,
    )


def _run_backend(
    backend: str,
    hz: SparseHZono,
    *,
    time_limit: float,
    submit_basis: bool,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    started = time.monotonic()
    try:
        if backend == "highspy":
            result = guarded_hz_box_hull_highs(
                hz, time_limit=time_limit, submit_basis=submit_basis
            )
        elif backend == "scipy":
            result = guarded_hz_box_hull_scipy(hz, time_limit=time_limit)
        else:  # pragma: no cover - guarded by the frozen caller
            raise ValueError(f"unknown backend {backend}")
        return _result_record(result, time.monotonic() - started)
    except Exception as error:  # preserve sound fallback and keep paired record
        return _fallback_record(backend, hz, error, time.monotonic() - started)


def _telemetry_totals(records: Iterable[dict[str, Any]], backend: str) -> dict[str, Any]:
    selected = [record[backend] for record in records]
    status_counts: Counter[str] = Counter()
    for result in selected:
        status_counts.update(result["telemetry"].get("status_counts", {}))
    return {
        "branches": len(selected),
        "complete_branches": sum(bool(result["complete"]) for result in selected),
        "fallback_sides": sum(int(result["fallback_sides"]) for result in selected),
        "wall_seconds": sum(float(result["wall_seconds"]) for result in selected),
        **{
            key: sum(int(result["telemetry"].get(key, 0)) for result in selected)
            for key in _COUNTER_FIELDS
        },
        **{
            key: sum(float(result["telemetry"].get(key, 0.0)) for result in selected)
            for key in _TIME_FIELDS
        },
        "status_counts": dict(sorted(status_counts.items())),
    }


def _summary(
    branches: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    comparison_tolerance: float,
) -> dict[str, Any]:
    paired_complete = [branch for branch in branches if branch["paired_complete"]]
    complete_differences = [
        float(branch["bound_max_abs_diff"])
        for branch in paired_complete
        if branch.get("bound_max_abs_diff") is not None
    ]
    ratios = [
        float(branch["scipy"]["wall_seconds"])
        / float(branch["highspy"]["wall_seconds"])
        for branch in paired_complete
        if float(branch["highspy"]["wall_seconds"]) > 0.0
    ]
    row_errors = [int(row["sample_rank"]) for row in rows if row.get("error")]
    all_complete = (
        len(rows) == int(expected_rows)
        and not row_errors
        and bool(branches)
        and len(paired_complete) == len(branches)
    )
    disagreements = [
        branch["branch_id"]
        for branch in paired_complete
        if branch.get("bound_max_abs_diff") is None
        or float(branch["bound_max_abs_diff"]) > float(comparison_tolerance)
    ]
    speed_conclusion = None
    if all_complete and not disagreements:
        speed_conclusion = {
            "eligible": True,
            "scope": "all frozen rematerialized exact feasible route branches",
            "scipy_over_highspy_wall_ratio_median": statistics.median(ratios)
            if ratios
            else None,
            "ratio_values": ratios,
            "interpretation": (
                "descriptive paired engineering speed ratio; no solver-internal "
                "warm-start claim"
            ),
        }
    else:
        reason = (
            "complete paired backends disagree beyond the frozen tolerance; "
            "no speed conclusion is made"
            if all_complete and disagreements
            else (
                "at least one frozen row or paired backend result is incomplete; "
                "sound fallback bounds are retained and no speed conclusion is made"
            )
        )
        speed_conclusion = {
            "eligible": False,
            "reason": reason,
            "scipy_over_highspy_wall_ratio_median": None,
            "ratio_values": [],
        }
    return {
        "result_semantics": "engineering_performance_rerun_not_confirmatory_overwrite",
        "expected_rows": int(expected_rows),
        "completed_rows": len(rows),
        "row_error_sample_ranks": row_errors,
        "route_branches": len(branches),
        "paired_complete_branches": len(paired_complete),
        "incomplete_or_fallback_branches": len(branches) - len(paired_complete),
        "backend_order_counts": dict(
            sorted(
                (
                    "->".join(order),
                    count,
                )
                for order, count in Counter(
                    tuple(branch["backend_order"]) for branch in branches
                ).items()
            )
        ),
        "highspy": _telemetry_totals(branches, "highspy"),
        "scipy": _telemetry_totals(branches, "scipy"),
        "comparison_tolerance": float(comparison_tolerance),
        "complete_pair_bound_max_abs_diff": max(complete_differences, default=None),
        "complete_pair_bound_disagreement_branch_ids": disagreements,
        "all_frozen_branches_complete": all_complete,
        "all_complete_bounds_within_tolerance": all_complete and not disagreements,
        "speed_conclusion": speed_conclusion,
        "original_confirmatory_overall_solved_rate_immutable": 0.56,
    }


def _run_branch(
    context: dict[str, Any],
    pair: tuple[int, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    guarded = guarded_input_topk_set(
        context["router"].input_hz,
        context["router"].output_hz,
        pair,
    ).hz
    if not isinstance(guarded, SparseHZono) or not guarded.exact:
        raise RuntimeError("benchmark requires an exact guarded input HZ")
    order = _backend_order(context["rank"])
    results: dict[str, dict[str, Any]] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for backend in order:
        results[backend], arrays[backend] = _run_backend(
            backend,
            guarded,
            time_limit=float(config["backend_time_limit_seconds"]),
            submit_basis=bool(config["highspy_submit_basis"]),
        )
    difference = _max_abs_bound_difference(arrays["highspy"], arrays["scipy"])
    paired_complete = all(results[name]["complete"] for name in _BACKEND_NAMES)
    branch_id = f"rank{context['rank']}:pair{pair[0]}-{pair[1]}"
    return {
        "branch_id": branch_id,
        "sample_rank": int(context["rank"]),
        "dataset_index": int(context["index"]),
        "epsilon": float(context["epsilon"]),
        "route_pair": list(pair),
        "guarded_hz": {
            "exact": bool(guarded.exact),
            "output_width": int(guarded.n_out),
            "continuous_generators": int(guarded.n_cont),
            "binary_generators": int(guarded.n_bin),
            "equality_constraints": int(guarded.Ac.shape[0]),
            "inequality_constraints": int(guarded.Auc.shape[0]),
            "frame_id": guarded.frame_id,
        },
        "backend_order": list(order),
        "highspy": results["highspy"],
        "scipy": results["scipy"],
        "paired_complete": paired_complete,
        "bound_max_abs_diff": difference,
        "bound_comparison_scope": (
            "complete_pair" if paired_complete else "descriptive_with_sound_fallback"
        ),
        "within_tolerance": (
            difference is not None
            and difference <= float(config["bound_comparison_tolerance"])
        )
        if paired_complete
        else None,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), RESULTS_ROOT)
    checkpoint_path = _inside(Path(config["checkpoint"]), PROJECT_ROOT)
    selection_path = _inside(Path(config["selection_manifest"]), PROJECT_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("guarded-hull benchmark requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("guarded-hull benchmark requires a clean worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("guarded-hull benchmark requires act-py312")
    data_root = Path(get_torchvision_data_root()).resolve()
    if not data_root.is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision data root escapes /data1/Kane/MOE")
    if _sha256(selection_path) != config["selection_manifest_sha256"]:
        raise RuntimeError("frozen Experiment 1D selection manifest changed")
    if _sha256(checkpoint_path) != config["checkpoint_sha256"]:
        raise RuntimeError("frozen bal010 checkpoint changed")
    selected = _load_frozen_selection(config)
    if output_dir.exists():
        raise RuntimeError(f"guarded-hull benchmark refuses to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    model, payload = load_output_moe_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    _write_json(
        output_dir / "config.json",
        {
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "selection_manifest_sha256": _sha256(selection_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "git_head": _git_value("rev-parse", "HEAD"),
            "config": config,
        },
    )
    _write_json(
        output_dir / "selection.json",
        {
            "parent_results_sha256": config["parent_results_sha256"],
            "rows": [
                {key: value for key, value in row.items() if key != "parent"}
                for row in selected
            ],
        },
    )

    branches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    with (
        (output_dir / "branches.jsonl").open("x", encoding="utf-8") as branch_handle,
        (output_dir / "rows.jsonl").open("x", encoding="utf-8") as row_handle,
        (output_dir / "benchmark.log").open("x", encoding="utf-8") as log_handle,
    ):
        for position, selection in enumerate(selected, 1):
            row_started = time.monotonic()
            row_branches: list[dict[str, Any]] = []
            error = None
            try:
                context = _row_context(selection, model, dataset, config)
                if not context["route_sets"].exact:
                    raise RuntimeError("route-set enumeration is not exact")
                for pair_values in context["route_sets"].feasible:
                    pair = tuple(sorted(int(value) for value in pair_values))
                    branch = _run_branch(context, pair, config)
                    branches.append(branch)
                    row_branches.append(branch)
                    _append_json(branch_handle, branch)
                    _append_log(
                        log_handle,
                        f"BRANCH rank={context['rank']} pair={pair} "
                        f"complete={branch['paired_complete']} "
                        f"diff={branch['bound_max_abs_diff']}",
                    )
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            row = {
                "sample_rank": int(selection["sample_rank"]),
                "dataset_index": int(selection["dataset_index"]),
                "epsilon": float(selection["epsilon"]),
                "route_branch_count": len(row_branches),
                "paired_complete_branches": sum(
                    bool(branch["paired_complete"]) for branch in row_branches
                ),
                "branch_ids": [branch["branch_id"] for branch in row_branches],
                "backend_order": list(_backend_order(int(selection["sample_rank"]))),
                "complete": error is None
                and all(branch["paired_complete"] for branch in row_branches),
                "error": error,
                "total_seconds": time.monotonic() - row_started,
            }
            rows.append(row)
            _append_json(row_handle, row)
            _append_log(
                log_handle,
                f"ROW {position}/{len(selected)} rank={row['sample_rank']} "
                f"branches={row['route_branch_count']} complete={row['complete']} "
                f"error={row['error']}",
            )

    summary = _summary(
        branches,
        rows,
        expected_rows=int(config["expected_rows"]),
        comparison_tolerance=float(config["bound_comparison_tolerance"]),
    )
    _write_json(output_dir / "summary.json", summary)
    return {"output_dir": str(output_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
