# ===- act/pipeline/moe/exact_big_m_engineering.py -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-----------------------------------------------------------------===#

"""Paired engineering study for constraint-aware top-k membership big-M.

This runner deliberately stops at router membership feasibility.  The frozen
Experiment 1D property rows condition on an already known top-k set and do not
consume selector big-M constants; relabelling an Experiment 1D rerun as a
big-M experiment would therefore be a false attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from act.back_end.moe import (
    build_act_router_program,
    condition_topk_membership,
    load_output_moe_checkpoint,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_check_feasibility,
    hz_numerical_policy_manifest,
)
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _git_value,
    _inside,
    _propagate_component,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1d import _load_frozen_selection
from act.pipeline.moe.train import _load_dataset
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/exact_big_m_engineering_r1.json"
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")


def _append_json(handle, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0 or not math.isfinite(numerator + denominator):
        return None
    return numerator / denominator


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _support_side_fell_back(status: str) -> bool:
    return str(status) not in {
        "fast_generator",
        "optimal",
        "lp_optimal",
        "milp_optimal",
    }


def _mode_summary(expert_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["feasibility_status"]) for row in expert_rows)
    feasible = sorted(
        int(row["expert"])
        for row in expert_rows
        if row["feasibility_status"] == "feasible"
    )
    unknown = sorted(
        int(row["expert"])
        for row in expert_rows
        if row["feasibility_status"] == "unknown"
    )
    return {
        "candidate_experts": feasible,
        "unresolved_experts": unknown,
        "complete": not unknown,
        "status_counts": dict(statuses),
        "selector_binaries": sum(int(row["selector_binaries"]) for row in expert_rows),
        "conditioned_binary_width": sum(int(row["conditioned_binary_width"]) for row in expert_rows),
        "support_exact_experts": sum(bool(row["big_m_support_exact"]) for row in expert_rows),
        "support_fallback_sides": sum(int(row["support_fallback_sides"]) for row in expert_rows),
        "mip_nodes": sum(int(row["feasibility_nodes"]) for row in expert_rows),
        "membership_seconds": sum(float(row["membership_seconds"]) for row in expert_rows),
        "feasibility_seconds": sum(float(row["feasibility_seconds"]) for row in expert_rows),
        "total_seconds": sum(float(row["total_seconds"]) for row in expert_rows),
    }


def summarize_conditions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute paired aggregate metrics from raw per-condition rows."""
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["sample_rank"]), str(row["mode"]))
        if key in by_key:
            raise ValueError(f"duplicate condition row {key}")
        by_key[key] = row
    ranks = sorted({rank for rank, _ in by_key})
    pairs: list[dict[str, Any]] = []
    for rank in ranks:
        if (rank, "fast") not in by_key or (rank, "exact") not in by_key:
            continue
        fast, exact = by_key[(rank, "fast")], by_key[(rank, "exact")]
        pairs.append(
            {
                "sample_rank": rank,
                "both_complete": bool(fast["complete"] and exact["complete"]),
                "candidate_sets_equal": fast["candidate_experts"] == exact["candidate_experts"],
                "selector_binary_reduction": int(fast["selector_binaries"]) - int(exact["selector_binaries"]),
                "node_reduction": int(fast["mip_nodes"]) - int(exact["mip_nodes"]),
                "exact_over_fast_total_time": _finite_ratio(
                    float(exact["total_seconds"]), float(fast["total_seconds"])
                ),
                "exact_over_fast_feasibility_time": _finite_ratio(
                    float(exact["feasibility_seconds"]),
                    float(fast["feasibility_seconds"]),
                ),
            }
        )
    complete_pairs = [row for row in pairs if row["both_complete"]]
    return {
        "condition_rows": len(rows),
        "sample_ranks": len(ranks),
        "paired_ranks": len(pairs),
        "both_complete_pairs": len(complete_pairs),
        "complete_candidate_set_mismatches": sum(
            not row["candidate_sets_equal"] for row in complete_pairs
        ),
        "pairs_with_selector_binary_reduction": sum(
            row["selector_binary_reduction"] > 0 for row in pairs
        ),
        "total_selector_binary_reduction": sum(
            row["selector_binary_reduction"] for row in pairs
        ),
        "median_exact_over_fast_total_time": _median(
            [row["exact_over_fast_total_time"] for row in pairs if row["exact_over_fast_total_time"] is not None]
        ),
        "median_exact_over_fast_feasibility_time": _median(
            [row["exact_over_fast_feasibility_time"] for row in pairs if row["exact_over_fast_feasibility_time"] is not None]
        ),
        "pairs": pairs,
        "claim_scope": "top-k membership feasibility only; expert property solving is unchanged",
    }


def paired_audit_issues(
    condition_rows: Sequence[Mapping[str, Any]],
    expert_rows: Sequence[Mapping[str, Any]],
    *,
    expected_ranks: int,
    num_experts: int = 8,
    tolerance: float = 1e-8,
) -> list[str]:
    """Return independently checkable semantic and accounting issues."""
    issues: list[str] = []
    expected_conditions = expected_ranks * 2
    if len(condition_rows) != expected_conditions:
        issues.append(f"condition row count {len(condition_rows)} != {expected_conditions}")
    keys = [(int(row["sample_rank"]), str(row["mode"])) for row in condition_rows]
    if len(keys) != len(set(keys)):
        issues.append("duplicate condition key")
    expert_keys = [
        (int(row["sample_rank"]), str(row["mode"]), int(row["expert"]))
        for row in expert_rows
    ]
    if len(expert_keys) != len(set(expert_keys)):
        issues.append("duplicate expert condition key")
    expected_expert_rows = expected_conditions * num_experts
    if len(expert_rows) != expected_expert_rows:
        issues.append(f"expert row count {len(expert_rows)} != {expected_expert_rows}")

    experts_by_key = {
        (int(row["sample_rank"]), str(row["mode"]), int(row["expert"])): row
        for row in expert_rows
    }
    ranks = sorted({rank for rank, _ in keys})
    for rank in ranks:
        fast_condition = next((row for row in condition_rows if int(row["sample_rank"]) == rank and row["mode"] == "fast"), None)
        exact_condition = next((row for row in condition_rows if int(row["sample_rank"]) == rank and row["mode"] == "exact"), None)
        if fast_condition is None or exact_condition is None:
            issues.append(f"rank {rank} lacks a paired condition")
            continue
        if fast_condition["complete"] and exact_condition["complete"] and fast_condition["candidate_experts"] != exact_condition["candidate_experts"]:
            issues.append(f"rank {rank} complete candidate semantics differ")
        if int(exact_condition["selector_binaries"]) > int(fast_condition["selector_binaries"]):
            issues.append(f"rank {rank} exact selector width increased")
        for expert in range(num_experts):
            fast = experts_by_key.get((rank, "fast", expert))
            exact = experts_by_key.get((rank, "exact", expert))
            if fast is None or exact is None:
                continue
            if exact["big_m_support_mode"] != "exact":
                issues.append(f"rank {rank} expert {expert} wrong exact support label")
            fast_m = {int(key): float(value) for key, value in fast["big_m"].items()}
            exact_m = {int(key): float(value) for key, value in exact["big_m"].items()}
            for competitor, value in exact_m.items():
                if competitor not in fast_m:
                    issues.append(f"rank {rank} expert {expert} exact introduced competitor {competitor}")
                elif value > fast_m[competitor] + tolerance:
                    issues.append(f"rank {rank} expert {expert} exact M increased for competitor {competitor}")
    return issues


def _run_mode(
    router_hz: SparseHZono,
    *,
    rank: int,
    dataset_index: int,
    epsilon: float,
    mode: str,
    top_k: int,
    support_time_limit: float,
    feasibility_time_limit: float,
    expert_handle,
) -> dict[str, Any]:
    expert_rows: list[dict[str, Any]] = []
    for expert in range(router_hz.n_out):
        started = time.monotonic()
        membership_started = time.monotonic()
        membership = condition_topk_membership(
            router_hz,
            expert,
            top_k,
            big_m_support_mode=mode,
            big_m_support_time_limit=support_time_limit,
        )
        membership_seconds = time.monotonic() - membership_started
        result = hz_check_feasibility(
            membership.hz,
            time_limit=feasibility_time_limit,
        )
        statuses = tuple(str(value) for value in membership.big_m_upper_status)
        row = {
            "sample_rank": rank,
            "dataset_index": dataset_index,
            "epsilon": epsilon,
            "mode": mode,
            "expert": expert,
            "big_m": {str(key): float(value) for key, value in membership.big_m.items()},
            "selector_binaries": int(membership.selection_binaries),
            "conditioned_binary_width": int(membership.hz.n_bin),
            "router_binary_width": int(router_hz.n_bin),
            "big_m_support_mode": membership.big_m_support_mode,
            "big_m_support_exact": bool(membership.big_m_support_exact),
            "big_m_upper_status": list(statuses),
            "support_fallback_sides": sum(
                _support_side_fell_back(value) for value in statuses
            ),
            "feasibility_status": result.status,
            "feasibility_nodes": int(result.nodes),
            "membership_seconds": membership_seconds,
            "feasibility_seconds": float(result.elapsed),
            "total_seconds": time.monotonic() - started,
        }
        expert_rows.append(row)
        _append_json(expert_handle, row)
    return {
        "sample_rank": rank,
        "dataset_index": dataset_index,
        "epsilon": epsilon,
        "mode": mode,
        **_mode_summary(expert_rows),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_config_path = _inside(Path(config["selection_source_config"]), PROJECT_ROOT)
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("exact big-M study requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("exact big-M study requires a clean worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("exact big-M study requires act-py312")
    if _sha256(source_config_path) != config["selection_source_config_sha256"]:
        raise RuntimeError("frozen selection source config changed")
    if config["numerical_safety"] != hz_numerical_policy_manifest():
        raise RuntimeError("frozen numerical policy differs from implementation")
    if not Path(get_torchvision_data_root()).resolve().is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision data root escapes /data1/Kane/MOE")
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")

    selected = _load_frozen_selection(source_config)
    if len(selected) != int(config["expected_sample_ranks"]):
        raise RuntimeError("frozen selection size changed")
    checkpoint = Path(config["checkpoint"])
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    output_dir.mkdir(parents=True)
    runtime = {
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "selection_source_config": str(source_config_path),
        "selection_source_config_sha256": _sha256(source_config_path),
        "selection_manifest_sha256": _sha256(Path(source_config["selection_manifest"])),
        "parent_results_sha256": source_config["parent_results_sha256"],
        "checkpoint_sha256": _sha256(checkpoint),
        "git_head": _git_value("rev-parse", "HEAD"),
        "config": config,
    }
    _write_json(output_dir / "config.json", runtime)

    condition_rows: list[dict[str, Any]] = []
    with (
        (output_dir / "conditions.jsonl").open("x", encoding="utf-8") as condition_handle,
        (output_dir / "experts.jsonl").open("x", encoding="utf-8") as expert_handle,
        (output_dir / "run.log").open("x", encoding="utf-8") as log_handle,
    ):
        for position, selection in enumerate(selected):
            rank = int(selection["sample_rank"])
            dataset_index = int(selection["dataset_index"])
            epsilon = float(selection["epsilon"])
            image, _ = dataset[dataset_index]
            x = image.unsqueeze(0).double()
            lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
            router_net = build_act_router_program(
                model,
                center=x,
                lower=lower,
                upper=upper,
            )
            propagation = _propagate_component(router_net)
            router_hz = propagation.output_hz
            if not isinstance(router_hz, SparseHZono) or not router_hz.exact:
                raise RuntimeError("frozen exact-router study requires an exact SparseHZono")
            modes = ["fast", "exact"] if position % 2 == 0 else ["exact", "fast"]
            for order_index, mode in enumerate(modes):
                row = _run_mode(
                    router_hz,
                    rank=rank,
                    dataset_index=dataset_index,
                    epsilon=epsilon,
                    mode=mode,
                    top_k=int(model.spec.top_k),
                    support_time_limit=float(config["support_time_limit_seconds"]),
                    feasibility_time_limit=float(config["feasibility_time_limit_seconds"]),
                    expert_handle=expert_handle,
                )
                row["pair_order_index"] = order_index
                condition_rows.append(row)
                _append_json(condition_handle, row)
                log_handle.write(
                    f"rank={rank} mode={mode} complete={row['complete']} "
                    f"candidates={row['candidate_experts']} selectors={row['selector_binaries']} "
                    f"nodes={row['mip_nodes']} seconds={row['total_seconds']:.6f}\n"
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())

    summary = summarize_conditions(condition_rows)
    _write_json(output_dir / "summary.json", summary)
    return {"output_dir": str(output_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(run(Path(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
