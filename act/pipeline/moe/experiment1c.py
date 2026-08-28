# ===- act/pipeline/moe/experiment1c.py - Route-A Closure Diagnostics -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Experiment 1C: UNKNOWN taxonomy and exact route-boundary diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from act.back_end.moe import (
    analyze_candidates,
    analyze_topk_sets,
    build_act_moe_program,
    build_act_router_program,
    condition_topk_membership,
    load_output_moe_checkpoint,
)
from act.back_end.solver.solver_hz import SparseHZono, hz_check_feasibility
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _git_value,
    _inside,
    _propagate_component,
    _sha256,
    _solve_output,
    _write_json,
)
from act.pipeline.moe.train import _device, _load_dataset
from act.util.stats import VerifyResult, VerifyStatus


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/experiment1c_bal010.json"
REASONS = {
    "SAFE_PROVED",
    "UNSAFE_FULL_FORWARD",
    "UNKNOWN_GATE_SUFFICIENCY",
    "UNKNOWN_EXPERT_WITNESS_NOT_LIFTED",
    "UNKNOWN_SOLVER_LIMIT",
    "UNKNOWN_NUMERICAL",
    "TIMEOUT_SUPPORT",
    "TIMEOUT_EXPERT_SOLVE",
}
EPSILON_ORDER = {"0.25/255": 0, "0.5/255": 1, "1/255": 2, "2/255": 3}
DIAGNOSTIC_FIELDS = (
    "sample_rank",
    "dataset_index",
    "label",
    "clean_prediction",
    "clean_topk_set",
    "source_status",
    "source_epsilon_label",
    "epsilon",
    "epsilon_multiplier",
    "bracket_lower",
    "bracket_upper",
    "candidate_experts",
    "feasible_route_sets",
    "status",
    "reason",
    "full_model_witness_valid",
    "counterexample_prediction",
    "counterexample_topk_set",
    "candidate_seconds",
    "tightening_seconds",
    "solve_seconds",
    "total_seconds",
    "monotonic_inference",
    "branches",
)


def _json_line(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _csv_row(value: dict[str, Any]) -> dict[str, Any]:
    bracket = value.get("bracket") or {}
    nested = {
        "clean_topk_set",
        "candidate_experts",
        "feasible_route_sets",
        "counterexample_topk_set",
        "monotonic_inference",
        "branches",
    }
    row = {field: value.get(field) for field in DIAGNOSTIC_FIELDS}
    row["bracket_lower"] = bracket.get("lower")
    row["bracket_upper"] = bracket.get("upper")
    for field in nested:
        row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
    return row


def _log_line(handle, message: str) -> None:
    handle.write(message.rstrip() + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _select_diagnostic_samples(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if row["phase"] == "fixed"
        and row["route_set_unstable"] == "True"
        and row["overall_status"] in {"UNKNOWN", "TIMEOUT"}
    ]
    eligible.sort(
        key=lambda row: (
            int(row["sample_rank"]),
            EPSILON_ORDER[row["epsilon_label"]],
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    all_fixed = [row for row in rows if row["phase"] == "fixed"]
    for row in eligible:
        rank = int(row["sample_rank"])
        if rank in seen:
            continue
        seen.add(rank)
        unstable = [
            candidate
            for candidate in all_fixed
            if int(candidate["sample_rank"]) == rank
            and candidate["route_set_unstable"] == "True"
        ]
        unstable.sort(key=lambda item: float(item["epsilon"]))
        selected.append(
            {
                "sample_rank": rank,
                "dataset_index": int(row["dataset_index"]),
                "source_status": row["overall_status"],
                "source_epsilon_label": row["epsilon_label"],
                "first_exact_unstable_epsilon": float(unstable[0]["epsilon"]),
            }
        )
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise RuntimeError(f"only {len(selected)} eligible diagnostic samples")
    return selected


def _router_route_change(
    model,
    x: torch.Tensor,
    clean_set: tuple[int, ...],
    epsilon: float,
    *,
    query_timeout: float,
) -> dict[str, Any]:
    lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
    net = build_act_router_program(
        model,
        center=x,
        lower=lower,
        upper=upper,
    )
    started = time.monotonic()
    propagation = _propagate_component(net)
    if not isinstance(propagation.output_hz, SparseHZono) or not propagation.output_hz.exact:
        raise RuntimeError("route-radius exact label requires an exact sparse router HZ")
    entering_experts: list[int] = []
    branch_statuses: dict[int, str] = {}
    unresolved = False
    for expert in range(model.spec.num_experts):
        if expert in clean_set:
            continue
        membership = condition_topk_membership(
            propagation.output_hz,
            expert,
            model.spec.top_k,
        )
        result = hz_check_feasibility(
            membership.hz,
            time_limit=query_timeout,
        )
        branch_statuses[expert] = result.status
        if result.status == "feasible":
            entering_experts.append(expert)
            break
        if result.status == "unknown":
            unresolved = True
    status = "unstable" if entering_experts else ("unknown" if unresolved else "stable")
    return {
        "status": status,
        "entering_experts": entering_experts,
        "branch_statuses": branch_statuses,
        "elapsed": time.monotonic() - started,
    }


def exact_route_change_bracket(
    model,
    x: torch.Tensor,
    clean_set: Sequence[int],
    initial_upper: float,
    *,
    steps: int,
    query_timeout: float,
) -> dict[str, Any]:
    """Return a certified stable/unstable bracket for top-k set change."""
    clean = tuple(sorted(int(value) for value in clean_set))
    lower, upper = 0.0, float(initial_upper)
    with torch.no_grad():
        scores = model.router(x).reshape(-1)
    order = torch.argsort(scores, descending=True, stable=True)
    concrete_set = tuple(sorted(int(value) for value in order[: model.spec.top_k]))
    if concrete_set != clean:
        raise RuntimeError("clean route set disagrees with router scores")
    if model.spec.top_k < model.spec.num_experts:
        point_margin = float(
            (
                scores[order[model.spec.top_k - 1]]
                - scores[order[model.spec.top_k]]
            ).item()
        )
    else:
        point_margin = float("inf")
    if point_margin <= 0.0:
        raise RuntimeError("clean point has a top-k boundary tie")
    lower_report = {
        "status": "stable",
        "entering_experts": [],
        "branch_statuses": {},
        "elapsed": 0.0,
        "certificate": "strict_clean_router_margin",
        "router_margin": point_margin,
    }
    upper_report = _router_route_change(
        model, x, clean, upper, query_timeout=query_timeout
    )
    if upper_report["status"] != "unstable":
        raise RuntimeError("initial route-radius upper endpoint is not unstable")
    history = [
        {"epsilon": lower, **lower_report},
        {"epsilon": upper, **upper_report},
    ]
    for _ in range(int(steps)):
        middle = (lower + upper) / 2.0
        report = _router_route_change(
            model, x, clean, middle, query_timeout=query_timeout
        )
        history.append({"epsilon": middle, **report})
        if report["status"] == "unknown":
            raise RuntimeError("route-radius feasibility bisection was undecided")
        if report["status"] == "unstable":
            upper, upper_report = middle, report
        else:
            lower, lower_report = middle, report
    return {
        "lower": lower,
        "upper": upper,
        "lower_status": "stable",
        "upper_status": "unstable",
        "steps": int(steps),
        "queries": len(history),
        "seconds": sum(float(item["elapsed"]) for item in history),
        "upper_entering_experts": upper_report["entering_experts"],
        "history": history,
    }


def _prediction_attack(
    model,
    clean: torch.Tensor,
    clean_prediction: int,
    epsilon: float,
    *,
    steps: int,
    restarts: int,
    seed: int,
) -> torch.Tensor | None:
    lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
    label = torch.tensor([clean_prediction], device=clean.device)
    for restart in range(restarts):
        torch.manual_seed(seed + restart)
        if clean.is_cuda:
            torch.cuda.manual_seed_all(seed + restart)
        value = lower + torch.rand_like(clean) * (upper - lower)
        for _ in range(steps):
            value.requires_grad_(True)
            output = model(value)
            loss = F.cross_entropy(output, label)
            gradient = torch.autograd.grad(loss, value)[0]
            value = value.detach() + (epsilon / 4.0) * gradient.sign()
            value = torch.maximum(torch.minimum(value, upper), lower)
        with torch.no_grad():
            if int(model(value).argmax(dim=1).item()) != clean_prediction:
                return value.detach().cpu()[0]
    return None


def _property_margin_lower(bounds, prediction: int) -> float:
    lower = bounds.lb.reshape(1, -1)[0]
    upper = bounds.ub.reshape(1, -1)[0]
    others = torch.cat((upper[:prediction], upper[prediction + 1 :]))
    return float((lower[prediction] - others.max()).item())


def _support_summary(stats: Sequence[dict[str, object]]) -> dict[str, Any]:
    return {
        "layers": list(stats),
        "fast_unstable": sum(int(item["fast_unstable"]) for item in stats),
        "after_lp_unstable": sum(int(item["after_lp_unstable"]) for item in stats),
        "after_milp_unstable": sum(int(item["after_milp_unstable"]) for item in stats),
        "lp_eliminated": sum(int(item["lp_eliminated"]) for item in stats),
        "milp_eliminated": sum(int(item["milp_eliminated"]) for item in stats),
        "seconds": sum(
            float(item["lp_seconds"]) + float(item["milp_seconds"])
            for item in stats
        ),
        "fallback_sides": sum(
            int(item["lp_fallback_sides"]) + int(item["milp_fallback_sides"])
            for item in stats
        ),
    }


def _branch_reason(
    result: VerifyResult,
    *,
    full_witness_valid: bool,
    clean_expert_prediction: int,
    clean_prediction: int,
    support: dict[str, Any],
) -> str:
    if result.status == VerifyStatus.CERTIFIED:
        return "SAFE_PROVED"
    if result.status == VerifyStatus.FALSIFIED:
        if full_witness_valid:
            return "UNSAFE_FULL_FORWARD"
        if clean_expert_prediction != clean_prediction:
            return "UNKNOWN_GATE_SUFFICIENCY"
        return "UNKNOWN_EXPERT_WITNESS_NOT_LIFTED"
    if result.status == VerifyStatus.TIMEOUT:
        return "TIMEOUT_EXPERT_SOLVE"
    if support["fallback_sides"]:
        return "TIMEOUT_SUPPORT"
    if result.status in {
        VerifyStatus.VERIFIER_ERROR,
        VerifyStatus.MODEL_INFER_FAILURE,
    }:
        return "UNKNOWN_NUMERICAL"
    detail = str(result.metadata.get("reason", ""))
    if "failed" in detail or "mismatch" in detail or "missing" in detail:
        return "UNKNOWN_NUMERICAL"
    return "UNKNOWN_SOLVER_LIMIT"


def _solve_staged(
    propagation,
    output_spec,
    *,
    input_shape: tuple[int, ...],
    low_budget: float,
    escalation_budget: float,
) -> tuple[VerifyResult, list[dict[str, Any]]]:
    stages: list[dict[str, Any]] = []
    result = _solve_output(
        propagation,
        output_spec,
        input_shape=input_shape,
        time_limit=low_budget,
    )
    stages.append(
        {
            "stage": "low_budget",
            "budget": low_budget,
            "status": result.status.value,
            "metadata": dict(result.metadata),
        }
    )
    if result.status in {VerifyStatus.UNKNOWN, VerifyStatus.TIMEOUT}:
        result = _solve_output(
            propagation,
            output_spec,
            input_shape=input_shape,
            time_limit=escalation_budget,
        )
        stages.append(
            {
                "stage": "escalation",
                "budget": escalation_budget,
                "status": result.status.value,
                "metadata": dict(result.metadata),
            }
        )
    return result, stages


def diagnose_radius(
    *,
    model,
    x: torch.Tensor,
    label: int,
    clean_prediction: int,
    clean_set: Sequence[int],
    epsilon: float,
    epsilon_multiplier: float,
    bracket: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[clean_prediction])
    program = build_act_moe_program(
        model,
        center=x,
        lower=lower,
        upper=upper,
        output_spec=output_spec,
    )
    candidate_started = time.monotonic()
    router = _propagate_component(program.router)
    exact = isinstance(router.output_hz, SparseHZono) and router.output_hz.exact
    candidates = analyze_candidates(
        router.output_hz,
        model.spec.top_k,
        input_hz=router.input_hz,
        time_limit_per_expert=float(config["candidate_query_timeout"]),
        router_exact=exact,
    )
    route_sets = analyze_topk_sets(
        router.output_hz,
        model.spec.top_k,
        time_limit_per_set=float(config["candidate_query_timeout"]),
        router_exact=exact,
    )
    candidate_seconds = time.monotonic() - candidate_started
    if not candidates.minimal or not route_sets.exact:
        return {
            "label": label,
            "clean_prediction": clean_prediction,
            "clean_topk_set": sorted(int(value) for value in clean_set),
            "epsilon": epsilon,
            "epsilon_multiplier": epsilon_multiplier,
            "status": "UNKNOWN",
            "reason": "UNKNOWN_SOLVER_LIMIT",
            "phase": "candidate_feasibility",
            "candidate_experts": list(candidates.candidates),
            "candidate_unresolved": list(candidates.unresolved),
            "feasible_route_sets": [list(values) for values in route_sets.feasible],
            "route_set_unresolved": [list(values) for values in route_sets.unresolved],
            "full_model_witness_valid": False,
            "counterexample_prediction": None,
            "counterexample_topk_set": None,
            "candidate_seconds": candidate_seconds,
            "tightening_seconds": 0.0,
            "solve_seconds": 0.0,
            "bracket": bracket,
            "branches": [],
            "total_seconds": time.monotonic() - started,
            "monotonic_inference": None,
        }

    support_config = HybridZConfig(
        max_input_dim=1024,
        guarded_support_enabled=True,
        guarded_support_lp_neurons=int(config["support"]["lp_neurons"]),
        guarded_support_milp_neurons=int(config["support"]["milp_neurons"]),
        guarded_support_lp_time_limit=float(config["support"]["lp_time_limit"]),
        guarded_support_milp_time_limit=float(config["support"]["milp_time_limit"]),
    )
    by_expert = {branch.expert: branch for branch in candidates.branches}
    branch_rows: list[dict[str, Any]] = []
    full_witness = None
    tightening_seconds = solve_seconds = 0.0
    for expert in candidates.candidates:
        branch = by_expert[expert]
        unguarded = _propagate_component(program.experts[expert])
        tighten_started = time.monotonic()
        guarded = _propagate_component(
            program.experts[expert],
            entry_hz=branch.guarded_input,
            hybridz_config=support_config,
        )
        tightening_seconds += time.monotonic() - tighten_started
        support = _support_summary(guarded.guarded_support)
        solve_started = time.monotonic()
        result, stages = _solve_staged(
            guarded,
            output_spec,
            input_shape=tuple(x.shape),
            low_budget=float(config["solver"]["low_budget_per_branch"]),
            escalation_budget=float(config["solver"]["escalation_budget_per_branch"]),
        )
        solve_seconds += time.monotonic() - solve_started
        validated = _forward_validate(
            model,
            result.counterexample,
            lower=lower,
            upper=upper,
            clean_prediction=clean_prediction,
        )
        with torch.no_grad():
            clean_expert_prediction = int(model.experts[expert](x).argmax(dim=1).item())
        reason = _branch_reason(
            result,
            full_witness_valid=bool(validated["valid"]),
            clean_expert_prediction=clean_expert_prediction,
            clean_prediction=clean_prediction,
            support=support,
        )
        if validated["valid"]:
            full_witness = validated
        route_sets_for_expert = [
            list(values) for values in route_sets.feasible if expert in values
        ]
        branch_rows.append(
            {
                "candidate": expert,
                "feasible_route_sets": route_sets_for_expert,
                "branch_status": result.status.value,
                "property_lower_bound": _property_margin_lower(
                    guarded.output_bounds, clean_prediction
                ),
                "property_bound_kind": "fast_hz_interval",
                "solver_status": result.status.value,
                "solver_reason": result.metadata.get("reason"),
                "solver_gap": result.metadata.get("mip_gap"),
                "binary_width_before_guard": (
                    router.binary_width
                    + branch.selection_binaries
                    + unguarded.binary_width
                ),
                "binary_width_after_guard": guarded.binary_width,
                "expert_relu_binaries_before_guard": unguarded.unstable_total,
                "expert_relu_binaries_after_guard": max(
                    0, guarded.binary_width - branch.guarded_input.n_bin
                ),
                "router_binary_count": router.binary_width,
                "route_membership_binary_count": branch.selection_binaries,
                "violation_witness_found": result.status == VerifyStatus.FALSIFIED,
                "full_model_witness_valid": bool(validated["valid"]),
                "clean_expert_prediction": clean_expert_prediction,
                "unknown_reason": reason,
                "support": support,
                "candidate_time": candidate_seconds,
                "tightening_time": guarded.elapsed,
                "solve_time": sum(
                    float(stage["metadata"].get("elapsed", 0.0)) for stage in stages
                ),
                "solve_stages": stages,
            }
        )
    reasons = [row["unknown_reason"] for row in branch_rows]
    if full_witness is not None:
        status, reason = "UNSAFE", "UNSAFE_FULL_FORWARD"
    elif reasons and all(value == "SAFE_PROVED" for value in reasons):
        status, reason = "SAFE", "SAFE_PROVED"
    else:
        priority = (
            "UNKNOWN_GATE_SUFFICIENCY",
            "UNKNOWN_EXPERT_WITNESS_NOT_LIFTED",
            "TIMEOUT_SUPPORT",
            "TIMEOUT_EXPERT_SOLVE",
            "UNKNOWN_NUMERICAL",
            "UNKNOWN_SOLVER_LIMIT",
        )
        reason = next((value for value in priority if value in reasons), "UNKNOWN_SOLVER_LIMIT")
        status = "TIMEOUT" if reason.startswith("TIMEOUT_") else "UNKNOWN"
    if reason not in REASONS:
        raise RuntimeError(f"unregistered diagnostic reason {reason}")
    return {
        "label": label,
        "clean_prediction": clean_prediction,
        "clean_topk_set": sorted(int(value) for value in clean_set),
        "epsilon": epsilon,
        "epsilon_multiplier": epsilon_multiplier,
        "bracket": bracket,
        "candidate_experts": list(candidates.candidates),
        "feasible_route_sets": [list(values) for values in route_sets.feasible],
        "status": status,
        "reason": reason,
        "full_model_witness_valid": full_witness is not None,
        "counterexample_prediction": full_witness["prediction"] if full_witness else None,
        "counterexample_topk_set": full_witness["topk_set"] if full_witness else None,
        "branches": branch_rows,
        "candidate_seconds": candidate_seconds,
        "tightening_seconds": tightening_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": time.monotonic() - started,
        "monotonic_inference": None,
    }


def _inferred_row(
    source: dict[str, Any],
    *,
    epsilon: float,
    multiplier: float,
    rule: str,
) -> dict[str, Any]:
    return {
        "label": source.get("label"),
        "clean_prediction": source.get("clean_prediction"),
        "clean_topk_set": source.get("clean_topk_set"),
        "epsilon": epsilon,
        "epsilon_multiplier": multiplier,
        "bracket": source.get("bracket"),
        "candidate_experts": None,
        "feasible_route_sets": None,
        "status": source["status"],
        "reason": source["reason"],
        "full_model_witness_valid": source.get("full_model_witness_valid", False),
        "counterexample_prediction": source.get("counterexample_prediction"),
        "counterexample_topk_set": source.get("counterexample_topk_set"),
        "branches": [],
        "candidate_seconds": 0.0,
        "tightening_seconds": 0.0,
        "solve_seconds": 0.0,
        "total_seconds": 0.0,
        "monotonic_inference": {
            "rule": rule,
            "source_multiplier": source["epsilon_multiplier"],
        },
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    support = [
        branch["support"]
        for row in rows
        for branch in row.get("branches", [])
    ]
    return {
        "rows": len(rows),
        "samples": len({row["sample_rank"] for row in rows}),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "branch_reason_counts": dict(
            Counter(
                branch["unknown_reason"]
                for row in rows
                for branch in row.get("branches", [])
            )
        ),
        "monotonic_inferred_rows": sum(
            row.get("monotonic_inference") is not None for row in rows
        ),
        "support_fast_unstable": sum(item["fast_unstable"] for item in support),
        "support_after_lp_unstable": sum(item["after_lp_unstable"] for item in support),
        "support_after_milp_unstable": sum(item["after_milp_unstable"] for item in support),
        "support_lp_eliminated": sum(item["lp_eliminated"] for item in support),
        "support_milp_eliminated": sum(item["milp_eliminated"] for item in support),
        "support_seconds": sum(item["seconds"] for item in support),
        "full_forward_unsafe_rows": sum(
            row["status"] == "UNSAFE" and row["full_model_witness_valid"] for row in rows
        ),
    }


def run(args) -> dict[str, Any]:
    config_path = _inside(Path(args.config), PROJECT_ROOT)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    source_csv = _inside(Path(config["development_results_csv"]), WRITE_ROOT)
    indices_path = _inside(Path(config["development_sample_indices"]), WRITE_ROOT)
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("Experiment 1C must run on feat/moe-route-verification")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("Experiment 1C requires a clean feature-branch worktree")
    expected_python = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError("Experiment 1C requires the act-py312 conda environment")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "selection": output_dir / "selection.json",
        "runtime_config": output_dir / "config.json",
        "jsonl": output_dir / "diagnostics.jsonl",
        "csv": output_dir / "diagnostics.csv",
        "summary": output_dir / "summary.json",
        "log": output_dir / "experiment1c.log",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise RuntimeError(f"Experiment 1C refuses to overwrite outputs: {existing}")
    selected = _select_diagnostic_samples(source_csv, int(config["diagnostic_samples"]))
    saved_indices = json.load(indices_path.open(encoding="utf-8"))["indices"]
    for item in selected:
        if saved_indices[item["sample_rank"]] != item["dataset_index"]:
            raise RuntimeError("development rank/index mapping changed")
    _write_json(output_paths["selection"], {"samples": selected})

    device = _device(config["device"])
    attack_model, payload = load_output_moe_checkpoint(checkpoint, map_location=device)
    attack_model.to(device).eval()
    model, _ = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    runtime = {
        "source_config": str(config_path),
        "git_head": _git_value("rev-parse", "HEAD"),
        "checkpoint_sha256": _sha256(checkpoint),
        "development_results_sha256": _sha256(source_csv),
        "development_sample_indices_sha256": _sha256(indices_path),
        "source_config_sha256": _sha256(config_path),
        "config": config,
    }
    _write_json(output_paths["runtime_config"], runtime)

    results_path = output_paths["jsonl"]
    rows: list[dict[str, Any]] = []
    with (
        results_path.open("x", encoding="utf-8") as jsonl_handle,
        output_paths["csv"].open("x", newline="", encoding="utf-8") as csv_handle,
        output_paths["log"].open("x", encoding="utf-8") as log_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        _log_line(
            log_handle,
            f"START git_head={runtime['git_head']} samples={len(selected)}",
        )
        for selection in selected:
            rank = selection["sample_rank"]
            index = selection["dataset_index"]
            image, label = dataset[index]
            x = image.unsqueeze(0).double()
            with torch.no_grad():
                output, decision = model.forward_with_routing(x)
            clean_prediction = int(output.argmax(dim=1).item())
            if clean_prediction != int(label):
                raise RuntimeError(f"development rank {rank} is no longer clean-correct")
            clean_set = sorted(int(value) for value in decision.indices[0].tolist())
            bracket = exact_route_change_bracket(
                model,
                x,
                clean_set,
                selection["first_exact_unstable_epsilon"],
                steps=int(config["route_radius"]["bisection_steps"]),
                query_timeout=float(config["route_radius"]["query_timeout"]),
            )
            multipliers = [float(value) for value in config["route_radius"]["multipliers"]]
            radii = {value: value * bracket["upper"] for value in multipliers}
            sample_rows: dict[float, dict[str, Any]] = {}
            attack_dtype = next(attack_model.parameters()).dtype
            attack_x = image.unsqueeze(0).to(device=device, dtype=attack_dtype)
            for multiplier in sorted(multipliers):
                witness = _prediction_attack(
                    attack_model,
                    attack_x,
                    clean_prediction,
                    radii[multiplier],
                    steps=int(config["attack"]["steps"]),
                    restarts=int(config["attack"]["restarts"]),
                    seed=int(config["seed"]) + rank * 100,
                )
                if witness is None:
                    continue
                validated = _forward_validate(
                    model,
                    witness,
                    lower=(x - radii[multiplier]).clamp(0, 1),
                    upper=(x + radii[multiplier]).clamp(0, 1),
                    clean_prediction=clean_prediction,
                )
                if not validated["valid"]:
                    raise RuntimeError("prediction attack witness failed full-model validation")
                base = {
                    "label": int(label),
                    "clean_prediction": clean_prediction,
                    "clean_topk_set": clean_set,
                    "epsilon": radii[multiplier],
                    "epsilon_multiplier": multiplier,
                    "bracket": bracket,
                    "candidate_experts": None,
                    "feasible_route_sets": None,
                    "status": "UNSAFE",
                    "reason": "UNSAFE_FULL_FORWARD",
                    "full_model_witness_valid": True,
                    "counterexample_prediction": validated["prediction"],
                    "counterexample_topk_set": validated["topk_set"],
                    "branches": [],
                    "candidate_seconds": 0.0,
                    "tightening_seconds": 0.0,
                    "solve_seconds": 0.0,
                    "total_seconds": 0.0,
                    "witness_source": "prediction_attack",
                    "monotonic_inference": None,
                }
                sample_rows[multiplier] = base
                for larger in multipliers:
                    if larger > multiplier and larger not in sample_rows:
                        sample_rows[larger] = _inferred_row(
                            base,
                            epsilon=radii[larger],
                            multiplier=larger,
                            rule="smaller_radius_validated_unsafe_implies_larger_unsafe",
                        )
                break
            for multiplier in sorted(multipliers, reverse=True):
                if multiplier in sample_rows:
                    continue
                diagnosed = diagnose_radius(
                    model=model,
                    x=x,
                    label=int(label),
                    clean_prediction=clean_prediction,
                    clean_set=clean_set,
                    epsilon=radii[multiplier],
                    epsilon_multiplier=multiplier,
                    bracket=bracket,
                    config=config,
                )
                sample_rows[multiplier] = diagnosed
                if diagnosed["status"] == "SAFE":
                    for smaller in multipliers:
                        if smaller < multiplier and smaller not in sample_rows:
                            sample_rows[smaller] = _inferred_row(
                                diagnosed,
                                epsilon=radii[smaller],
                                multiplier=smaller,
                                rule="larger_radius_safe_implies_smaller_safe",
                            )
            unsafe_sources = [
                row for row in sample_rows.values() if row["status"] == "UNSAFE"
            ]
            if unsafe_sources:
                smallest = min(unsafe_sources, key=lambda row: row["epsilon"])
                for multiplier in multipliers:
                    replaceable = sample_rows[multiplier]["status"] in {
                        "UNKNOWN",
                        "TIMEOUT",
                    }
                    if radii[multiplier] >= smallest["epsilon"] and replaceable:
                        sample_rows[multiplier] = _inferred_row(
                            smallest,
                            epsilon=radii[multiplier],
                            multiplier=multiplier,
                            rule="smaller_radius_validated_unsafe_replaces_larger_unknown",
                        )
            for multiplier in sorted(multipliers):
                row = sample_rows[multiplier]
                row.update(
                    sample_rank=rank,
                    dataset_index=index,
                    source_status=selection["source_status"],
                    source_epsilon_label=selection["source_epsilon_label"],
                )
                rows.append(row)
                _json_line(jsonl_handle, row)
                writer.writerow(_csv_row(row))
                csv_handle.flush()
                os.fsync(csv_handle.fileno())
            sample_counts = Counter(
                sample_rows[multiplier]["status"] for multiplier in multipliers
            )
            _log_line(
                log_handle,
                f"SAMPLE rank={rank} dataset_index={index} statuses={dict(sample_counts)}",
            )
        _log_line(log_handle, f"COMPLETE rows={len(rows)}")
    summary = _summarize(rows)
    summary.update(
        {
            "results": str(results_path),
            "selection": str(output_paths["selection"]),
            "csv": str(output_paths["csv"]),
            "log": str(output_paths["log"]),
            "partial_rows_flushed": True,
        }
    )
    _write_json(output_paths["summary"], summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Experiment 1C closure diagnostics")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
