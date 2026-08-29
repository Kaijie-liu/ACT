# ===- act/pipeline/moe/experiment1d.py - Applicable Closure -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Experiment 1D: close the frozen applicable-but-unresolved cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from act.back_end.moe import (
    SAFE_WEIGHTED_RANGE,
    WeightedTop2F0Decision,
    UNKNOWN_WEIGHTED_NUMERICAL,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    analyze_candidates,
    analyze_topk_sets,
    build_act_moe_program,
    build_weighted_top2_f0,
    compute_weighted_top2_gate_range,
    condition_topk_set,
    guarded_input_topk_set,
    linear_safety_rows,
    load_output_moe_checkpoint,
    solve_weighted_top2_f0,
)
from act.back_end.solver.solver_hz import SparseHZono, hz_numerical_policy_manifest
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _git_value,
    _inside,
    _new_incremental_property_session,
    _propagate_component,
    _sha256,
    _solve_output,
    _write_json,
    shared_input_pair_propagation,
)
from act.pipeline.moe.experiment1c import _branch_reason, _support_summary
from act.pipeline.moe.experiment1f0 import (
    _pair_reason,
    _row_reason,
    _support_record,
    _support_status,
    _validate_candidate,
)
from act.pipeline.moe.train import _load_dataset
from act.util.path_config import get_torchvision_data_root
from act.util.stats import VerifyStatus


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/experiment1d_bal010_r2.json"
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
SEMANTIC_REASONS = {
    "UNKNOWN_GATE_SUFFICIENCY",
    "UNKNOWN_EXPERT_WITNESS_NOT_LIFTED",
}
CSV_FIELDS = (
    "sample_rank", "dataset_index", "category", "reuse_mode", "epsilon",
    "parent_status", "parent_reason", "status", "reason", "newly_solved",
    "full_model_witness_valid", "witness_path", "witness_sha256",
    "reused_property_rows", "rerun_property_rows", "checkpoint_recorded",
    "total_seconds", "error",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_frozen_selection(config: dict[str, Any]) -> list[dict[str, Any]]:
    parent_path = _inside(Path(config["parent_results_jsonl"]), WRITE_ROOT)
    manifest_path = _inside(Path(config["selection_manifest"]), PROJECT_ROOT)
    if _sha256(parent_path) != config["parent_results_sha256"]:
        raise RuntimeError("confirmatory parent artifact hash changed")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["parent_results_sha256"] != config["parent_results_sha256"]:
        raise RuntimeError("selection manifest parent hash mismatch")
    raw_lines = parent_path.read_bytes().splitlines(keepends=True)
    selected: list[dict[str, Any]] = []
    for item in manifest["rows"]:
        line_number = int(item["parent_line_number"])
        raw = raw_lines[line_number - 1]
        if _sha256_bytes(raw) != item["parent_row_sha256"]:
            raise RuntimeError(f"parent row hash changed at line {line_number}")
        parent = json.loads(raw)
        for key in ("sample_rank", "dataset_index"):
            if int(parent[key]) != int(item[key]):
                raise RuntimeError(f"selection {key} mismatch at line {line_number}")
        if parent["status"] not in {"UNKNOWN", "TIMEOUT"}:
            raise RuntimeError("selection contains a solved parent row")
        if parent["reason"] == "NO_ROUTE_BOUNDARY_WITHIN_SEARCH":
            raise RuntimeError("selection contains an inapplicable no-boundary row")
        epsilon = parent.get("epsilon", item.get("epsilon"))
        if epsilon is None:
            raise RuntimeError(f"frozen epsilon missing for rank {item['sample_rank']}")
        if item["reuse_mode"] == "fixed_radius_rebuild":
            progress_path = (
                parent_path.parent / "row_work"
                / f"rank{int(item['sample_rank'])}" / "progress.json"
            )
            if _sha256(progress_path) != item.get("progress_sha256"):
                raise RuntimeError("hard-deadline progress artifact hash changed")
            with progress_path.open(encoding="utf-8") as handle:
                progress = json.load(handle)
            if not abs(float(progress["epsilon"]) - float(epsilon)) <= 1e-15:
                raise RuntimeError("hard-deadline frozen epsilon changed")
        selected.append({**item, "epsilon": float(epsilon), "parent": parent})
    if len(selected) != int(config["expected_rows"]):
        raise RuntimeError("Experiment 1D frozen row count changed")
    expected = Counter(item["category"] for item in selected)
    if dict(expected) != manifest["category_counts"]:
        raise RuntimeError("Experiment 1D category counts changed")
    return selected


def _support_signature(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "relu_binaries", "binary_width", "fast_unstable", "after_lp_unstable",
        "after_milp_unstable", "lp_eliminated", "milp_eliminated",
        "fallback_sides",
    )
    return {key: record.get(key) for key in keys}


def _assert_support_identity(
    actual: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    actual_signature = _support_signature(actual)
    parent_signature = _support_signature(parent)
    required_keys = tuple(
        key for key in actual_signature if key != "fallback_sides"
    )
    if any(
        actual_signature[key] != parent_signature[key]
        for key in required_keys
    ):
        raise RuntimeError("rematerialized guarded-support signature changed")
    return {
        "structural_identity": True,
        "actual": actual_signature,
        "parent": parent_signature,
        "fallback_side_drift": (
            actual_signature["fallback_sides"]
            - parent_signature["fallback_sides"]
        ),
        "interpretation": (
            "time-limited side-status drift is recorded but is not a "
            "structural binary-elimination change"
        ),
    }


def _gate_support_record(propagation, shared_binary_width: int) -> dict[str, Any]:
    """Match the confirmatory gate branch's post-support binary universe."""
    return {
        **_support_summary(propagation.guarded_support),
        "relu_binaries": max(
            0, int(propagation.binary_width) - int(shared_binary_width)
        ),
        "binary_width": int(propagation.binary_width),
    }


class RowRecorder:
    def __init__(self, work_dir: Path, checkpoint_seconds: float):
        self.work_dir = work_dir
        self.started = time.monotonic()
        self.checkpoint_seconds = float(checkpoint_seconds)
        self.latest: dict[str, Any] = {}
        self.checkpoint_recorded = False

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def progress(self, *, active: dict[str, Any], solver: dict[str, Any]) -> None:
        self.latest = {
            "elapsed_seconds": self.elapsed(),
            "active_branch": active,
            "solver": solver,
        }
        _write_json(self.work_dir / "progress.json", self.latest)
        if self.elapsed() >= self.checkpoint_seconds and not self.checkpoint_recorded:
            _write_json(
                self.work_dir / "checkpoint_300.json",
                {
                    **self.latest,
                    "target_seconds": self.checkpoint_seconds,
                    "capture_semantics": "first black-box solver return at or after target",
                },
            )
            self.checkpoint_recorded = True

    def finish_checkpoint(self, status: str) -> None:
        if self.checkpoint_recorded:
            return
        _write_json(
            self.work_dir / "checkpoint_300.json",
            {
                **self.latest,
                "elapsed_seconds": self.elapsed(),
                "target_seconds": self.checkpoint_seconds,
                "status": "COMPLETED_BEFORE_CHECKPOINT",
                "final_status": status,
            },
        )
        self.checkpoint_recorded = True


def _solver_snapshot(decision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "incumbent": decision.solver_primal_objective,
        "dual_bound": decision.solver_dual_objective,
        "gap": decision.solver_gap,
        "certified_lower_bound": decision.minimum,
        "solver_status": decision.solver_status,
        "bound_semantics": "full scalar objective including affine center",
    }


def _gate_solver_snapshot(result) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "reason": result.metadata.get("reason"),
        "incumbent": None,
        "dual_bound": None,
        "gap": result.metadata.get("mip_gap"),
        "solver_status": result.status.value,
        "bound_semantics": "not_applicable_feasibility_encoding",
        "metadata": dict(result.metadata),
    }


def _save_witness(
    work_dir: Path,
    rank: int,
    kind: str,
    identity: str,
    candidate: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    relative = Path("witnesses") / f"rank{rank}_{kind}_{identity}.pt"
    path = work_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    torch.save({"input": candidate.detach().cpu(), "metadata": metadata}, path)
    return str(relative), _sha256(path)


def _row_context(selection, model, dataset, config):
    parent = selection["parent"]
    rank, index = int(selection["sample_rank"]), int(selection["dataset_index"])
    epsilon = float(selection["epsilon"])
    image, label = dataset[index]
    x = image.unsqueeze(0).double()
    lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
    with torch.no_grad():
        clean_output, clean_route = model.forward_with_routing(x)
    prediction = int(clean_output.argmax(dim=1).item())
    clean_set = sorted(int(value) for value in clean_route.indices[0].tolist())
    if prediction != int(label):
        raise RuntimeError(f"rank {rank} is no longer clean-correct")
    if parent.get("clean_prediction") is not None and prediction != int(parent["clean_prediction"]):
        raise RuntimeError("clean prediction changed from parent")
    if parent.get("clean_topk_set") is not None and clean_set != sorted(parent["clean_topk_set"]):
        raise RuntimeError("clean top-k set changed from parent")
    spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[prediction])
    program = build_act_moe_program(
        model, center=x, lower=lower, upper=upper, output_spec=spec
    )
    router = _propagate_component(program.router)
    if not isinstance(router.output_hz, SparseHZono) or not router.output_hz.exact:
        raise RuntimeError("D0 requires an exact router HZ")
    candidates = analyze_candidates(
        router.output_hz,
        model.spec.top_k,
        input_hz=router.input_hz,
        time_limit_per_expert=float(config["candidate_query_timeout"]),
        router_exact=True,
    )
    route_sets = analyze_topk_sets(
        router.output_hz,
        model.spec.top_k,
        time_limit_per_set=float(config["candidate_query_timeout"]),
        router_exact=True,
    )
    if not candidates.minimal or not route_sets.exact:
        raise RuntimeError("frozen candidate or route-set query did not complete")
    actual_candidates = list(candidates.candidates)
    actual_pairs = [list(pair) for pair in route_sets.feasible]
    gate_parent = parent.get("gate") or {}
    f0_parent = parent.get("f0") or {}
    expected_candidates = gate_parent.get("candidate_experts")
    expected_pairs = f0_parent.get("feasible_pairs") or gate_parent.get("feasible_route_sets")
    if expected_candidates is not None and actual_candidates != expected_candidates:
        raise RuntimeError("rematerialized candidate experts changed")
    if expected_pairs is not None and actual_pairs != expected_pairs:
        raise RuntimeError("rematerialized feasible pairs changed")
    return {
        "rank": rank, "index": index, "label": int(label), "epsilon": epsilon,
        "x": x, "lower": lower, "upper": upper, "prediction": prediction,
        "clean_set": clean_set, "spec": spec, "program": program,
        "router": router, "candidates": candidates, "route_sets": route_sets,
        "properties": linear_safety_rows(spec, int(clean_output.shape[1])),
    }


def _support_config(config: dict[str, Any]) -> HybridZConfig:
    support = config["support"]
    return HybridZConfig(
        max_input_dim=1024,
        guarded_support_enabled=True,
        guarded_support_lp_neurons=int(support["lp_neurons"]),
        guarded_support_milp_neurons=int(support["milp_neurons"]),
        guarded_support_lp_time_limit=float(support["lp_time_limit"]),
        guarded_support_milp_time_limit=float(support["milp_time_limit"]),
        guarded_support_solver_backend=str(
            support.get("solver_backend", "scipy")
        ),
        expert_property_solver_backend=str(
            config.get("solver", {}).get("backend", "scipy")
        ),
    )


def _attempt_budgets(recorder: RowRecorder, remaining_items: int, deadline: float) -> tuple[float, float]:
    remaining_wall = max(0.0, deadline - recorder.elapsed() - 2.0)
    allocation = remaining_wall / max(1, int(remaining_items))
    first = min(allocation / 3.0, max(0.0, recorder.checkpoint_seconds - recorder.elapsed()))
    second = max(0.0, allocation - first)
    return max(0.01, first), max(0.0, second)


def _solve_f0_attempt(
    encoding,
    *,
    input_shape: tuple[int, ...],
    time_limit: float,
    tolerance: float,
    backend: str,
    incremental_session=None,
):
    if backend == "scipy":
        if incremental_session is not None:
            raise ValueError("incremental F0 session supplied to scipy path")
        return solve_weighted_top2_f0(
            encoding,
            input_shape=input_shape,
            time_limit=float(time_limit),
            tolerance=float(tolerance),
        )
    if backend != "highspy_incremental":
        raise ValueError(f"unsupported F0 solver backend {backend!r}")
    from act.back_end.moe.incremental_hz_solver import (
        IncrementalHZBranchSolver,
    )

    session = incremental_session or IncrementalHZBranchSolver(
        encoding.output_hz,
        time_limit=float(time_limit),
        relax_binaries=False,
    )
    if session.hz is not encoding.output_hz:
        raise ValueError("incremental F0 session belongs to another augmented HZ")
    result = session.minimize_output(
        0,
        input_hz=encoding.input_hz,
        input_shape=input_shape,
    )
    if (
        result.status == "optimal"
        and result.minimum is not None
        and result.minimum > float(tolerance)
    ):
        status, reason = "SAFE", SAFE_WEIGHTED_RANGE
    elif result.status == "timeout":
        status, reason = "UNKNOWN", UNKNOWN_WEIGHTED_SOLVER_LIMIT
    elif result.status == "optimal":
        status, reason = "UNKNOWN", UNKNOWN_WEIGHTED_RELAXATION
    else:
        status, reason = "UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL
    return WeightedTop2F0Decision(
        status=status,
        reason=reason,
        minimum=result.minimum,
        candidate_objective=result.candidate_objective,
        candidate_input=result.candidate_input,
        solver_status=result.solver_status,
        solver_gap=result.solver_gap,
        elapsed=result.elapsed,
        solver_certified_lower_bound=result.solver_certified_lower_bound,
        solver_bound_kind=result.solver_bound_kind,
        solver_primal_objective=result.solver_primal_objective,
        solver_dual_objective=result.solver_dual_objective,
    )


def _run_f0(
    selection, context, model, work_dir: Path, config, recorder: RowRecorder,
    *, parent_f0: dict[str, Any] | None,
) -> dict[str, Any]:
    parent_pairs = {
        tuple(row["pair"]): row for row in (parent_f0 or {}).get("pairs", [])
    }
    unresolved_total = sum(
        prop.get("status") not in {"SAFE", "UNSAFE"}
        for pair in parent_pairs.values()
        for prop in pair.get("property_rows", [])
    )
    if not parent_pairs:
        unresolved_total = len(context["route_sets"].feasible) * len(context["properties"])
    remaining_items = max(1, unresolved_total)
    pair_rows, selected_witness = [], None
    support_config = _support_config(config)
    reused_properties = rerun_properties = 0
    for pair_values in context["route_sets"].feasible:
        pair = tuple(sorted(int(v) for v in pair_values))
        parent_pair = parent_pairs.get(pair)
        if parent_pair is not None and parent_pair.get("status") == "SAFE":
            reused_properties += len(parent_pair.get("property_rows", []))
            pair_rows.append({**parent_pair, "reused_parent": True})
            continue
        conditioned_router = condition_topk_set(context["router"].output_hz, pair).hz
        guarded_entry = guarded_input_topk_set(
            context["router"].input_hz, context["router"].output_hz, pair
        ).hz
        propagated = shared_input_pair_propagation(
            context["program"].experts[pair[0]],
            context["program"].experts[pair[1]],
            entry_hz=guarded_entry,
            hybridz_config=support_config,
        )
        support_a, support_b = _support_record(propagated.expert_a), _support_record(propagated.expert_b)
        if parent_pair is not None:
            identity_a = _assert_support_identity(
                support_a, parent_pair["expert_a_support"]
            )
            identity_b = _assert_support_identity(
                support_b, parent_pair["expert_b_support"]
            )
        else:
            identity_a = identity_b = None
        parent_props = {
            int(row["property_index"]): row
            for row in (parent_pair or {}).get("property_rows", [])
        }
        property_rows: list[dict[str, Any]] = []
        gate_range = None
        for property_index, (q, constant) in enumerate(context["properties"]):
            parent_prop = parent_props.get(property_index)
            if parent_prop is not None and parent_prop.get("status") in {"SAFE", "UNSAFE"}:
                property_rows.append({**parent_prop, "reused_parent": True})
                reused_properties += 1
                continue
            rerun_properties += 1
            if gate_range is None:
                gate_range = compute_weighted_top2_gate_range(
                    conditioned_router,
                    pair,
                    time_limit=float(
                        config["f0"]["margin_support_seconds"]
                    ),
                )
            encoding = build_weighted_top2_f0(
                propagated.joint, conditioned_router, pair, q, constant,
                difference_time_limit=float(
                    config["f0"]["difference_support_seconds"]
                ),
                gate_range=gate_range,
            )
            first_budget, second_budget = _attempt_budgets(
                recorder, remaining_items, float(config["instance_timeout_seconds"])
            )
            attempts = []
            f0_backend = str(
                config.get("solver", {}).get("f0_backend", "scipy")
            )
            if f0_backend == "highspy_incremental":
                from act.back_end.moe.incremental_hz_solver import (
                    IncrementalHZBranchSolver,
                )

                incremental_session = IncrementalHZBranchSolver(
                    encoding.output_hz,
                    time_limit=first_budget,
                    relax_binaries=False,
                )
            else:
                incremental_session = None
            decision = _solve_f0_attempt(
                encoding,
                input_shape=tuple(context["x"].shape),
                time_limit=first_budget,
                tolerance=float(config["f0"]["safety_tolerance"]),
                backend=f0_backend,
                incremental_session=incremental_session,
            )
            active = {"tier": "F0", "pair": list(pair), "property_index": property_index, "attempt": 1}
            attempt = {"budget": first_budget, **_solver_snapshot(decision)}
            if incremental_session is not None:
                attempt["incremental_hz"] = (
                    incremental_session.telemetry().as_dict()
                )
            recorder.progress(active=active, solver=attempt)
            attempts.append(attempt)
            if decision.status == "UNKNOWN" and decision.reason == UNKNOWN_WEIGHTED_SOLVER_LIMIT and second_budget > 0.01:
                if incremental_session is not None:
                    incremental_session.extend_budget(second_budget)
                decision = _solve_f0_attempt(
                    encoding,
                    input_shape=tuple(context["x"].shape),
                    time_limit=second_budget,
                    tolerance=float(config["f0"]["safety_tolerance"]),
                    backend=f0_backend,
                    incremental_session=incremental_session,
                )
                active["attempt"] = 2
                attempt = {"budget": second_budget, **_solver_snapshot(decision)}
                if incremental_session is not None:
                    attempt["incremental_hz"] = (
                        incremental_session.telemetry().as_dict()
                    )
                recorder.progress(active=active, solver=attempt)
                attempts.append(attempt)
            remaining_items -= 1
            checked = _validate_candidate(
                model, decision.candidate_input, clean=context["x"],
                lower=context["lower"], upper=context["upper"],
                clean_prediction=context["prediction"],
            )
            status = "UNSAFE" if checked["valid"] else decision.status
            reason = UNSAFE_FULL_FORWARD_FALLBACK if checked["valid"] else decision.reason
            witness_path = witness_hash = None
            if checked["valid"]:
                witness_path, witness_hash = _save_witness(
                    work_dir, context["rank"], "f0", f"{pair[0]}_{pair[1]}_{property_index}",
                    decision.candidate_input,
                    {"pair": list(pair), "property_index": property_index,
                     "epsilon": context["epsilon"], "clean_prediction": context["prediction"]},
                )
                selected_witness = {**checked, "path": witness_path, "sha256": witness_hash}
            property_rows.append({
                "property_index": property_index, "status": status, "reason": reason,
                "minimum": decision.minimum, "candidate_objective": decision.candidate_objective,
                "solver_status": decision.solver_status, "solver_gap": decision.solver_gap,
                "solver_certified_lower_bound": decision.solver_certified_lower_bound,
                "solver_bound_kind": decision.solver_bound_kind,
                "solver_primal_objective": decision.solver_primal_objective,
                "solver_dual_objective": decision.solver_dual_objective,
                "margin_bounds": list(encoding.margin_bounds),
                "lambda_bounds": [encoding.bounds.lambda_lower, encoding.bounds.lambda_upper],
                "difference_bounds": [encoding.bounds.difference_lower, encoding.bounds.difference_upper],
                "product_bounds": [encoding.bounds.product_lower, encoding.bounds.product_upper],
                "margin_support": _support_status(encoding.margin_support),
                "difference_support": _support_status(encoding.difference_support),
                "full_model_witness_valid": bool(checked["valid"]),
                "counterexample_prediction": checked["prediction"],
                "counterexample_topk_set": checked["topk_set"],
                "witness_path": witness_path, "witness_sha256": witness_hash,
                "attempts": attempts, "reused_parent": False,
                **({
                    "solver_backend": f0_backend,
                    "incremental_reuse_scope": (
                        "same_augmented_property_low_to_escalation"
                    ),
                    "cross_augmented_hz_reuse": False,
                } if incremental_session is not None else {}),
            })
            if checked["valid"]:
                break
        pair_status, pair_reason = _pair_reason(property_rows)
        pair_rows.append({
            "pair": list(pair), "status": pair_status, "reason": pair_reason,
            "expert_a_support": support_a, "expert_b_support": support_b,
            "expert_a_support_identity": identity_a,
            "expert_b_support_identity": identity_b,
            "property_rows": property_rows, "reused_parent": False,
        })
        if selected_witness is not None:
            break
    status, reason = _row_reason(pair_rows)
    return {
        "status": status, "reason": reason,
        "feasible_pairs": [list(pair) for pair in context["route_sets"].feasible],
        "pairs": pair_rows, "reused_property_rows": reused_properties,
        "rerun_property_rows": rerun_properties,
        "full_model_witness_valid": selected_witness is not None,
        "counterexample_prediction": selected_witness["prediction"] if selected_witness else None,
        "counterexample_topk_set": selected_witness["topk_set"] if selected_witness else None,
        "witness_path": selected_witness["path"] if selected_witness else None,
        "witness_sha256": selected_witness["sha256"] if selected_witness else None,
    }


def _run_gate(selection, context, model, work_dir, config, recorder):
    parent_gate = selection["parent"].get("gate") or {}
    parent_by_expert = {int(row["candidate"]): row for row in parent_gate.get("branches", [])}
    by_expert = {branch.expert: branch for branch in context["candidates"].branches}
    unresolved = [
        expert for expert in context["candidates"].candidates
        if parent_by_expert.get(expert, {}).get("branch_status") not in {"certified", "falsified"}
    ]
    branch_rows, full_witness = [], None
    support_config = _support_config(config)
    for position, expert in enumerate(context["candidates"].candidates):
        parent_branch = parent_by_expert.get(expert)
        if parent_branch is not None and parent_branch.get("branch_status") in {"certified", "falsified"}:
            branch_rows.append({**parent_branch, "reused_parent": True})
            continue
        branch = by_expert[expert]
        guarded = _propagate_component(
            context["program"].experts[expert], entry_hz=branch.guarded_input,
            hybridz_config=support_config,
        )
        support = _support_summary(guarded.guarded_support)
        if parent_branch is not None:
            actual = _gate_support_record(
                guarded, branch.guarded_input.n_bin
            )
            expected = {**parent_branch["support"], "relu_binaries": parent_branch["expert_relu_binaries_after_guard"], "binary_width": parent_branch["binary_width_after_guard"]}
            support_identity = _assert_support_identity(actual, expected)
        else:
            support_identity = None
        first_budget, second_budget = _attempt_budgets(
            recorder, len(unresolved), float(config["instance_timeout_seconds"])
        )
        attempts = []
        incremental_session = _new_incremental_property_session(
            guarded,
            time_limit=first_budget,
        )
        result = _solve_output(
            guarded, context["spec"], input_shape=tuple(context["x"].shape),
            time_limit=first_budget,
            incremental_session=incremental_session,
        )
        active = {"tier": "gate_elimination", "expert": expert, "attempt": 1}
        recorder.progress(active=active, solver=_gate_solver_snapshot(result))
        attempts.append({"budget": first_budget, **_gate_solver_snapshot(result)})
        if result.status in {VerifyStatus.UNKNOWN, VerifyStatus.TIMEOUT} and second_budget > 0.01:
            if incremental_session is not None:
                incremental_session.extend_budget(second_budget)
            result = _solve_output(
                guarded, context["spec"], input_shape=tuple(context["x"].shape),
                time_limit=second_budget,
                incremental_session=incremental_session,
            )
            active["attempt"] = 2
            recorder.progress(active=active, solver=_gate_solver_snapshot(result))
            attempts.append({"budget": second_budget, **_gate_solver_snapshot(result)})
        unresolved.pop(0)
        checked = _forward_validate(
            model, result.counterexample, lower=context["lower"], upper=context["upper"],
            clean_prediction=context["prediction"],
        )
        with torch.no_grad():
            clean_expert_prediction = int(model.experts[expert](context["x"]).argmax(dim=1).item())
        reason = _branch_reason(
            result, full_witness_valid=bool(checked["valid"]),
            clean_expert_prediction=clean_expert_prediction,
            clean_prediction=context["prediction"], support=support,
        )
        witness_path = witness_hash = None
        if checked["valid"]:
            witness_path, witness_hash = _save_witness(
                work_dir, context["rank"], "gate", str(expert), result.counterexample,
                {"expert": expert, "epsilon": context["epsilon"],
                 "clean_prediction": context["prediction"]},
            )
            full_witness = {**checked, "path": witness_path, "sha256": witness_hash}
        branch_rows.append({
            "candidate": expert, "branch_status": result.status.value,
            "unknown_reason": reason, "support": support,
            "full_model_witness_valid": bool(checked["valid"]),
            "counterexample_prediction": checked["prediction"],
            "counterexample_topk_set": checked["topk_set"],
            "attempts": attempts, "reused_parent": False,
            "support_identity": support_identity,
        })
    reasons = [row["unknown_reason"] for row in branch_rows]
    if full_witness is not None:
        status, reason = "UNSAFE", "UNSAFE_FULL_FORWARD"
    elif reasons and all(value == "SAFE_PROVED" for value in reasons):
        status, reason = "SAFE", "SAFE_PROVED"
    else:
        priority = (
            "UNKNOWN_GATE_SUFFICIENCY", "UNKNOWN_EXPERT_WITNESS_NOT_LIFTED",
            "TIMEOUT_SUPPORT", "TIMEOUT_EXPERT_SOLVE", "UNKNOWN_NUMERICAL",
            "UNKNOWN_SOLVER_LIMIT",
        )
        reason = next((value for value in priority if value in reasons), "UNKNOWN_SOLVER_LIMIT")
        status = "TIMEOUT" if reason.startswith("TIMEOUT_") else "UNKNOWN"
    return {
        "status": status, "reason": reason, "branches": branch_rows,
        "candidate_experts": list(context["candidates"].candidates),
        "feasible_route_sets": [list(pair) for pair in context["route_sets"].feasible],
        "full_model_witness_valid": full_witness is not None,
        "witness_path": full_witness["path"] if full_witness else None,
        "witness_sha256": full_witness["sha256"] if full_witness else None,
        "counterexample_prediction": full_witness["prediction"] if full_witness else None,
        "counterexample_topk_set": full_witness["topk_set"] if full_witness else None,
    }


def _run_row(selection, model, dataset, work_dir: Path, config) -> dict[str, Any]:
    started = time.monotonic()
    recorder = RowRecorder(work_dir, float(config["checkpoint_seconds"]))
    context = _row_context(selection, model, dataset, config)
    parent = selection["parent"]
    gate = f0 = None
    if selection["reuse_mode"] == "f0_unresolved_properties":
        f0 = _run_f0(selection, context, model, work_dir, config, recorder, parent_f0=parent["f0"])
        status, reason = f0["status"], f0["reason"]
    elif selection["reuse_mode"] == "fixed_radius_rebuild":
        f0 = _run_f0(selection, context, model, work_dir, config, recorder, parent_f0=None)
        status, reason = f0["status"], f0["reason"]
    else:
        gate = _run_gate(selection, context, model, work_dir, config, recorder)
        status, reason = gate["status"], gate["reason"]
        if reason in SEMANTIC_REASONS:
            f0 = _run_f0(selection, context, model, work_dir, config, recorder, parent_f0=None)
            status, reason = f0["status"], f0["reason"]
        elif status == "SAFE":
            reason = "SAFE_GATE_ELIMINATION"
    source = f0 if f0 is not None else gate
    recorder.finish_checkpoint(status)
    return {
        "sample_rank": context["rank"], "dataset_index": context["index"],
        "category": selection["category"], "reuse_mode": selection["reuse_mode"],
        "epsilon": context["epsilon"], "clean_prediction": context["prediction"],
        "clean_topk_set": context["clean_set"],
        "parent_status": parent["status"], "parent_reason": parent["reason"],
        "parent_line_number": selection["parent_line_number"],
        "parent_row_sha256": selection["parent_row_sha256"],
        "status": status, "reason": reason,
        "newly_solved": status in {"SAFE", "UNSAFE"},
        "full_model_witness_valid": bool(source and source["full_model_witness_valid"]),
        "witness_path": source.get("witness_path") if source else None,
        "witness_sha256": source.get("witness_sha256") if source else None,
        "counterexample_prediction": source.get("counterexample_prediction") if source else None,
        "counterexample_topk_set": source.get("counterexample_topk_set") if source else None,
        "reused_property_rows": f0.get("reused_property_rows", 0) if f0 else 0,
        "rerun_property_rows": f0.get("rerun_property_rows", 0) if f0 else 0,
        "checkpoint_recorded": recorder.checkpoint_recorded,
        "gate": gate, "f0": f0, "total_seconds": time.monotonic() - started,
    }


def _child(result_path, selection, work_dir, config):
    try:
        model, payload = load_output_moe_checkpoint(Path(config["checkpoint"]), map_location="cpu")
        model.double().eval()
        dataset = _load_dataset(payload["dataset"], False, download=False)
        row = _run_row(selection, model, dataset, work_dir, config)
    except Exception as exc:
        row = {
            "sample_rank": selection["sample_rank"], "dataset_index": selection["dataset_index"],
            "category": selection["category"], "reuse_mode": selection["reuse_mode"],
            "parent_status": selection["parent"]["status"],
            "parent_reason": selection["parent"]["reason"],
            "status": "ERROR", "reason": "EXPLICIT_NUMERICAL_OR_RUNNER_ERROR",
            "error": f"{type(exc).__name__}: {exc}", "total_seconds": 0.0,
        }
    _write_json(result_path, row)


def _run_with_deadline(selection, output_dir: Path, config) -> dict[str, Any]:
    rank = int(selection["sample_rank"])
    work_dir = output_dir / "row_work" / f"rank{rank}"
    work_dir.mkdir(parents=True, exist_ok=False)
    result_path = work_dir / "row.json"
    process = mp.get_context("spawn").Process(
        target=_child, args=(result_path, selection, work_dir, config), daemon=False
    )
    started = time.monotonic()
    process.start()
    timeout = float(config["instance_timeout_seconds"])
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate(); process.join()
        progress_path = work_dir / "progress.json"
        if progress_path.exists():
            with progress_path.open(encoding="utf-8") as handle:
                progress = json.load(handle)
        else:
            progress = {}
        return {
            "sample_rank": rank, "dataset_index": selection["dataset_index"],
            "category": selection["category"], "reuse_mode": selection["reuse_mode"],
            "epsilon": selection["epsilon"], "parent_status": selection["parent"]["status"],
            "parent_reason": selection["parent"]["reason"], "status": "TIMEOUT",
            "reason": "INSTANCE_HARD_DEADLINE", "newly_solved": False,
            "full_model_witness_valid": False, "checkpoint_recorded": (work_dir / "checkpoint_300.json").exists(),
            "active_at_deadline": progress.get("active_branch"),
            "solver_at_deadline": progress.get("solver"),
            "partial_work_dir": str(work_dir.relative_to(output_dir)),
            "deadline_seconds": timeout, "deadline_enforced": True,
            "total_seconds": time.monotonic() - started,
        }
    if process.exitcode != 0 or not result_path.exists():
        raise RuntimeError(f"rank {rank} child failed with exit code {process.exitcode}")
    with result_path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    if row.get("witness_path"):
        row["witness_path"] = str(
            Path("row_work") / f"rank{rank}" / row["witness_path"]
        )
    row["deadline_seconds"] = timeout
    row["deadline_enforced"] = True
    row["total_seconds"] = time.monotonic() - started
    return row


def _summary(rows: Sequence[dict[str, Any]], config) -> dict[str, Any]:
    statuses = Counter(row["status"] for row in rows)
    reasons = Counter(row["reason"] for row in rows)
    solved = statuses["SAFE"] + statuses["UNSAFE"]
    applicable_coverage = (int(config["preregistered_unlock"]["parent_solved_applicable_rows"]) + solved) / int(config["preregistered_unlock"]["parent_applicable_rows"])
    return {
        "rows": len(rows), "status_counts": dict(statuses), "reason_counts": dict(reasons),
        "newly_solved_applicable_rows": solved,
        "parent_overall_solved_rate_immutable": 0.56,
        "parent_boundary_applicability": "76/100",
        "parent_conditional_coverage": "56/76",
        "closure_conditional_coverage": applicable_coverage,
        "new_unsafe_rows": statuses["UNSAFE"],
        "new_unsafe_full_forward_validated": sum(row["status"] == "UNSAFE" and row.get("full_model_witness_valid") for row in rows),
        "baseline_unlock_pre_audit": (
            len(rows) == int(config["expected_rows"])
            and solved >= int(config["preregistered_unlock"]["minimum_newly_solved_applicable_rows"])
            and applicable_coverage >= float(config["preregistered_unlock"]["minimum_applicable_coverage"])
            and not any(row["status"] == "ERROR" or "NUMERICAL" in row["reason"] for row in rows)
        ),
        "total_seconds": sum(float(row["total_seconds"]) for row in rows),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("Experiment 1D requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("Experiment 1D requires a clean worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("Experiment 1D requires act-py312")
    if Path(get_torchvision_data_root()).resolve().is_relative_to(WRITE_ROOT.resolve()) is False:
        raise RuntimeError("TorchVision data root escapes /data1/Kane/MOE")
    if config["numerical_safety"] != hz_numerical_policy_manifest():
        raise RuntimeError("frozen numerical policy differs from implementation")
    selected = _load_frozen_selection(config)
    if output_dir.exists():
        raise RuntimeError(f"Experiment 1D refuses to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    runtime = {
        "source_config": str(config_path), "source_config_sha256": _sha256(config_path),
        "selection_manifest_sha256": _sha256(Path(config["selection_manifest"])),
        "git_head": _git_value("rev-parse", "HEAD"), "checkpoint_sha256": _sha256(Path(config["checkpoint"])),
        "config": config,
    }
    _write_json(output_dir / "config.json", runtime)
    _write_json(output_dir / "selection.json", {
        "parent_results_sha256": config["parent_results_sha256"],
        "rows": [{key: value for key, value in row.items() if key != "parent"} for row in selected],
    })
    rows = []
    with (
        (output_dir / "results.jsonl").open("x", encoding="utf-8") as jsonl,
        (output_dir / "results.csv").open("x", newline="", encoding="utf-8") as csv_handle,
        (output_dir / "experiment1d.log").open("x", encoding="utf-8") as log,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS); writer.writeheader()
        for position, selection in enumerate(selected, 1):
            row = _run_with_deadline(selection, output_dir, config)
            rows.append(row); _append_json(jsonl, row)
            writer.writerow({key: row.get(key) for key in CSV_FIELDS}); csv_handle.flush(); os.fsync(csv_handle.fileno())
            log.write(f"ROW {position}/20 rank={row['sample_rank']} status={row['status']} reason={row['reason']} seconds={row['total_seconds']:.3f}\n"); log.flush(); os.fsync(log.fileno())
    summary = _summary(rows, config); _write_json(output_dir / "summary.json", summary)
    return {"output_dir": str(output_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(run(Path(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
