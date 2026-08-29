# ===- crown_adapter_cohort.py - Four-way expert adapter cohort ------====#
"""Four-configuration adapter consistency cohort on 43 frozen route pairs.

The cohort uses the verification-scale bal010 model and the exact feasible
unordered top-2 pairs already frozen by the guarded-box-hull benchmark.  For
each pair and each member expert it compares:

* exact HZ propagation with the pair guard retained;
* CROWN on the guarded coordinate box hull;
* CROWN on the original perturbation box; and
* CROWN on a tie-safe eta implication over the original box.

Every negative relaxation bound remains ``UNKNOWN``.  This runner never emits
``UNSAFE`` and never promotes a CROWN candidate without a full-model replay.
The eta compiler uses ``g_S=max(outside)-min(inside)`` for a complete unordered
top-k set, so every pair legal under ``ANY_LEGAL_TOPK`` has ``g_S<=0``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import typing
from typing import Any, Sequence

from typing_extensions import override

# The pinned CROWN environment is Python 3.11 while ACT uses typing.override.
if not hasattr(typing, "override"):
    typing.override = override  # type: ignore[attr-defined]

import numpy as np
import torch
from torch import nn

from act.back_end.moe import (
    build_act_moe_program,
    guarded_input_topk_set,
    linear_safety_rows,
    load_output_moe_checkpoint,
)
from act.front_end.specs import OutKind, OutputSpec
from act.back_end.moe.tie_safe_implication import (
    relu_pairwise_max,
    relu_pairwise_min,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_numerical_policy_manifest,
    hz_support_bounds,
    sparse_hz_linear,
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


DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/crown_adapter_cohort_bal010_43.json"
)
CROWN_PYTHON = Path("/data1/Kane/MOE/envs/alpha-beta-crown/bin/python")
RESULTS_ROOT = PROJECT_ROOT / "data/moe/results"
VARIANTS = (
    "hz_retained_guard",
    "crown_guarded_box",
    "crown_original_box",
    "crown_tie_safe_eta",
)


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_status(
    lower: Sequence[float],
    *,
    complete: bool,
    exact: bool,
    tolerance: float,
) -> str:
    values = np.asarray(lower, dtype=np.float64)
    if not complete or values.size == 0 or not np.all(np.isfinite(values)):
        return "UNKNOWN_INCOMPLETE"
    if exact and bool(np.all(values >= float(tolerance))):
        return "CERTIFIED"
    if exact:
        return "NOT_CERTIFIED_COMPLETE"
    if bool(np.all(values >= float(tolerance))):
        return "CERTIFIED_MARGIN_FILTER"
    return "UNKNOWN_RELAXATION"


def _combine_experts(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [record["status"] for record in records]
    if records and all(status in {"CERTIFIED", "CERTIFIED_MARGIN_FILTER"} for status in statuses):
        status = "CERTIFIED"
    elif any(status == "ERROR" for status in statuses):
        status = "ERROR"
    elif all(status == "NOT_CERTIFIED_COMPLETE" for status in statuses):
        status = "NOT_CERTIFIED_COMPLETE"
    else:
        status = "UNKNOWN"
    lower_values = [
        float(record["minimum_lower_bound"])
        for record in records
        if record.get("minimum_lower_bound") is not None
    ]
    return {
        "status": status,
        "expert_statuses": statuses,
        "minimum_lower_bound": min(lower_values, default=None),
        "complete": bool(records) and all(bool(record["complete"]) for record in records),
        "seconds": sum(float(record["seconds"]) for record in records),
    }


class TieSafeTopKSetImplication(nn.Module):
    """Compile one complete unordered top-k route-set implication.

    A route set ``S`` is legal (including every tie) exactly when
    ``max_{j notin S} r_j - min_{i in S} r_i <= 0``.  The compiled scalar is
    ``max(g_S-eta, min_property_row)``.  Its only extra obligations occur for
    non-member points with ``0 < g_S < eta``.
    """

    def __init__(
        self,
        router_logits: nn.Module,
        expert: nn.Module,
        route_set: Sequence[int],
        num_experts: int,
        property_matrix: torch.Tensor | Sequence[Sequence[float]],
        property_offset: torch.Tensor | Sequence[float],
        *,
        eta: float,
    ) -> None:
        super().__init__()
        selected = tuple(sorted({int(value) for value in route_set}))
        num_experts = int(num_experts)
        eta = float(eta)
        if len(selected) < 1:
            raise ValueError("route_set must be non-empty")
        if num_experts <= len(selected):
            raise ValueError("num_experts must include an outside expert")
        if selected[0] < 0 or selected[-1] >= num_experts:
            raise IndexError("route_set is outside router output")
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and strictly positive")
        matrix = torch.as_tensor(property_matrix)
        offset = torch.as_tensor(property_offset)
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError("property_matrix must be non-empty and two-dimensional")
        if offset.ndim != 1 or offset.shape[0] != matrix.shape[0]:
            raise ValueError("property_offset must match property rows")
        try:
            module_dtype = next(router_logits.parameters()).dtype
        except StopIteration:
            module_dtype = torch.get_default_dtype()
        matrix = matrix.to(dtype=module_dtype)
        offset = offset.to(dtype=module_dtype)
        self.router_logits = router_logits
        self.expert = expert
        self.route_set = selected
        self.num_experts = num_experts
        self.eta = eta
        inside_selector = torch.zeros(len(selected), num_experts, dtype=matrix.dtype)
        inside_selector[torch.arange(len(selected)), torch.tensor(selected)] = 1.0
        outside_indices = [
            index for index in range(num_experts) if index not in set(selected)
        ]
        outside_selector = torch.zeros(
            len(outside_indices), num_experts, dtype=matrix.dtype
        )
        outside_selector[
            torch.arange(len(outside_indices)), torch.tensor(outside_indices)
        ] = 1.0
        self.inside_projector = nn.Linear(
            num_experts, len(selected), bias=False, dtype=matrix.dtype
        )
        self.outside_projector = nn.Linear(
            num_experts, len(outside_indices), bias=False, dtype=matrix.dtype
        )
        self.property_projector = nn.Linear(
            int(matrix.shape[1]), int(matrix.shape[0]), bias=True, dtype=matrix.dtype
        )
        with torch.no_grad():
            self.inside_projector.weight.copy_(inside_selector)
            self.outside_projector.weight.copy_(outside_selector)
            self.property_projector.weight.copy_(matrix)
            self.property_projector.bias.copy_(offset.to(dtype=matrix.dtype))

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.router_logits(x)
        output = self.expert(x)
        if scores.ndim != 2 or scores.shape[1] != self.num_experts:
            raise ValueError("router score width differs from num_experts")
        inside = self.inside_projector(scores)
        outside = self.outside_projector(scores)
        guard = relu_pairwise_max(outside) - relu_pairwise_min(inside)
        rows = self.property_projector(output)
        safety = relu_pairwise_min(rows)
        compiled = guard - self.eta + torch.relu(safety - (guard - self.eta))
        return guard, safety, compiled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)[2].unsqueeze(1)


def _load_source_branches(config: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = _inside(Path(config["source_branches_jsonl"]), RESULTS_ROOT)
    if _sha256(source_path) != config["source_branches_sha256"]:
        raise RuntimeError("frozen 43-branch source changed")
    raw_lines = source_path.read_bytes().splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(raw_lines, 1):
        row = json.loads(raw)
        branch_id = str(row["branch_id"])
        if branch_id in seen:
            raise RuntimeError(f"duplicate frozen branch {branch_id}")
        seen.add(branch_id)
        if not row.get("paired_complete") or not row.get("within_tolerance"):
            raise RuntimeError(f"source branch {branch_id} is incomplete or disagreed")
        artifact = _inside(source_path.parent / row["bounds_artifact"], source_path.parent)
        if _sha256(artifact) != row["bounds_artifact_sha256"]:
            raise RuntimeError(f"source bound artifact changed for {branch_id}")
        rows.append(
            {
                **row,
                "source_line_number": line_number,
                "source_line_sha256": _sha256_bytes(raw),
                "bounds_artifact_absolute": str(artifact),
            }
        )
    if len(rows) != int(config["expected_branches"]):
        raise RuntimeError("frozen branch count changed")
    return rows


def _load_guarded_box(source: dict[str, Any], shape: tuple[int, ...]):
    with np.load(source["bounds_artifact_absolute"]) as artifact:
        lower = np.asarray(artifact["highspy_lower"], dtype=np.float64)
        upper = np.asarray(artifact["highspy_upper"], dtype=np.float64)
    if lower.size != int(np.prod(shape)) or upper.size != int(np.prod(shape)):
        raise RuntimeError("guarded box artifact shape changed")
    return (
        torch.from_numpy(lower.copy()).reshape(shape),
        torch.from_numpy(upper.copy()).reshape(shape),
    )


def _parent_exact_pairs(
    selection: dict[str, Any]
) -> set[tuple[int, ...]] | None:
    """Read the already-audited exact route sets without solving them again."""

    parent = selection["parent"]
    f0_parent = parent.get("f0") or {}
    gate_parent = parent.get("gate") or {}
    raw_pairs = (
        f0_parent.get("feasible_pairs")
        or gate_parent.get("feasible_route_sets")
        or parent.get("exact_feasible_pairs")
    )
    if not raw_pairs:
        return None
    return {
        tuple(sorted(int(value) for value in pair))
        for pair in raw_pairs
    }


def _validate_frozen_pairs(
    selection: dict[str, Any],
    source_branches: Sequence[dict[str, Any]],
    *,
    source_exact_audit_passed: bool = False,
) -> set[tuple[int, ...]]:
    """Validate source identities against the frozen exact parent artifact.

    The 43 guarded-box branches were produced from exact, independently audited
    route sets. Re-running their feasibility queries here adds no semantic
    evidence and can incorrectly censor a valid branch when the repeated query
    reaches its time limit.
    """

    rank = int(selection["sample_rank"])
    source_pairs: set[tuple[int, ...]] = set()
    for branch in source_branches:
        if int(branch["sample_rank"]) != rank:
            raise RuntimeError("frozen source branch rank changed")
        pair = tuple(sorted(int(value) for value in branch["route_pair"]))
        if len(pair) != 2 or len(set(pair)) != 2:
            raise RuntimeError("frozen source route pair is not unordered top-2")
        source_pairs.add(pair)
    if not source_pairs:
        raise RuntimeError(f"rank {rank} has no frozen source branches")
    parent_pairs = _parent_exact_pairs(selection)
    if parent_pairs is None and not source_exact_audit_passed:
        raise RuntimeError(
            "frozen parent has no pair list and source exactness audit is absent"
        )
    if parent_pairs is not None and source_pairs != parent_pairs:
        raise RuntimeError("frozen source pairs differ from frozen exact parent")
    return source_pairs


def _frozen_row_context(selection, model, dataset) -> dict[str, Any]:
    """Rebuild the model/HZ frame while treating frozen pairs as authoritative."""

    parent = selection["parent"]
    rank = int(selection["sample_rank"])
    index = int(selection["dataset_index"])
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
    if parent.get("clean_prediction") is not None and prediction != int(
        parent["clean_prediction"]
    ):
        raise RuntimeError("clean prediction changed from frozen parent")
    if parent.get("clean_topk_set") is not None and clean_set != sorted(
        int(value) for value in parent["clean_topk_set"]
    ):
        raise RuntimeError("clean top-k set changed from frozen parent")
    spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[prediction])
    program = build_act_moe_program(
        model, center=x, lower=lower, upper=upper, output_spec=spec
    )
    router = _propagate_component(program.router)
    if not isinstance(router.output_hz, SparseHZono) or not router.output_hz.exact:
        raise RuntimeError("adapter cohort requires an exact router HZ")
    return {
        "rank": rank,
        "index": index,
        "label": int(label),
        "epsilon": epsilon,
        "x": x,
        "lower": lower,
        "upper": upper,
        "prediction": prediction,
        "clean_set": clean_set,
        "spec": spec,
        "program": program,
        "router": router,
        "properties": linear_safety_rows(spec, int(clean_output.shape[1])),
        "route_pair_authority": "FROZEN_EXACT_INDEPENDENTLY_AUDITED",
    }


def _hz_expert_record(
    expert_net,
    guarded: SparseHZono,
    property_rows: tuple[tuple[np.ndarray, float], ...],
    *,
    time_limit: float,
    tolerance: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        propagation = _propagate_component(expert_net, entry_hz=guarded)
        matrix = np.stack([row for row, _constant in property_rows], axis=0)
        offset = np.asarray([constant for _row, constant in property_rows])
        property_hz = sparse_hz_linear(propagation.output_hz, matrix, offset)
        support = hz_support_bounds(
            property_hz,
            range(len(property_rows)),
            time_limit=float(time_limit),
            relax_binaries=False,
        )
        lower = support.bounds.lb.detach().cpu().double().numpy().reshape(-1)
        exact = bool(property_hz.exact and support.exact)
        status = _safe_status(
            lower, complete=support.exact, exact=exact, tolerance=tolerance
        )
        return {
            "status": status,
            "complete": bool(support.exact),
            "exact": exact,
            "minimum_lower_bound": float(lower.min()) if lower.size else None,
            "lower_bounds": lower.tolist(),
            "lower_status": list(support.lower_status),
            "upper_status": list(support.upper_status),
            "solver_gaps": list(support.solver_gap),
            "solves": int(support.solves),
            "property_rows": len(property_rows),
            "binary_width": int(propagation.binary_width),
            "propagation_seconds": float(propagation.elapsed),
            "support_seconds": float(support.elapsed),
            "seconds": time.monotonic() - started,
            "negative_bound_semantics": "NOT_CERTIFIED_NEVER_UNSAFE",
            "full_model_witness_valid": None,
            "error": None,
        }
    except Exception as error:
        return {
            "status": "ERROR",
            "complete": False,
            "exact": False,
            "minimum_lower_bound": None,
            "lower_bounds": [],
            "seconds": time.monotonic() - started,
            "negative_bound_semantics": "NOT_CERTIFIED_NEVER_UNSAFE",
            "full_model_witness_valid": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _crown_bounds(
    module: nn.Module,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    property_rows: tuple[tuple[np.ndarray, float], ...] | None,
    device: str,
    tolerance: float,
    method: str,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        dtype = torch.float32
        module = module.to(device=device, dtype=dtype).eval()
        center = center.to(device=device, dtype=dtype)
        lower = lower.to(device=device, dtype=dtype)
        upper = upper.to(device=device, dtype=dtype)
        build_started = time.monotonic()
        bounded = BoundedModule(module, center, device=device)
        build_seconds = time.monotonic() - build_started
        bounded_input = BoundedTensor(
            center,
            PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper),
        )
        C = None
        offsets = None
        if property_rows is not None:
            C = torch.as_tensor(
                np.stack([row for row, _constant in property_rows]),
                device=device,
                dtype=dtype,
            ).unsqueeze(0)
            offsets = torch.as_tensor(
                [constant for _row, constant in property_rows],
                device=device,
                dtype=dtype,
            )
        solve_started = time.monotonic()
        bound_lower, bound_upper = bounded.compute_bounds(
            x=(bounded_input,), C=C, method=method
        )
        solve_seconds = time.monotonic() - solve_started
        lower_values = bound_lower.reshape(-1)
        upper_values = bound_upper.reshape(-1)
        if offsets is not None:
            lower_values = lower_values + offsets
            upper_values = upper_values + offsets
        lower_np = lower_values.detach().cpu().double().numpy()
        upper_np = upper_values.detach().cpu().double().numpy()
        complete = bool(
            lower_np.size > 0
            and np.all(np.isfinite(lower_np))
            and np.all(np.isfinite(upper_np))
        )
        status = _safe_status(
            lower_np, complete=complete, exact=False, tolerance=tolerance
        )
        return {
            "status": status,
            "complete": complete,
            "exact": False,
            "minimum_lower_bound": float(lower_np.min()) if lower_np.size else None,
            "lower_bounds": lower_np.tolist(),
            "upper_bounds": upper_np.tolist(),
            "property_rows": int(lower_np.size),
            "graph_build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "seconds": time.monotonic() - started,
            "method": method,
            "dtype": str(dtype),
            "device": device,
            "acceptance_rule": f"finite CROWN lower bound >= {tolerance}",
            "numerical_soundness_status": (
                "POSITIVE_MARGIN_FILTER_NOT_OUTWARD_ROUNDED"
            ),
            "negative_bound_semantics": "UNKNOWN_NEVER_UNSAFE",
            "full_model_witness_valid": None,
            "error": None,
        }
    except Exception as error:
        return {
            "status": "ERROR",
            "complete": False,
            "exact": False,
            "minimum_lower_bound": None,
            "lower_bounds": [],
            "upper_bounds": [],
            "seconds": time.monotonic() - started,
            "method": method,
            "device": device,
            "negative_bound_semantics": "UNKNOWN_NEVER_UNSAFE",
            "full_model_witness_valid": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _soundness_issues(experts: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for record in experts:
        hz = record["hz_retained_guard"]
        if (
            hz["status"] != "NOT_CERTIFIED_COMPLETE"
            or hz.get("minimum_lower_bound") is None
            or float(hz["minimum_lower_bound"]) >= -1e-7
        ):
            continue
        for variant in VARIANTS[1:]:
            if record[variant]["status"] == "CERTIFIED_MARGIN_FILTER":
                issues.append(
                    f"expert{record['expert']}:{variant}_certified_but_exact_hz_refuted"
                )
    return issues


def _run_branch(
    selection: dict[str, Any],
    context: dict[str, Any],
    source: dict[str, Any],
    model,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    pair = tuple(sorted(int(value) for value in source["route_pair"]))
    guarded = guarded_input_topk_set(
        context["router"].input_hz, context["router"].output_hz, pair
    ).hz
    if not isinstance(guarded, SparseHZono) or not guarded.exact:
        raise RuntimeError("cohort requires an exact pair-guarded input HZ")
    guarded_lower, guarded_upper = _load_guarded_box(
        source, tuple(context["x"].shape)
    )
    numerical = config["numerical_safety"]
    containment_tolerance = (
        float(numerical["outward_absolute"])
        + float(numerical["outward_relative"])
        * max(
            1.0,
            float(context["lower"].abs().max()),
            float(context["upper"].abs().max()),
        )
        + np.finfo(np.float64).eps
    )
    if bool(
        (guarded_lower < context["lower"] - containment_tolerance).any()
    ) or bool(
        (guarded_upper > context["upper"] + containment_tolerance).any()
    ):
        raise RuntimeError("guarded hull is not contained in the original box")
    tolerance = float(config["safe_positive_margin"])
    experts: list[dict[str, Any]] = []
    for expert_index in pair:
        expert = model.experts[expert_index]
        hz = _hz_expert_record(
            context["program"].experts[expert_index],
            guarded,
            context["properties"],
            time_limit=float(config["hz_property_time_limit_seconds"]),
            tolerance=tolerance,
        )
        guarded_box = _crown_bounds(
            copy.deepcopy(expert),
            (guarded_lower + guarded_upper) / 2.0,
            guarded_lower,
            guarded_upper,
            property_rows=context["properties"],
            device=config["crown_device"],
            tolerance=tolerance,
            method=config["crown_method"],
        )
        original_box = _crown_bounds(
            copy.deepcopy(expert),
            context["x"],
            context["lower"],
            context["upper"],
            property_rows=context["properties"],
            device=config["crown_device"],
            tolerance=tolerance,
            method=config["crown_method"],
        )
        property_matrix = np.stack(
            [row for row, _constant in context["properties"]], axis=0
        )
        property_offset = np.asarray(
            [constant for _row, constant in context["properties"]]
        )
        implication = TieSafeTopKSetImplication(
            copy.deepcopy(model.router),
            copy.deepcopy(expert),
            pair,
            model.spec.num_experts,
            property_matrix,
            property_offset,
            eta=tolerance,
        )
        tie_safe = _crown_bounds(
            implication,
            context["x"],
            context["lower"],
            context["upper"],
            property_rows=None,
            device=config["crown_device"],
            tolerance=tolerance,
            method=config["crown_method"],
        )
        experts.append(
            {
                "expert": int(expert_index),
                "hz_retained_guard": hz,
                "crown_guarded_box": guarded_box,
                "crown_original_box": original_box,
                "crown_tie_safe_eta": tie_safe,
            }
        )
    variants = {
        name: _combine_experts([record[name] for record in experts])
        for name in VARIANTS
    }
    issues = _soundness_issues(experts)
    return {
        "branch_id": source["branch_id"],
        "sample_rank": int(selection["sample_rank"]),
        "dataset_index": int(selection["dataset_index"]),
        "epsilon": float(selection["epsilon"]),
        "route_pair": list(pair),
        "clean_prediction": int(context["prediction"]),
        "clean_topk_set": list(context["clean_set"]),
        "source": {
            "line_number": source["source_line_number"],
            "line_sha256": source["source_line_sha256"],
            "bounds_artifact": source["bounds_artifact"],
            "bounds_artifact_sha256": source["bounds_artifact_sha256"],
            "outward_containment_tolerance": containment_tolerance,
        },
        "route_semantics": {
            "policy": "ANY_LEGAL_TOPK",
            "unordered_top_k": 2,
            "tie_inclusive": True,
            "pair_guard": "max_outside_score - min_inside_score <= 0",
            "eta": tolerance,
            "eta_extra_obligation_band": "0 < g_S < eta",
        },
        "property_semantics": (
            "both member experts satisfy every clean-prediction margin row; "
            "selected-softmax convexity then proves the weighted pair"
        ),
        "experts": experts,
        "variants": variants,
        "soundness_issues": issues,
        "unsafe_claimed": False,
        "relaxation_candidate_replay_required_for_unsafe": True,
        "seconds": time.monotonic() - started,
        "error": None,
    }


def _summary(rows: list[dict[str, Any]], expected_branches: int) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    transitions: dict[str, dict[str, int]] = {}
    for variant in VARIANTS[1:]:
        counter = Counter(
            f"{row['variants']['hz_retained_guard']['status']}->{row['variants'][variant]['status']}"
            for row in valid
        )
        transitions[variant] = dict(sorted(counter.items()))
    issues = [
        f"{row['branch_id']}:{issue}"
        for row in valid
        for issue in row.get("soundness_issues", [])
    ]
    error_ids = [row.get("branch_id") for row in rows if row.get("error")]
    variant_counts = {
        variant: dict(
            sorted(Counter(row["variants"][variant]["status"] for row in valid).items())
        )
        for variant in VARIANTS
    }
    runtimes = {
        variant: {
            "total_seconds": sum(
                float(row["variants"][variant]["seconds"]) for row in valid
            ),
            "median_branch_seconds": statistics.median(
                [float(row["variants"][variant]["seconds"]) for row in valid]
            )
            if valid
            else None,
        }
        for variant in VARIANTS
    }
    no_unsafe = all(not row.get("unsafe_claimed") for row in rows)
    return {
        "scope": "adapter_consistency_engineering_not_main_certificate",
        "expected_branches": int(expected_branches),
        "completed_branches": len(rows),
        "valid_branches": len(valid),
        "error_branch_ids": error_ids,
        "variant_status_counts": variant_counts,
        "hz_reference_transitions": transitions,
        "variant_runtime": runtimes,
        "soundness_issue_count": len(issues),
        "soundness_issues": issues,
        "no_unsafe_statuses_emitted": no_unsafe,
        "all_source_branches_consumed": len(rows) == int(expected_branches),
        "audit_passed": (
            len(rows) == int(expected_branches)
            and not error_ids
            and not issues
            and no_unsafe
        ),
        "safe_positive_margin": 1e-7,
        "tie_policy": "ANY_LEGAL_TOPK",
        "numerical_caveat": (
            "CROWN rows use a positive-margin acceptance filter but are not "
            "outward-rounded; this cohort is adapter consistency evidence"
        ),
    }


def independently_summarize(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Re-read flushed JSONL and source hashes without using in-memory rows."""

    result_path = _inside(output_dir / "branches.jsonl", RESULTS_ROOT)
    raw_lines = result_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(raw) for raw in raw_lines]
    source = {row["branch_id"]: row for row in _load_source_branches(config)}
    audit_issues: list[str] = []
    seen: set[str] = set()
    for row in rows:
        branch_id = row.get("branch_id")
        if branch_id in seen:
            audit_issues.append(f"duplicate_result:{branch_id}")
        seen.add(branch_id)
        parent = source.get(branch_id)
        if parent is None:
            audit_issues.append(f"unknown_source:{branch_id}")
            continue
        if row.get("source", {}).get("line_sha256") != parent["source_line_sha256"]:
            audit_issues.append(f"source_hash_mismatch:{branch_id}")
        if row.get("route_semantics", {}).get("policy") != "ANY_LEGAL_TOPK":
            audit_issues.append(f"tie_policy_mismatch:{branch_id}")
        if float(row.get("route_semantics", {}).get("eta", -1.0)) != float(
            config["safe_positive_margin"]
        ):
            audit_issues.append(f"eta_mismatch:{branch_id}")
        if set(row.get("variants", {})) != set(VARIANTS):
            audit_issues.append(f"variant_set_mismatch:{branch_id}")
        if row.get("unsafe_claimed"):
            audit_issues.append(f"unreplayed_unsafe:{branch_id}")
    missing = sorted(set(source) - seen)
    audit_issues.extend(f"missing_result:{branch_id}" for branch_id in missing)
    summary = _summary(rows, int(config["expected_branches"]))
    return {
        "schema_version": 1,
        "result_jsonl_sha256": _sha256(result_path),
        "result_line_sha256": [_sha256_bytes(raw) for raw in raw_lines],
        "recomputed_summary": summary,
        "structural_audit_issues": audit_issues,
        "issue_count": len(audit_issues) + int(summary["soundness_issue_count"]),
        "passed": not audit_issues and bool(summary["audit_passed"]),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), RESULTS_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("CROWN adapter cohort requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("CROWN adapter cohort requires a clean worktree")
    if Path(sys.executable).resolve() != CROWN_PYTHON.resolve():
        raise RuntimeError("CROWN adapter cohort requires alpha-beta-crown Python")
    data_root = Path(get_torchvision_data_root()).resolve()
    if not data_root.is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision data root escapes /data1/Kane/MOE")
    if config["numerical_safety"] != hz_numerical_policy_manifest():
        raise RuntimeError("frozen HZ numerical policy changed")
    if float(config["safe_positive_margin"]) != 1e-7:
        raise RuntimeError("tie-safe eta must remain safe_positive_margin=1e-7")
    checkpoint = _inside(Path(config["checkpoint"]), PROJECT_ROOT)
    if _sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("frozen bal010 checkpoint changed")
    source = _load_source_branches(config)
    source_summary = _inside(Path(config["source_summary_json"]), RESULTS_ROOT)
    if _sha256(source_summary) != config["source_summary_sha256"]:
        raise RuntimeError("frozen 43-branch summary changed")
    source_audit_path = _inside(
        Path(config["source_independent_audit_json"]), RESULTS_ROOT
    )
    if _sha256(source_audit_path) != config["source_independent_audit_sha256"]:
        raise RuntimeError("frozen 43-branch independent audit changed")
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    source_exact_audit_passed = bool(
        int(source_audit.get("issue_count", -1)) == 0
        and source_audit.get("all_exact_feasible_pairs_covered") is True
        and source_audit.get("all_paired_complete") is True
        and int(source_audit.get("route_branches", -1))
        == int(config["expected_branches"])
    )
    if not source_exact_audit_passed:
        raise RuntimeError("frozen branch source did not pass its exactness audit")
    selection_manifest = _inside(Path(config["selection_manifest"]), PROJECT_ROOT)
    if _sha256(selection_manifest) != config["selection_manifest_sha256"]:
        raise RuntimeError("frozen Experiment 1D selection changed")
    selected = _load_frozen_selection(config)
    source_by_rank: dict[int, list[dict[str, Any]]] = {}
    for branch in source:
        source_by_rank.setdefault(int(branch["sample_rank"]), []).append(branch)
    if output_dir.exists():
        raise RuntimeError(f"CROWN adapter cohort refuses to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    _write_json(
        output_dir / "config.json",
        {
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "git_head": _git_value("rev-parse", "HEAD"),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_branches_sha256": config["source_branches_sha256"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "auto_lirpa": importlib.metadata.version("auto_LiRPA"),
            "config": config,
        },
    )

    rows: list[dict[str, Any]] = []
    with (
        (output_dir / "branches.jsonl").open("x", encoding="utf-8") as handle,
        (output_dir / "cohort.log").open("x", encoding="utf-8") as log,
    ):
        for selection in selected:
            rank = int(selection["sample_rank"])
            frozen = source_by_rank.get(rank, [])
            try:
                _validate_frozen_pairs(
                    selection,
                    frozen,
                    source_exact_audit_passed=source_exact_audit_passed,
                )
                context = _frozen_row_context(selection, model, dataset)
            except Exception as error:
                context = None
                row_error = f"{type(error).__name__}: {error}"
            else:
                row_error = None
            for source_branch in source_by_rank.get(rank, []):
                if row_error is None:
                    try:
                        row = _run_branch(
                            selection, context, source_branch, model, config
                        )
                    except Exception as error:
                        row_error_branch = f"{type(error).__name__}: {error}"
                    else:
                        row_error_branch = None
                else:
                    row_error_branch = row_error
                if row_error_branch is None:
                    rows.append(row)
                    _append_json(handle, row)
                    log.write(
                        f"BRANCH {row['branch_id']} variants="
                        f"{json.dumps(row['variants'], sort_keys=True)}\n"
                    )
                    log.flush()
                    os.fsync(log.fileno())
                    if config["crown_device"] == "cuda":
                        torch.cuda.empty_cache()
                else:
                    row = {
                        "branch_id": source_branch["branch_id"],
                        "sample_rank": rank,
                        "dataset_index": int(selection["dataset_index"]),
                        "route_pair": source_branch["route_pair"],
                        "source": {
                            "line_sha256": source_branch["source_line_sha256"]
                        },
                        "unsafe_claimed": False,
                        "error": row_error_branch,
                    }
                    rows.append(row)
                    _append_json(handle, row)
                    log.write(
                        f"BRANCH_ERROR {source_branch['branch_id']} "
                        f"{row_error_branch}\n"
                    )
                    log.flush()
                    os.fsync(log.fileno())

    summary = _summary(rows, int(config["expected_branches"]))
    _write_json(output_dir / "summary.json", summary)
    independent = independently_summarize(output_dir, config)
    _write_json(output_dir / "independent_summary.json", independent)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "independent_summary": independent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
