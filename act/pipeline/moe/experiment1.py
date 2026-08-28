# ===- act/pipeline/moe/experiment1.py - Route-A Experiment 1 ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Incremental candidate-set and binary-width study for output-level MoEs.

The runner is intentionally narrow: it accepts ACT's controlled CIFAR-10
checkpoint, preserves unordered/tie-inclusive top-k semantics, and writes one
CSV row after every verification instance.  It does not label a per-expert
failure as a weighted-MoE counterexample unless the concrete full model is
also forward-validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from act.back_end.analyze import analyze
from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import HybridzTF
from act.back_end.layer_schema import LayerKind
from act.back_end.moe import (
    analyze_candidates,
    analyze_topk_sets,
    build_act_moe_program,
    load_output_moe_checkpoint,
)
from act.back_end.moe.route_a import _component_output_hz
from act.back_end.moe.routing import interval_candidate_mask
from act.back_end.solver.solver_hz import HZSolver, SparseHZono
from act.back_end.transfer_functions import (
    get_solver_mode,
    get_transfer_function,
    set_solver_mode,
    set_transfer_function,
)
from act.back_end.verifier import (
    add_all_input_specs,
    find_entry_layer_id,
    gather_input_spec_layers,
    get_input_ids,
    seed_from_input_specs,
)
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.route_flips import _route_margin
from act.pipeline.moe.train import _device, _load_dataset
from act.util.path_config import get_torchvision_data_root
from act.util.stats import VerifyResult, VerifyStatus


PROJECT_ROOT = Path("/data1/Kane/MOE/ACT")
WRITE_ROOT = Path("/data1/Kane/MOE")
DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/experiment1_bal010.json"

RESULT_FIELDS = (
    "phase",
    "sample_rank",
    "dataset_index",
    "label",
    "clean_prediction",
    "clean_topk_set",
    "epsilon_label",
    "epsilon",
    "boundary_route_radius_upper_bound",
    "boundary_radius_kind",
    "ibp_candidate_experts",
    "ibp_candidate_count",
    "zonotope_candidate_experts",
    "zonotope_candidate_count",
    "exact_hz_candidate_experts",
    "exact_hz_candidate_count",
    "exact_hz_candidate_minimal",
    "exact_feasible_topk_sets",
    "exact_feasible_topk_set_count",
    "exact_topk_sets_complete",
    "route_set_unstable",
    "unguarded_unstable_relu_per_expert",
    "guarded_unstable_relu_per_expert",
    "router_binary_count",
    "monolithic_route_binary_count",
    "monolithic_binary_width",
    "candidate_pruned_monolithic_binary_width",
    "route_conditioned_binary_widths",
    "route_conditioned_max_binary_width",
    "route_conditioned_mean_binary_width",
    "route_invariance_status",
    "monolithic_hz_status",
    "monolithic_hz_implementation",
    "monolithic_expert_statuses",
    "gate_elimination_status",
    "route_conditioned_expert_statuses",
    "overall_status",
    "verified_route",
    "forward_validated_counterexample",
    "counterexample_prediction",
    "counterexample_topk_set",
    "ibp_candidate_seconds",
    "zonotope_candidate_seconds",
    "exact_candidate_seconds",
    "exact_topk_set_seconds",
    "candidate_seconds",
    "tightening_seconds",
    "monolithic_solve_seconds",
    "route_conditioned_solve_seconds",
    "solve_seconds",
    "total_seconds",
    "obbt",
    "error",
)


@dataclass
class Propagation:
    output_hz: Any
    input_hz: Any
    output_bounds: Bounds
    unstable_per_relu: tuple[int, ...]
    guarded_support: tuple[dict[str, object], ...]
    elapsed: float

    @property
    def unstable_total(self) -> int:
        return sum(self.unstable_per_relu)

    @property
    def binary_width(self) -> int:
        return int(getattr(self.output_hz, "n_bin", 0))


def _inside(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes allowed root {root}: {resolved}")
    return resolved


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), *args], text=True
    ).strip()


def _file_manifest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _candidate_tuple(lower: torch.Tensor, upper: torch.Tensor, top_k: int) -> tuple[int, ...]:
    bounds = Bounds(lower.reshape(1, -1), upper.reshape(1, -1))
    mask = interval_candidate_mask(bounds, top_k)[0]
    return tuple(int(index) for index in mask.nonzero(as_tuple=False).flatten())


def _ibp_router_bounds(
    router: nn.Module, lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lb = lower.reshape(-1).double()
    ub = upper.reshape(-1).double()
    for layer in router.children():
        if isinstance(layer, nn.Flatten):
            lb, ub = lb.reshape(-1), ub.reshape(-1)
        elif isinstance(layer, nn.Linear):
            weight = layer.weight.detach().cpu().double()
            bias = layer.bias.detach().cpu().double()
            positive, negative = weight.clamp(min=0), weight.clamp(max=0)
            lb, ub = positive @ lb + negative @ ub + bias, positive @ ub + negative @ lb + bias
        elif isinstance(layer, nn.ReLU):
            lb, ub = lb.clamp(min=0), ub.clamp(min=0)
        else:
            raise NotImplementedError(f"IBP router does not support {type(layer).__name__}")
    return lb, ub


def _zonotope_router_bounds(
    router: nn.Module, lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary unconstrained DeepZ propagation for the controlled MLP router."""
    center = ((lower + upper) / 2).reshape(-1).double()
    input_radius = ((upper - lower) / 2).reshape(-1).double()
    generators: torch.Tensor | None = None
    for layer in router.children():
        if isinstance(layer, nn.Flatten):
            center = center.reshape(-1)
        elif isinstance(layer, nn.Linear):
            weight = layer.weight.detach().cpu().double()
            bias = layer.bias.detach().cpu().double()
            center = weight @ center + bias
            generators = (
                weight * input_radius.unsqueeze(0)
                if generators is None
                else weight @ generators
            )
        elif isinstance(layer, nn.ReLU):
            if generators is None:
                raise RuntimeError("ReLU encountered before a zonotope generator matrix")
            radius = generators.abs().sum(dim=1)
            lb, ub = center - radius, center + radius
            active, inactive = lb >= 0, ub <= 0
            unstable = ~(active | inactive)
            slope = torch.zeros_like(center)
            slope[active] = 1.0
            slope[unstable] = ub[unstable] / (ub[unstable] - lb[unstable])
            offset = torch.zeros_like(center)
            offset[unstable] = (
                -ub[unstable] * lb[unstable] / (2 * (ub[unstable] - lb[unstable]))
            )
            center = slope * center + offset
            generators = slope.unsqueeze(1) * generators
            slots = unstable.nonzero(as_tuple=False).flatten()
            if slots.numel():
                relaxation = torch.zeros(
                    (center.numel(), slots.numel()),
                    dtype=center.dtype,
                    device=center.device,
                )
                relaxation[
                    slots,
                    torch.arange(slots.numel(), device=center.device),
                ] = offset[slots]
                generators = torch.cat((generators, relaxation), dim=1)
        else:
            raise NotImplementedError(
                f"ordinary zonotope router does not support {type(layer).__name__}"
            )
    if generators is None:
        radius = input_radius
    else:
        radius = generators.abs().sum(dim=1)
    return center - radius, center + radius


def _component_input_hz(net, tf: HybridzTF, output_hz):
    specs = gather_input_spec_layers(net)
    if isinstance(output_hz, SparseHZono):
        for layer in reversed(specs):
            candidate = tf.get_sparse_hz(layer.id)
            if candidate is not None and candidate.frame_id == output_hz.frame_id:
                return candidate
    else:
        for layer in reversed(specs):
            candidate = tf.get_hz(layer.id)
            if candidate is not None:
                return candidate
    raise RuntimeError("component HZ has no correlated input state")


def _propagate_component(net, *, entry_hz=None, hybridz_config=None) -> Propagation:
    previous_solver = get_solver_mode()
    try:
        previous_tf = get_transfer_function()
    except RuntimeError:
        previous_tf = None
    tf = HybridzTF(
        config=hybridz_config or HybridZConfig(max_input_dim=1024)
    )
    set_transfer_function(tf)
    set_solver_mode("hybridz")
    if entry_hz is not None:
        tf.set_entry_hz(entry_hz)
    try:
        specs = gather_input_spec_layers(net)
        seed = seed_from_input_specs(specs)
        entry = Fact(bounds=seed, cons=ConSet())
        add_all_input_specs(entry.cons, get_input_ids(net), specs)
        started = time.monotonic()
        _before, after, _global = analyze(net, find_entry_layer_id(net), entry)
        elapsed = time.monotonic() - started
        output_hz = _component_output_hz(net, tf)
        input_hz = _component_input_hz(net, tf, output_hz)
        assertion = next(layer for layer in net.layers if layer.kind == LayerKind.ASSERT.value)
        output_ids = net.preds.get(assertion.id, [])
        if len(output_ids) != 1:
            raise ValueError("component ASSERT must have exactly one predecessor")
        output_bounds = after[output_ids[0]].bounds
        unstable: list[int] = []
        for layer in net.layers:
            if layer.kind != LayerKind.RELU.value:
                continue
            predecessors = net.preds.get(layer.id, [])
            if len(predecessors) != 1:
                raise ValueError("ReLU must have exactly one predecessor")
            bounds = after[predecessors[0]].bounds
            unstable.append(int(((bounds.lb < 0) & (bounds.ub > 0)).sum().item()))
        return Propagation(
            output_hz,
            input_hz,
            output_bounds,
            tuple(unstable),
            tf.guarded_support_stats(),
            elapsed,
        )
    finally:
        tf.clear_entry_hz()
        set_solver_mode(previous_solver)
        if previous_tf is not None:
            set_transfer_function(previous_tf)


def _solve_output(
    propagation: Propagation,
    output_spec: OutputSpec,
    *,
    input_shape: tuple[int, ...],
    time_limit: float,
) -> VerifyResult:
    if time_limit <= 0:
        return VerifyResult(VerifyStatus.TIMEOUT, metadata={"reason": "instance_deadline"})
    solver = HZSolver(time_limit=time_limit)
    result = solver.evaluate_spec(
        propagation.output_hz,
        output_spec,
        batch_size=1,
        n_out=int(propagation.output_hz.n_out),
        input_hz=propagation.input_hz,
        input_shape=input_shape,
        timelimit=time_limit,
    )[0]
    elapsed = float(result.metadata.get("elapsed", 0.0))
    if result.status == VerifyStatus.UNKNOWN and elapsed >= 0.98 * time_limit:
        result.status = VerifyStatus.TIMEOUT
        result.metadata["reason"] = "solver_deadline"
    return result


def _paper_status(status: VerifyStatus) -> str:
    return {
        VerifyStatus.CERTIFIED: "SAFE",
        VerifyStatus.FALSIFIED: "UNSAFE",
        VerifyStatus.TIMEOUT: "TIMEOUT",
    }.get(status, "UNKNOWN")


def _weighted_expert_status(status: VerifyStatus) -> str:
    """A single selected-softmax expert witness is never a full-model UNSAFE."""
    if status == VerifyStatus.FALSIFIED:
        return "UNKNOWN"
    return _paper_status(status)


def _forward_validate(
    model,
    candidate: torch.Tensor | None,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    clean_prediction: int,
) -> dict[str, Any]:
    empty = {
        "valid": False,
        "prediction": None,
        "topk_set": None,
    }
    if candidate is None:
        return empty
    value = candidate.unsqueeze(0).to(dtype=next(model.parameters()).dtype)
    if value.shape != lower.shape:
        return empty
    tolerance = 1e-7
    if bool((value < lower - tolerance).any()) or bool((value > upper + tolerance).any()):
        return empty
    with torch.no_grad():
        output, route = model.forward_with_routing(value)
    prediction = int(output.argmax(dim=1).item())
    topk_set = sorted(int(v) for v in route.indices[0].tolist())
    return {
        "valid": prediction != clean_prediction,
        "prediction": prediction,
        "topk_set": topk_set,
    }


def _aggregate_expert_results(
    results: dict[int, VerifyResult],
    *,
    model,
    lower: torch.Tensor,
    upper: torch.Tensor,
    clean_prediction: int,
) -> tuple[str, dict[str, Any]]:
    witness = {"valid": False, "prediction": None, "topk_set": None}
    for result in results.values():
        if result.status != VerifyStatus.FALSIFIED:
            continue
        checked = _forward_validate(
            model,
            result.counterexample,
            lower=lower,
            upper=upper,
            clean_prediction=clean_prediction,
        )
        if checked["valid"]:
            return "UNSAFE", checked
        if checked["prediction"] is not None:
            witness = checked
    statuses = [result.status for result in results.values()]
    if statuses and all(status == VerifyStatus.CERTIFIED for status in statuses):
        return "SAFE", witness
    if any(status == VerifyStatus.TIMEOUT for status in statuses):
        return "TIMEOUT", witness
    return "UNKNOWN", witness


def _route_change_upper_bound(
    model,
    clean: torch.Tensor,
    original_set: Sequence[int],
    *,
    max_epsilon: float,
    steps: int,
    bisection_steps: int,
    seed: int,
) -> dict[str, Any] | None:
    """Return a concrete, forward-validated upper bound on route-change radius."""
    original = torch.tensor(original_set, device=clean.device, dtype=torch.long)[None]
    original_key = tuple(sorted(int(v) for v in original_set))

    def attack(epsilon: float, restart_seed: int):
        torch.manual_seed(restart_seed)
        if clean.is_cuda:
            torch.cuda.manual_seed_all(restart_seed)
        lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
        adversarial = lower + torch.rand_like(clean) * (upper - lower)
        step_size = epsilon / 4.0
        for _ in range(steps):
            adversarial.requires_grad_(True)
            scores = model.route(adversarial).scores
            loss = _route_margin(scores, original).mean()
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = adversarial.detach() + step_size * gradient.sign()
            adversarial = torch.maximum(torch.minimum(adversarial, upper), lower)
        with torch.no_grad():
            route = model.route(adversarial).indices[0]
        route_key = tuple(sorted(int(v) for v in route.tolist()))
        if route_key == original_key:
            return None
        distance = float((adversarial - clean).abs().max().item())
        return adversarial.detach(), route_key, distance

    grid = [value / 255.0 for value in (0.25, 0.5, 1.0, 2.0, 4.0)]
    grid = [value for value in grid if value <= max_epsilon + 1e-12]
    if not grid or grid[-1] < max_epsilon:
        grid.append(max_epsilon)
    lower_search = 0.0
    best = None
    high = None
    for slot, epsilon in enumerate(grid):
        for restart in range(2):
            found = attack(epsilon, seed + slot * 17 + restart)
            if found is not None:
                best, high = found, epsilon
                break
        if best is not None:
            break
        lower_search = epsilon
    if best is None or high is None:
        return None
    for slot in range(bisection_steps):
        middle = (lower_search + high) / 2.0
        found = attack(middle, seed + 1000 + slot)
        if found is None:
            lower_search = middle
        else:
            best, high = found, middle
    adversarial, route_key, distance = best
    with torch.no_grad():
        validated = tuple(sorted(int(v) for v in model.route(adversarial).indices[0]))
    if validated == original_key:
        raise RuntimeError("boundary attack witness failed route-set forward validation")
    return {
        "radius_upper_bound": distance,
        "route_set": list(route_key),
    }


def _select_clean_correct(model, dataset, device: torch.device, count: int) -> list[int]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    selected: list[int] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs).argmax(dim=1)
            for local in (predictions == labels).nonzero(as_tuple=False).flatten().tolist():
                selected.append(offset + int(local))
                if len(selected) == count:
                    return selected
            offset += int(labels.numel())
    raise RuntimeError(f"dataset contains only {len(selected)} clean-correct samples")


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return sum(data) / len(data) if data else None


def _run_instance(
    *,
    model,
    x: torch.Tensor,
    label: int,
    clean_prediction: int,
    clean_route: Sequence[int],
    sample_rank: int,
    dataset_index: int,
    epsilon: float,
    epsilon_label: str,
    phase: str,
    timeout: float,
    query_timeout: float,
    boundary_upper: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + timeout
    lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[clean_prediction])
    row: dict[str, Any] = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "phase": phase,
            "sample_rank": sample_rank,
            "dataset_index": dataset_index,
            "label": label,
            "clean_prediction": clean_prediction,
            "clean_topk_set": _json(sorted(int(v) for v in clean_route)),
            "epsilon_label": epsilon_label,
            "epsilon": epsilon,
            "boundary_route_radius_upper_bound": boundary_upper or "",
            "boundary_radius_kind": (
                "forward_validated_upper_bound" if boundary_upper is not None else ""
            ),
            "obbt": False,
        }
    )

    candidate_started = time.monotonic()
    stage = time.monotonic()
    ibp_lb, ibp_ub = _ibp_router_bounds(model.router, lower, upper)
    ibp_candidates = _candidate_tuple(ibp_lb, ibp_ub, model.spec.top_k)
    ibp_seconds = time.monotonic() - stage
    stage = time.monotonic()
    zono_lb, zono_ub = _zonotope_router_bounds(model.router, lower, upper)
    zono_candidates = _candidate_tuple(zono_lb, zono_ub, model.spec.top_k)
    zono_seconds = time.monotonic() - stage

    program = build_act_moe_program(
        model,
        center=x,
        lower=lower,
        upper=upper,
        output_spec=output_spec,
    )
    stage = time.monotonic()
    router = _propagate_component(program.router)
    if not isinstance(router.output_hz, SparseHZono) or not router.output_hz.exact:
        raise RuntimeError("exact-router-HZ label requires an exact sparse HZ")
    remaining = max(deadline - time.monotonic(), 0.01)
    exact_candidates = analyze_candidates(
        router.output_hz,
        model.spec.top_k,
        input_hz=router.input_hz,
        time_limit_per_expert=min(query_timeout, remaining / model.spec.num_experts),
        router_exact=True,
    )
    exact_candidate_seconds = time.monotonic() - stage
    stage = time.monotonic()
    route_set_count = math.comb(model.spec.num_experts, model.spec.top_k)
    remaining = max(deadline - time.monotonic(), 0.01)
    exact_sets = analyze_topk_sets(
        router.output_hz,
        model.spec.top_k,
        time_limit_per_set=min(query_timeout, remaining / route_set_count),
        router_exact=True,
    )
    exact_set_seconds = time.monotonic() - stage
    candidate_seconds = time.monotonic() - candidate_started

    row.update(
        {
            "ibp_candidate_experts": _json(ibp_candidates),
            "ibp_candidate_count": len(ibp_candidates),
            "zonotope_candidate_experts": _json(zono_candidates),
            "zonotope_candidate_count": len(zono_candidates),
            "exact_hz_candidate_experts": _json(exact_candidates.candidates),
            "exact_hz_candidate_count": len(exact_candidates.candidates),
            "exact_hz_candidate_minimal": exact_candidates.minimal,
            "exact_feasible_topk_sets": _json(exact_sets.feasible),
            "exact_feasible_topk_set_count": len(exact_sets.feasible),
            "exact_topk_sets_complete": exact_sets.exact,
            "route_set_unstable": (
                len(exact_sets.feasible) > 1 if exact_sets.exact else ""
            ),
            "router_binary_count": router.binary_width,
            "ibp_candidate_seconds": ibp_seconds,
            "zonotope_candidate_seconds": zono_seconds,
            "exact_candidate_seconds": exact_candidate_seconds,
            "exact_topk_set_seconds": exact_set_seconds,
            "candidate_seconds": candidate_seconds,
        }
    )

    tightening_started = time.monotonic()
    unguarded: dict[int, Propagation] = {}
    for expert, net in enumerate(program.experts):
        unguarded[expert] = _propagate_component(net)
    guarded: dict[int, Propagation] = {}
    by_expert = {branch.expert: branch for branch in exact_candidates.branches}
    for expert in exact_candidates.candidates:
        branch = by_expert[expert]
        if branch.guarded_input is not None:
            guarded[expert] = _propagate_component(
                program.experts[expert], entry_hz=branch.guarded_input
            )
    tightening_seconds = time.monotonic() - tightening_started
    unguarded_counts = {expert: value.unstable_total for expert, value in unguarded.items()}
    guarded_counts = {expert: value.unstable_total for expert, value in guarded.items()}
    branch_widths = {expert: value.binary_width for expert, value in guarded.items()}
    route_unstable = exact_sets.exact and len(exact_sets.feasible) > 1
    monolithic_route_binaries = model.spec.num_experts if route_unstable else 0
    monolithic_width = (
        router.binary_width + monolithic_route_binaries + sum(unguarded_counts.values())
    )
    candidate_pruned_width = (
        router.binary_width
        + monolithic_route_binaries
        + sum(unguarded_counts[index] for index in exact_candidates.candidates)
    )
    row.update(
        {
            "unguarded_unstable_relu_per_expert": _json(unguarded_counts),
            "guarded_unstable_relu_per_expert": _json(guarded_counts),
            "monolithic_route_binary_count": monolithic_route_binaries,
            "monolithic_binary_width": monolithic_width,
            "candidate_pruned_monolithic_binary_width": candidate_pruned_width,
            "route_conditioned_binary_widths": _json(branch_widths),
            "route_conditioned_max_binary_width": max(branch_widths.values(), default=0),
            "route_conditioned_mean_binary_width": _mean(branch_widths.values()) or 0.0,
            "route_invariance_status": (
                "INVARIANT"
                if exact_sets.exact
                and exact_sets.feasible == (tuple(sorted(int(v) for v in clean_route)),)
                else "UNKNOWN"
            ),
            "tightening_seconds": tightening_seconds,
        }
    )

    # This baseline uses the same exact HZ support and gate-elimination property
    # but no route guards. Solves are decomposed to avoid actually allocating the
    # structural monolithic width; runtime is therefore not a monolithic runtime.
    monolithic_started = time.monotonic()
    monolithic_results: dict[int, VerifyResult] = {}
    for position, expert in enumerate(exact_candidates.candidates):
        remaining = deadline - time.monotonic()
        slots = max(len(exact_candidates.candidates) - position, 1)
        monolithic_results[expert] = _solve_output(
            unguarded[expert],
            output_spec,
            input_shape=tuple(x.shape),
            time_limit=min(query_timeout, max(remaining / slots, 0.0)),
        )
    monolithic_status, monolithic_witness = _aggregate_expert_results(
        monolithic_results,
        model=model,
        lower=lower,
        upper=upper,
        clean_prediction=clean_prediction,
    )
    monolithic_seconds = time.monotonic() - monolithic_started

    guarded_started = time.monotonic()
    guarded_results: dict[int, VerifyResult] = {}
    for position, expert in enumerate(exact_candidates.candidates):
        remaining = deadline - time.monotonic()
        slots = max(len(exact_candidates.candidates) - position, 1)
        if expert not in guarded:
            guarded_results[expert] = VerifyResult(
                VerifyStatus.UNKNOWN, metadata={"reason": "missing_guarded_input"}
            )
            continue
        guarded_results[expert] = _solve_output(
            guarded[expert],
            output_spec,
            input_shape=tuple(x.shape),
            time_limit=min(query_timeout, max(remaining / slots, 0.0)),
        )
    gate_status, guarded_witness = _aggregate_expert_results(
        guarded_results,
        model=model,
        lower=lower,
        upper=upper,
        clean_prediction=clean_prediction,
    )
    guarded_seconds = time.monotonic() - guarded_started
    chosen_witness = guarded_witness if guarded_witness["valid"] else monolithic_witness
    overall = gate_status
    if time.monotonic() > deadline and overall not in {"SAFE", "UNSAFE"}:
        overall = "TIMEOUT"
    row.update(
        {
            "monolithic_hz_status": monolithic_status,
            "monolithic_hz_implementation": "route-unguarded exact-HZ gate-elimination; decomposed solve, structural width only",
            "monolithic_expert_statuses": _json(
                {
                    expert: _weighted_expert_status(result.status)
                    for expert, result in monolithic_results.items()
                }
            ),
            "gate_elimination_status": gate_status,
            "route_conditioned_expert_statuses": _json(
                {
                    expert: _weighted_expert_status(result.status)
                    for expert, result in guarded_results.items()
                }
            ),
            "overall_status": overall,
            "verified_route": _json(chosen_witness["topk_set"]) if chosen_witness["valid"] else "",
            "forward_validated_counterexample": bool(chosen_witness["valid"]),
            "counterexample_prediction": chosen_witness["prediction"] if chosen_witness["valid"] else "",
            "counterexample_topk_set": _json(chosen_witness["topk_set"]) if chosen_witness["valid"] else "",
            "monolithic_solve_seconds": monolithic_seconds,
            "route_conditioned_solve_seconds": guarded_seconds,
            "solve_seconds": monolithic_seconds + guarded_seconds,
            "total_seconds": time.monotonic() - started,
        }
    )
    return row


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fixed = [row for row in rows if row.get("phase") == "fixed"]
    exact_complete = [row for row in fixed if row.get("exact_topk_sets_complete") is True]
    route_unstable = [row for row in exact_complete if row.get("route_set_unstable") is True]
    reductions_ibp = [
        row for row in route_unstable
        if int(row["exact_hz_candidate_count"]) < int(row["ibp_candidate_count"])
    ]
    reductions_zono = [
        row for row in route_unstable
        if int(row["exact_hz_candidate_count"]) < int(row["zonotope_candidate_count"])
    ]
    guard_reductions: list[float] = []
    width_ratios: list[float] = []
    unique_safe = 0
    for row in fixed:
        unguarded = json.loads(row["unguarded_unstable_relu_per_expert"] or "{}")
        guarded = json.loads(row["guarded_unstable_relu_per_expert"] or "{}")
        for expert, guarded_count in guarded.items():
            original = float(unguarded[str(expert)])
            if original > 0:
                guard_reductions.append((original - float(guarded_count)) / original)
        mono = float(row["candidate_pruned_monolithic_binary_width"] or 0)
        branch = float(row["route_conditioned_max_binary_width"] or 0)
        if mono > 0:
            width_ratios.append(branch / mono)
        if (
            row.get("route_set_unstable") is True
            and row.get("route_invariance_status") == "UNKNOWN"
            and row.get("overall_status") == "SAFE"
        ):
            unique_safe += 1
    statuses: dict[str, int] = {}
    for row in rows:
        key = str(row.get("overall_status", "ERROR"))
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "rows": len(rows),
        "fixed_rows": len(fixed),
        "boundary_rows": len(rows) - len(fixed),
        "route_unstable_fixed_rows": len(route_unstable),
        "exact_hz_strictly_reduces_ibp": len(reductions_ibp),
        "exact_hz_strictly_reduces_zonotope": len(reductions_zono),
        "exact_hz_reduces_ibp_rate_among_route_unstable": len(reductions_ibp) / max(len(route_unstable), 1),
        "exact_hz_reduces_zonotope_rate_among_route_unstable": len(reductions_zono) / max(len(route_unstable), 1),
        "guard_unstable_relu_reduction_median": statistics.median(guard_reductions) if guard_reductions else None,
        "route_conditioned_to_candidate_monolithic_width_median": statistics.median(width_ratios) if width_ratios else None,
        "unique_safe_certificates": unique_safe,
        "unique_safe_rate_among_route_unstable": unique_safe / max(len(route_unstable), 1),
        "overall_status_counts": statuses,
        "go_no_go": {
            "candidate_claim": len(reductions_zono) / max(len(route_unstable), 1) >= 0.20,
            "guard_tightening_claim": bool(guard_reductions) and statistics.median(guard_reductions) >= 0.10,
            "unique_certificate_claim": unique_safe / max(len(route_unstable), 1) >= 0.10,
        },
    }


def run(args) -> dict[str, Any]:
    config_path = _inside(Path(args.config), PROJECT_ROOT)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    output_dir = _inside(Path(args.output_dir or config["output_dir"]), WRITE_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / ("smoke.log" if args.smoke else "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
        force=True,
    )
    logger = logging.getLogger("experiment1")

    branch = _git_value("branch", "--show-current")
    status = _git_value("status", "--short", "--branch")
    if branch != "feat/moe-route-verification":
        raise RuntimeError(f"refusing to run on branch {branch!r}")
    dirty_lines = [line for line in status.splitlines()[1:] if line.strip()]
    allowed_untracked = (
        "?? act/pipeline/moe/configs/",
        "?? act/pipeline/moe/experiment1.py",
    )
    unexpected = [
        line for line in dirty_lines
        if not line.startswith(" M act/back_end/moe/")
        and not any(line.startswith(prefix) for prefix in allowed_untracked)
    ]
    if unexpected:
        raise RuntimeError(f"unexpected worktree changes: {unexpected}")

    configured_root = Path(config["dataset_root"]).resolve()
    actual_root = Path(get_torchvision_data_root()).resolve()
    if actual_root != configured_root or not actual_root.is_relative_to(WRITE_ROOT):
        raise RuntimeError(f"torchvision root mismatch: {actual_root} != {configured_root}")
    device = _device(config["device"])
    if device.type != "cuda":
        raise RuntimeError("Experiment 1 configuration requires CUDA")

    attack_model, payload = load_output_moe_checkpoint(checkpoint, map_location=device)
    attack_model.to(device).eval()
    model, _ = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    target_count = int(config["sample_selection"]["count"])
    indices = _select_clean_correct(attack_model, dataset, device, target_count)
    sample_limit = min(args.max_samples or target_count, target_count)
    selected = indices[:sample_limit]

    import scipy
    import torchvision

    dataset_files = actual_root / "CIFAR10/raw/cifar-10-batches-py"
    runtime = {
        "source_config": str(config_path),
        "git_branch": branch,
        "git_head": _git_value("rev-parse", "HEAD"),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_config": payload.get("training_config"),
        "checkpoint_validation_metrics": payload.get("validation_metrics"),
        "checkpoint_test_metrics": payload.get("test_metrics"),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "scipy_version": scipy.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "dataset_file_sha256": _file_manifest(dataset_files),
        "selected_indices_file": str(output_dir / "sample_indices.json"),
        "sample_limit": sample_limit,
        "smoke": bool(args.smoke),
        "obbt": False,
        "monolithic_baseline_note": "Exact-HZ route-unguarded gate-elimination solves are decomposed; reported monolithic runtime is not claimed, while structural binary width instantiates all experts.",
        "config": config,
    }
    prefix = "smoke_" if args.smoke else ""
    _write_json(output_dir / f"{prefix}config.json", runtime)
    _write_json(
        output_dir / "sample_indices.json",
        {"selection_rule": config["sample_selection"]["rule"], "indices": indices},
    )

    csv_path = output_dir / f"{prefix}results.csv"
    rows: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        for rank, index in enumerate(selected):
            image, label = dataset[index]
            attack_dtype = next(attack_model.parameters()).dtype
            attack_x = image.unsqueeze(0).to(device=device, dtype=attack_dtype)
            with torch.no_grad():
                clean_output, clean_decision = attack_model.forward_with_routing(attack_x)
            clean_prediction = int(clean_output.argmax(dim=1).item())
            clean_route = sorted(int(v) for v in clean_decision.indices[0].tolist())
            if clean_prediction != int(label):
                raise RuntimeError("saved clean-correct index no longer predicts correctly")
            x = image.unsqueeze(0).double()
            for epsilon_entry in config["epsilons"]:
                try:
                    row = _run_instance(
                        model=model,
                        x=x,
                        label=int(label),
                        clean_prediction=clean_prediction,
                        clean_route=clean_route,
                        sample_rank=rank,
                        dataset_index=index,
                        epsilon=float(epsilon_entry["value"]),
                        epsilon_label=epsilon_entry["label"],
                        phase="fixed",
                        timeout=float(config["timeout_per_instance_seconds"]),
                        query_timeout=float(config["candidate_timeout_per_query_seconds"]),
                    )
                except Exception as exc:
                    logger.exception("instance failed index=%s epsilon=%s", index, epsilon_entry["label"])
                    row = {field: "" for field in RESULT_FIELDS}
                    row.update(
                        phase="fixed",
                        sample_rank=rank,
                        dataset_index=index,
                        label=int(label),
                        clean_prediction=clean_prediction,
                        clean_topk_set=_json(clean_route),
                        epsilon_label=epsilon_entry["label"],
                        epsilon=epsilon_entry["value"],
                        overall_status="UNKNOWN",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
                logger.info(
                    "fixed index=%s epsilon=%s candidates=%s status=%s seconds=%.3f",
                    index,
                    epsilon_entry["label"],
                    row.get("exact_hz_candidate_count"),
                    row.get("overall_status"),
                    float(row.get("total_seconds") or 0),
                )

            if config["boundary_adaptive"]["enabled"] and not args.skip_boundary:
                boundary = _route_change_upper_bound(
                    attack_model,
                    attack_x,
                    clean_route,
                    max_epsilon=float(config["boundary_adaptive"]["search_max_epsilon"]),
                    steps=int(config["boundary_adaptive"]["attack_steps"]),
                    bisection_steps=int(config["boundary_adaptive"]["bisection_steps"]),
                    seed=10000 + index,
                )
                if boundary is not None:
                    upper_bound = float(boundary["radius_upper_bound"])
                    epsilon = min(
                        float(config["boundary_adaptive"]["search_max_epsilon"]),
                        float(config["boundary_adaptive"]["multiplier"]) * upper_bound,
                    )
                    try:
                        row = _run_instance(
                            model=model,
                            x=x,
                            label=int(label),
                            clean_prediction=clean_prediction,
                            clean_route=clean_route,
                            sample_rank=rank,
                            dataset_index=index,
                            epsilon=epsilon,
                            epsilon_label="1.05x_route_change_upper_bound",
                            phase="boundary",
                            timeout=float(config["timeout_per_instance_seconds"]),
                            query_timeout=float(config["candidate_timeout_per_query_seconds"]),
                            boundary_upper=upper_bound,
                        )
                    except Exception as exc:
                        logger.exception("boundary instance failed index=%s", index)
                        row = {field: "" for field in RESULT_FIELDS}
                        row.update(
                            phase="boundary",
                            sample_rank=rank,
                            dataset_index=index,
                            label=int(label),
                            clean_prediction=clean_prediction,
                            clean_topk_set=_json(clean_route),
                            epsilon_label="1.05x_route_change_upper_bound",
                            epsilon=epsilon,
                            boundary_route_radius_upper_bound=upper_bound,
                            boundary_radius_kind="forward_validated_upper_bound",
                            overall_status="UNKNOWN",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                    logger.info(
                        "boundary index=%s upper=%.9f epsilon=%.9f status=%s seconds=%.3f",
                        index,
                        upper_bound,
                        epsilon,
                        row.get("overall_status"),
                        float(row.get("total_seconds") or 0),
                    )
                else:
                    logger.info("boundary index=%s no route-change witness found", index)

    summary = _summarize(rows)
    summary.update(
        {
            "csv": str(csv_path),
            "sample_indices": str(output_dir / "sample_indices.json"),
            "partial_rows_flushed": True,
        }
    )
    _write_json(output_dir / f"{prefix}summary.json", summary)
    logger.info("summary=%s", _json(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MoE Route-A Experiment 1")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
