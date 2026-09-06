# ===- act/pipeline/moe/experiment1_confirmatory.py - Confirmatory ---====#

"""Frozen ranks 100--199 census and route-boundary certification runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from act.back_end.moe import (
    analyze_candidates,
    analyze_topk_sets,
    build_act_moe_program,
    load_output_moe_checkpoint,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_numerical_policy_manifest,
)
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.accounting import guard_binary_accounting
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _candidate_tuple,
    _forward_validate,
    _git_value,
    _ibp_router_bounds,
    _inside,
    _propagate_component,
    _select_clean_correct,
    _sha256,
    _write_json,
    _zonotope_router_bounds,
)
from act.pipeline.moe.experiment1c import (
    _router_route_change,
    _support_summary,
    diagnose_radius,
    exact_route_change_bracket,
)
from act.pipeline.moe.experiment1f0 import _run_parent_row
from act.pipeline.moe.train import _device, _load_dataset
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/experiment1_confirmatory_bal010_r1.json"
)
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
SEMANTIC_REASONS = {
    "UNKNOWN_GATE_SUFFICIENCY",
    "UNKNOWN_EXPERT_WITNESS_NOT_LIFTED",
}
CENSUS_FIELDS = (
    "sample_rank",
    "dataset_index",
    "epsilon_label",
    "epsilon",
    "status",
    "route_set_unstable",
    "ibp_candidate_count",
    "zonotope_candidate_count",
    "exact_candidate_count",
    "exact_feasible_pair_count",
    "structural_monolithic_width",
    "candidate_pruned_monolithic_width",
    "route_conditioned_max_width",
    "route_conditioned_mean_width",
    "candidate_seconds",
    "propagation_seconds",
    "total_seconds",
    "ibp_candidates",
    "zonotope_candidates",
    "exact_candidates",
    "exact_feasible_pairs",
    "branches",
    "error",
)
BOUNDARY_FIELDS = (
    "sample_rank",
    "dataset_index",
    "status",
    "reason",
    "epsilon",
    "route_lower",
    "route_upper",
    "bracket_width",
    "bisection_complete",
    "exact_feasible_pair_count",
    "route_invariance_status",
    "unique_safe",
    "gate_status",
    "gate_reason",
    "f0_invoked",
    "f0_status",
    "f0_reason",
    "f0_time_observation",
    "active_stage_at_deadline",
    "full_model_witness_valid",
    "witness_path",
    "witness_sha256",
    "candidate_seconds",
    "gate_seconds",
    "f0_seconds",
    "total_seconds",
    "bracket",
    "gate",
    "f0",
    "error",
)


def _append_json(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _log(handle, message: str) -> None:
    handle.write(message.rstrip() + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _csv_row(row: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    result = {field: row.get(field) for field in fields}
    for key, value in list(result.items()):
        if isinstance(value, (dict, list, tuple)):
            result[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return result


def _support_config(config: dict[str, Any]) -> HybridZConfig:
    support = config["support"]
    return HybridZConfig(
        max_input_dim=1024,
        guarded_support_enabled=True,
        guarded_support_lp_neurons=int(support["lp_neurons"]),
        guarded_support_milp_neurons=int(support["milp_neurons"]),
        guarded_support_lp_time_limit=float(support["lp_time_limit"]),
        guarded_support_milp_time_limit=float(support["milp_time_limit"]),
    )


def _prepare(config_path: Path, config: dict[str, Any]):
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    selection_manifest = (
        _inside(Path(config["selection_manifest"]), PROJECT_ROOT)
        if config.get("selection_manifest")
        else None
    )
    development_indices = (
        _inside(Path(config["development_sample_indices"]), WRITE_ROOT)
        if selection_manifest is None
        else None
    )
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("confirmatory experiment requires the feature branch")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("confirmatory experiment requires a clean worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("confirmatory experiment requires act-py312")
    actual_policy = hz_numerical_policy_manifest()
    if actual_policy != config["numerical_safety"]:
        raise RuntimeError("tracked numerical SAFE policy differs from implementation")
    if (
        float(config["f0"]["solver"]["safety_tolerance"])
        != actual_policy["safe_positive_margin"]
    ):
        raise RuntimeError("F0 SAFE tolerance differs from frozen policy")
    dataset_root = Path(get_torchvision_data_root()).resolve()
    if not dataset_root.is_relative_to(WRITE_ROOT.resolve()):
        raise RuntimeError("TorchVision data root escapes /data1/Kane/MOE")

    runtime_path = output_dir / "config.json"
    indices_path = output_dir / "sample_indices.json"
    if runtime_path.exists() != indices_path.exists():
        raise RuntimeError("confirmatory root has incomplete provenance files")
    verification_model, payload = load_output_moe_checkpoint(
        checkpoint, map_location="cpu"
    )
    verification_model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    if not runtime_path.exists():
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError("confirmatory output directory is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        device = _device(config["selection_device"])
        selection_model, _ = load_output_moe_checkpoint(
            checkpoint, map_location=device
        )
        selection_model.to(device).eval()
        if selection_manifest is not None:
            manifest = json.load(selection_manifest.open(encoding="utf-8"))
            model_id = str(config["model_id"])
            registered = manifest["models"].get(model_id)
            if registered is None:
                raise RuntimeError(f"selection manifest lacks model {model_id}")
            if registered["checkpoint_sha256"] != _sha256(checkpoint):
                raise RuntimeError("selection manifest checkpoint hash mismatch")
            selected = list(manifest["samples"])
            expected_ranks = list(range(int(config["sample_count"])))
            if [int(row["sample_rank"]) for row in selected] != expected_ranks:
                raise RuntimeError("selection manifest ranks are not consecutive")
            indices = [int(row["dataset_index"]) for row in selected]
            if len(indices) != len(set(indices)) or any(
                index < 0 or index >= len(dataset) for index in indices
            ):
                raise RuntimeError("selection manifest indices are invalid")
            with torch.no_grad():
                for row in selected:
                    image, label = dataset[int(row["dataset_index"])]
                    prediction = int(
                        selection_model(image.unsqueeze(0).to(device)).argmax(1).item()
                    )
                    if prediction != int(label):
                        raise RuntimeError(
                            f"manifest rank {row['sample_rank']} is not clean-correct"
                        )
            selection_record = {
                "selection_rule": manifest["selection_rule"],
                "selection_manifest": str(selection_manifest),
                "selection_manifest_sha256": _sha256(selection_manifest),
                "samples": selected,
            }
        else:
            assert development_indices is not None
            stop = int(config["rank_start"]) + int(config["sample_count"])
            prefix = _select_clean_correct(selection_model, dataset, device, stop)
            frozen_development = json.load(
                development_indices.open(encoding="utf-8")
            )["indices"]
            if prefix[: len(frozen_development)] != frozen_development:
                raise RuntimeError("clean-correct rank prefix differs from development")
            start = int(config["rank_start"])
            selected = [
                {"sample_rank": rank, "dataset_index": prefix[rank]}
                for rank in range(start, stop)
            ]
            selection_record = {
                "selection_rule": "deterministic clean-correct ranks",
                "verified_development_prefix": len(frozen_development),
                "samples": selected,
            }
        _write_json(
            indices_path,
            selection_record,
        )
        runtime = {
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "git_head": _git_value("rev-parse", "HEAD"),
            "checkpoint_sha256": _sha256(checkpoint),
            "development_sample_indices_sha256": (
                _sha256(development_indices)
                if development_indices is not None
                else None
            ),
            "selection_manifest_sha256": (
                _sha256(selection_manifest) if selection_manifest is not None else None
            ),
            "torchvision_root": str(dataset_root),
            "numerical_safety": actual_policy,
            "config": config,
        }
        _write_json(runtime_path, runtime)
        del selection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        runtime = json.load(runtime_path.open(encoding="utf-8"))
        if runtime["source_config_sha256"] != _sha256(config_path):
            raise RuntimeError("confirmatory source config changed between stages")
        if runtime["git_head"] != _git_value("rev-parse", "HEAD"):
            raise RuntimeError("confirmatory implementation HEAD changed between stages")
        if runtime["checkpoint_sha256"] != _sha256(checkpoint):
            raise RuntimeError("confirmatory checkpoint changed between stages")
        if selection_manifest is not None and runtime.get(
            "selection_manifest_sha256"
        ) != _sha256(selection_manifest):
            raise RuntimeError("selection manifest changed between stages")
    selected = json.load(indices_path.open(encoding="utf-8"))["samples"]
    if len(selected) != int(config["sample_count"]):
        raise RuntimeError("confirmatory sample count changed")
    return output_dir, verification_model, dataset, selected, runtime


def _run_census_row(
    model,
    dataset,
    selection: dict[str, int],
    epsilon_item: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    rank = int(selection["sample_rank"])
    index = int(selection["dataset_index"])
    epsilon = float(epsilon_item["value"])
    image, label = dataset[index]
    x = image.unsqueeze(0).double()
    lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
    with torch.no_grad():
        clean_output, clean_route = model.forward_with_routing(x)
    prediction = int(clean_output.argmax(dim=1).item())
    if prediction != int(label):
        raise RuntimeError(f"rank {rank} is not clean-correct")

    candidate_started = time.monotonic()
    ibp_lb, ibp_ub = _ibp_router_bounds(model.router, lower, upper)
    ibp = _candidate_tuple(ibp_lb, ibp_ub, model.spec.top_k)
    zono_lb, zono_ub = _zonotope_router_bounds(model.router, lower, upper)
    zono = _candidate_tuple(zono_lb, zono_ub, model.spec.top_k)
    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[prediction])
    program = build_act_moe_program(
        model,
        center=x,
        lower=lower,
        upper=upper,
        output_spec=output_spec,
    )
    router = _propagate_component(program.router)
    if not isinstance(router.output_hz, SparseHZono) or not router.output_hz.exact:
        raise RuntimeError("confirmatory exact-router label requires exact sparse HZ")
    candidates = analyze_candidates(
        router.output_hz,
        model.spec.top_k,
        input_hz=router.input_hz,
        time_limit_per_expert=float(config["candidate_query_timeout"]),
        router_exact=True,
    )
    pairs = analyze_topk_sets(
        router.output_hz,
        model.spec.top_k,
        time_limit_per_set=float(config["candidate_query_timeout"]),
        router_exact=True,
    )
    candidate_seconds = time.monotonic() - candidate_started
    base = {
        "sample_rank": rank,
        "dataset_index": index,
        "label": int(label),
        "clean_prediction": prediction,
        "clean_topk_set": sorted(
            int(value) for value in clean_route.indices[0].tolist()
        ),
        "epsilon_label": epsilon_item["label"],
        "epsilon": epsilon,
        "ibp_candidates": list(ibp),
        "ibp_candidate_count": len(ibp),
        "zonotope_candidates": list(zono),
        "zonotope_candidate_count": len(zono),
        "exact_candidates": list(candidates.candidates),
        "exact_candidate_count": len(candidates.candidates),
        "exact_candidate_minimal": candidates.minimal,
        "exact_feasible_pairs": [list(pair) for pair in pairs.feasible],
        "exact_feasible_pair_count": len(pairs.feasible),
        "exact_pairs_complete": pairs.exact,
        "candidate_seconds": candidate_seconds,
    }
    if not candidates.minimal or not pairs.exact:
        return {
            **base,
            "status": "UNKNOWN_CANDIDATE_FEASIBILITY",
            "route_set_unstable": None,
            "branches": [],
            "propagation_seconds": 0.0,
            "total_seconds": time.monotonic() - started,
        }

    propagation_started = time.monotonic()
    unguarded = {
        expert: _propagate_component(net)
        for expert, net in enumerate(program.experts)
    }
    by_expert = {branch.expert: branch for branch in candidates.branches}
    support_config = _support_config(config)
    branch_rows: list[dict[str, Any]] = []
    branch_widths: list[int] = []
    for expert in candidates.candidates:
        branch = by_expert[expert]
        guarded = _propagate_component(
            program.experts[expert],
            entry_hz=branch.guarded_input,
            hybridz_config=support_config,
        )
        support = _support_summary(guarded.guarded_support)
        after = max(0, guarded.binary_width - branch.guarded_input.n_bin)
        accounting = guard_binary_accounting(
            unguarded[expert].unstable_total,
            after,
            support,
        )
        branch_widths.append(guarded.binary_width)
        branch_rows.append(
            {
                "expert": expert,
                "route_membership_binaries": branch.selection_binaries,
                "route_conditioned_width": guarded.binary_width,
                "support": support,
                "guard_accounting": accounting.as_dict(),
            }
        )
    propagation_seconds = time.monotonic() - propagation_started
    route_unstable = len(pairs.feasible) > 1
    route_binaries = model.spec.num_experts if route_unstable else 0
    all_expert_width = sum(value.unstable_total for value in unguarded.values())
    candidate_expert_width = sum(
        unguarded[expert].unstable_total for expert in candidates.candidates
    )
    result = {
        **base,
        "status": "COMPLETE",
        "route_set_unstable": route_unstable,
        "router_binary_count": router.binary_width,
        "monolithic_route_binary_count": route_binaries,
        "structural_monolithic_width": (
            router.binary_width + route_binaries + all_expert_width
        ),
        "candidate_pruned_monolithic_width": (
            router.binary_width + route_binaries + candidate_expert_width
        ),
        "route_conditioned_widths": branch_widths,
        "route_conditioned_max_width": max(branch_widths, default=0),
        "route_conditioned_mean_width": (
            statistics.mean(branch_widths) if branch_widths else 0.0
        ),
        "branches": branch_rows,
        "propagation_seconds": propagation_seconds,
        "total_seconds": time.monotonic() - started,
    }
    result["instance_timeout_exceeded"] = (
        result["total_seconds"] > float(config["instance_timeout_seconds"])
    )
    return result


def _find_route_upper(model, x, clean_set, config) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    unresolved = False
    for epsilon in config["route_radius"]["search_grid"]:
        report = _router_route_change(
            model,
            x,
            clean_set,
            float(epsilon),
            query_timeout=float(config["route_radius"]["query_timeout"]),
        )
        history.append({"epsilon": float(epsilon), **report})
        if report["status"] == "unknown":
            retry = _router_route_change(
                model,
                x,
                clean_set,
                float(epsilon),
                query_timeout=float(config["route_radius"]["retry_timeout"]),
            )
            history.append({"epsilon": float(epsilon), "retry": True, **retry})
            report = retry
        if report["status"] == "unstable":
            return {"status": "found", "upper": float(epsilon), "history": history}
        if report["status"] == "unknown":
            unresolved = True
    return {
        "status": "unresolved" if unresolved else "not_found",
        "upper": None,
        "history": history,
    }


def _save_gate_witness(
    boundary_dir: Path,
    rank: int,
    candidate: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    relative = Path("witnesses") / f"gate_rank{rank}.pt"
    path = boundary_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    torch.save({"input": candidate.detach().cpu(), "metadata": metadata}, path)
    return str(relative), _sha256(path)


def _run_boundary_row(
    model,
    dataset,
    selection: dict[str, int],
    boundary_dir: Path,
    runtime: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    rank = int(selection["sample_rank"])
    index = int(selection["dataset_index"])
    image, label = dataset[index]
    x = image.unsqueeze(0).double()
    with torch.no_grad():
        output, route = model.forward_with_routing(x)
    prediction = int(output.argmax(dim=1).item())
    if prediction != int(label):
        raise RuntimeError(f"rank {rank} is not clean-correct")
    clean_set = sorted(int(value) for value in route.indices[0].tolist())
    search = _find_route_upper(model, x, clean_set, config)
    if search["status"] != "found":
        reason = (
            "ROUTE_SEARCH_UNRESOLVED"
            if search["status"] == "unresolved"
            else "NO_ROUTE_BOUNDARY_WITHIN_SEARCH"
        )
        return {
            "sample_rank": rank,
            "dataset_index": index,
            "label": int(label),
            "clean_prediction": prediction,
            "clean_topk_set": clean_set,
            "status": "UNKNOWN",
            "reason": reason,
            "route_search": search,
            "unique_safe": False,
            "full_model_witness_valid": False,
            "f0_invoked": False,
            "total_seconds": time.monotonic() - started,
        }
    bracket = exact_route_change_bracket(
        model,
        x,
        clean_set,
        search["upper"],
        steps=int(config["route_radius"]["bisection_steps"]),
        query_timeout=float(config["route_radius"]["query_timeout"]),
        retry_timeout=float(config["route_radius"]["retry_timeout"]),
    )
    epsilon = float(config["route_radius"]["primary_multiplier"]) * float(
        bracket["upper"]
    )
    gate_config = {
        "candidate_query_timeout": config["candidate_query_timeout"],
        "support": config["support"],
        "solver": config["solver"],
        "matched_no_support_solve": True,
        "return_witness_tensor": True,
    }
    gate_started = time.monotonic()
    gate = diagnose_radius(
        model=model,
        x=x,
        label=int(label),
        clean_prediction=prediction,
        clean_set=clean_set,
        epsilon=epsilon,
        epsilon_multiplier=float(config["route_radius"]["primary_multiplier"]),
        bracket=bracket,
        config=gate_config,
    )
    gate_seconds = time.monotonic() - gate_started
    gate_candidate = gate.pop("_counterexample_input", None)
    gate_status, gate_reason = gate["status"], gate["reason"]
    _write_json(
        boundary_dir / "progress.json",
        {
            "sample_rank": rank,
            "dataset_index": index,
            "epsilon": epsilon,
            "gate_status": gate_status,
            "gate_reason": gate_reason,
            "active_stage": "TIER1_COMPLETE",
            "f0_invoked": False,
        },
    )
    f0 = None
    f0_seconds = 0.0
    witness_path = witness_hash = None
    final_status, final_reason = gate_status, gate_reason
    full_witness = bool(gate["full_model_witness_valid"])
    if gate_status == "SAFE":
        final_status, final_reason = "SAFE", "SAFE_GATE_ELIMINATION"
    elif gate_status == "UNSAFE":
        if gate_candidate is None:
            raise RuntimeError("gate UNSAFE did not return its concrete input")
        witness_path, witness_hash = _save_gate_witness(
            boundary_dir,
            rank,
            gate_candidate,
            {
                "sample_rank": rank,
                "dataset_index": index,
                "epsilon": epsilon,
                "clean_prediction": prediction,
                "counterexample_prediction": gate["counterexample_prediction"],
                "counterexample_topk_set": gate["counterexample_topk_set"],
            },
        )
        final_status, final_reason = "UNSAFE", "UNSAFE_FULL_FORWARD"
    elif gate_reason in SEMANTIC_REASONS:
        f0_started = time.monotonic()
        _write_json(
            boundary_dir / "progress.json",
            {
                "sample_rank": rank,
                "dataset_index": index,
                "epsilon": epsilon,
                "gate_status": gate_status,
                "gate_reason": gate_reason,
                "active_stage": "TIER2_F0",
                "f0_invoked": True,
                "f0_started_monotonic": f0_started,
            },
        )
        parent_id = hashlib.sha256(
            f"confirmatory:{runtime['source_config_sha256']}:{rank}:{epsilon:.17g}".encode()
        ).hexdigest()
        f0 = _run_parent_row(
            selection={
                "parent_row_id": parent_id,
                "parent_line_number": rank + 1,
                "parent_row_sha256": parent_id,
                "parent_artifact_sha256": runtime["source_config_sha256"],
                "parent": {
                    "sample_rank": rank,
                    "dataset_index": index,
                    "epsilon": epsilon,
                    "epsilon_multiplier": float(
                        config["route_radius"]["primary_multiplier"]
                    ),
                    "clean_prediction": prediction,
                    "clean_topk_set": clean_set,
                    "status": gate_status,
                    "reason": gate_reason,
                },
            },
            model=model,
            dataset=dataset,
            output_dir=boundary_dir,
            config=config["f0"],
        )
        f0_seconds = time.monotonic() - f0_started
        final_status, final_reason = f0["status"], f0["reason"]
        full_witness = bool(f0["full_model_witness_valid"])
        witness_path, witness_hash = f0["witness_path"], f0["witness_sha256"]
    feasible_pairs = gate.get("feasible_route_sets") or []
    route_invariance = "UNKNOWN" if len(feasible_pairs) > 1 else "INVARIANT"
    unique_safe = (
        len(feasible_pairs) > 1
        and route_invariance == "UNKNOWN"
        and final_status == "SAFE"
    )
    row = {
        "sample_rank": rank,
        "dataset_index": index,
        "label": int(label),
        "clean_prediction": prediction,
        "clean_topk_set": clean_set,
        "epsilon": epsilon,
        "route_lower": bracket["lower"],
        "route_upper": bracket["upper"],
        "bracket_width": float(bracket["upper"]) - float(bracket["lower"]),
        "bisection_complete": bracket["bisection_complete"],
        "bracket": bracket,
        "exact_feasible_pairs": feasible_pairs,
        "exact_feasible_pair_count": len(feasible_pairs),
        "route_invariance_status": route_invariance,
        "status": final_status,
        "reason": final_reason,
        "unique_safe": unique_safe,
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "gate": gate,
        "f0_invoked": f0 is not None,
        "f0_status": f0["status"] if f0 else None,
        "f0_reason": f0["reason"] if f0 else None,
        "f0": f0,
        "f0_time_observation": (
            {
                "kind": "OBSERVED",
                "seconds": f0_seconds,
                "lower_bound_seconds": None,
            }
            if f0 is not None
            else {
                "kind": "NOT_INVOKED",
                "seconds": None,
                "lower_bound_seconds": None,
            }
        ),
        "full_model_witness_valid": full_witness,
        "counterexample_prediction": (
            f0["counterexample_prediction"]
            if f0 and full_witness
            else gate.get("counterexample_prediction")
        ),
        "counterexample_topk_set": (
            f0["counterexample_topk_set"]
            if f0 and full_witness
            else gate.get("counterexample_topk_set")
        ),
        "witness_path": witness_path,
        "witness_sha256": witness_hash,
        "candidate_seconds": gate.get("candidate_seconds", 0.0),
        "gate_seconds": gate_seconds,
        "f0_seconds": f0_seconds,
        "total_seconds": time.monotonic() - started,
    }
    row["instance_timeout_exceeded"] = (
        row["total_seconds"] > float(config["instance_timeout_seconds"])
    )
    return row


def _boundary_child(
    result_path: Path,
    *,
    model,
    dataset,
    selection: dict[str, int],
    work_dir: Path,
    runtime: dict[str, Any],
    config: dict[str, Any],
    row_runner=_run_boundary_row,
) -> None:
    """Run one boundary row in a killable child and persist its return value."""
    try:
        if model is None or dataset is None:
            checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
            model, payload = load_output_moe_checkpoint(
                checkpoint, map_location="cpu"
            )
            model.double().eval()
            dataset = _load_dataset(payload["dataset"], False, download=False)
        row = row_runner(
            model,
            dataset,
            selection,
            work_dir,
            runtime,
            config,
        )
    except Exception as exc:
        row = {
            "sample_rank": selection["sample_rank"],
            "dataset_index": selection["dataset_index"],
            "status": "ERROR",
            "reason": "EXPLICIT_NUMERICAL_OR_RUNNER_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "total_seconds": 0.0,
        }
    _write_json(result_path, row)


def _promote_row_artifacts(work_dir: Path, stage_dir: Path) -> None:
    """Promote completed-row artifacts; timed-out work remains quarantined."""
    for source in sorted(work_dir.rglob("*")):
        if not source.is_file() or source.name in {"progress.json", "row.json"}:
            continue
        destination = stage_dir / source.relative_to(work_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite {destination}")
        source.replace(destination)


def _run_boundary_with_deadline(
    *,
    model,
    dataset,
    selection: dict[str, int],
    stage_dir: Path,
    runtime: dict[str, Any],
    config: dict[str, Any],
    row_runner=_run_boundary_row,
) -> dict[str, Any]:
    """Enforce the preregistered wall deadline with process termination."""
    rank = int(selection["sample_rank"])
    work_dir = stage_dir / "row_work" / f"rank{rank}"
    work_dir.mkdir(parents=True, exist_ok=False)
    result_path = work_dir / "row.json"
    context = mp.get_context("spawn")
    process = context.Process(
        target=_boundary_child,
        kwargs={
            "result_path": result_path,
            "model": model,
            "dataset": dataset,
            "selection": selection,
            "work_dir": work_dir,
            "runtime": runtime,
            "config": config,
            "row_runner": row_runner,
        },
        daemon=False,
    )
    started = time.monotonic()
    process.start()
    timeout = float(config["instance_timeout_seconds"])
    try:
        process.join(timeout=timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            elapsed = time.monotonic() - started
            progress_path = work_dir / "progress.json"
            progress: dict[str, Any] = {}
            if progress_path.exists():
                try:
                    with progress_path.open(encoding="utf-8") as handle:
                        progress = json.load(handle)
                except (OSError, ValueError):
                    progress = {}
            f0_invoked = bool(progress.get("f0_invoked", False))
            f0_started_monotonic = progress.get("f0_started_monotonic")
            f0_lower_bound = (
                max(0.0, time.monotonic() - float(f0_started_monotonic))
                if f0_invoked and f0_started_monotonic is not None
                else None
            )
            return {
                "sample_rank": rank,
                "dataset_index": int(selection["dataset_index"]),
                "status": "TIMEOUT",
                "reason": "INSTANCE_HARD_DEADLINE",
                "unique_safe": False,
                "gate_status": progress.get("gate_status"),
                "gate_reason": progress.get("gate_reason"),
                "f0_invoked": f0_invoked,
                "f0_seconds": None,
                "f0_time_observation": (
                    {
                        "kind": "RIGHT_CENSORED_AT_INSTANCE_DEADLINE",
                        "seconds": None,
                        "lower_bound_seconds": f0_lower_bound,
                    }
                    if f0_invoked
                    else {
                        "kind": "NOT_INVOKED",
                        "seconds": None,
                        "lower_bound_seconds": None,
                    }
                ),
                "active_stage_at_deadline": progress.get("active_stage"),
                "full_model_witness_valid": False,
                "deadline_seconds": timeout,
                "deadline_enforced": True,
                "deadline_overshoot_seconds": max(0.0, elapsed - timeout),
                "partial_work_dir": str(work_dir.relative_to(stage_dir)),
                "total_seconds": elapsed,
            }
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    elapsed = time.monotonic() - started
    if process.exitcode != 0 or not result_path.exists():
        return {
            "sample_rank": rank,
            "dataset_index": int(selection["dataset_index"]),
            "status": "ERROR",
            "reason": "BOUNDARY_CHILD_FAILED",
            "error": f"child exit code {process.exitcode}",
            "deadline_seconds": timeout,
            "deadline_enforced": True,
            "total_seconds": elapsed,
        }
    with result_path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    internal_seconds = float(row.get("total_seconds", 0.0))
    _promote_row_artifacts(work_dir, stage_dir)
    row["internal_total_seconds"] = internal_seconds
    row["total_seconds"] = elapsed
    row["deadline_seconds"] = timeout
    row["deadline_enforced"] = True
    row["deadline_overshoot_seconds"] = max(0.0, elapsed - timeout)
    row["instance_timeout_exceeded"] = False
    return row


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    values = [float(value) for value in values]
    if not values:
        return {"median": None, "q1": None, "q3": None, "p90": None}
    if len(values) == 1:
        return {
            "median": values[0],
            "q1": values[0],
            "q3": values[0],
            "p90": values[0],
        }
    q = statistics.quantiles(values, n=4, method="inclusive")
    d = statistics.quantiles(values, n=10, method="inclusive")
    return {
        "median": statistics.median(values),
        "q1": q[0],
        "q3": q[2],
        "p90": d[8],
    }


def _census_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    unstable = [row for row in complete if row["route_set_unstable"]]
    conditional = [row for row in complete if row["exact_candidate_count"] > 2]
    width = [
        row["route_conditioned_max_width"]
        / row["candidate_pruned_monolithic_width"]
        for row in complete
        if row["candidate_pruned_monolithic_width"]
    ]
    conditional_width = [
        row["route_conditioned_max_width"]
        / row["candidate_pruned_monolithic_width"]
        for row in conditional
        if row["candidate_pruned_monolithic_width"]
    ]
    branches = [branch for row in complete for branch in row["branches"]]
    return {
        "rows": len(rows),
        "samples": len({row["sample_rank"] for row in rows}),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "route_unstable_rows": len(unstable),
        "exact_reduces_ibp_rows": sum(
            row["exact_candidate_count"] < row["ibp_candidate_count"]
            for row in unstable
        ),
        "exact_reduces_zonotope_rows": sum(
            row["exact_candidate_count"] < row["zonotope_candidate_count"]
            for row in unstable
        ),
        "width_ratio_unconditional": _quantiles(width),
        "width_ratio_candidate_gt_topk": _quantiles(conditional_width),
        "guard_branches": len(branches),
        "guard_branches_with_elimination": sum(
            branch["guard_accounting"]["binary_eliminated"] > 0
            for branch in branches
        ),
        "guard_accounting": {
            key: sum(branch["guard_accounting"][key] for branch in branches)
            for key in (
                "binaries_before",
                "binaries_after",
                "binary_eliminated",
                "lp_support_eliminated",
                "milp_support_eliminated",
                "structural_or_propagation_eliminated",
            )
        },
        "support_seconds": sum(
            branch["support"]["seconds"] for branch in branches
        ),
        "total_seconds": sum(row["total_seconds"] for row in rows),
    }


def _boundary_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(row["status"] for row in rows)
    reason = Counter(row["reason"] for row in rows)
    invoked = [row for row in rows if row.get("f0_invoked")]
    semantic = [row for row in rows if row.get("gate_reason") in SEMANTIC_REASONS]
    gate_branches = [
        branch
        for row in rows
        for branch in (row.get("gate") or {}).get("branches", [])
    ]
    observed_f0_seconds = [
        float(row["f0_seconds"])
        for row in invoked
        if row.get("f0_seconds") is not None
    ]
    right_censored_f0 = [
        row
        for row in invoked
        if row.get("f0_seconds") is None
        or (row.get("f0_time_observation") or {}).get("kind")
        == "RIGHT_CENSORED_AT_INSTANCE_DEADLINE"
    ]
    known_censored_lower_bounds = [
        float(row["f0_time_observation"]["lower_bound_seconds"])
        for row in right_censored_f0
        if (row.get("f0_time_observation") or {}).get("lower_bound_seconds")
        is not None
    ]
    return {
        "rows": len(rows),
        "samples": len({row["sample_rank"] for row in rows}),
        "status_counts": dict(status),
        "reason_counts": dict(reason),
        "solved_rows": status["SAFE"] + status["UNSAFE"],
        "unique_safe_samples": sum(bool(row.get("unique_safe")) for row in rows),
        "unique_safe_sample_ranks": [
            row["sample_rank"] for row in rows if row.get("unique_safe")
        ],
        "base_semantic_incompleteness": len(semantic),
        "f0_invoked": len(invoked),
        "f0_resolved": sum(row["status"] in {"SAFE", "UNSAFE"} for row in invoked),
        "f0_added_safe": sum(row["status"] == "SAFE" for row in invoked),
        "f0_added_unsafe": sum(row["status"] == "UNSAFE" for row in invoked),
        "f0_remaining_unresolved": sum(
            row["status"] not in {"SAFE", "UNSAFE"} for row in invoked
        ),
        "matched_no_support_solved_branches": sum(
            branch.get("matched_no_support_status") in {"certified", "falsified"}
            for branch in gate_branches
        ),
        "support_solved_branches": sum(
            branch.get("branch_status") in {"certified", "falsified"}
            for branch in gate_branches
        ),
        "f0_seconds": sum(observed_f0_seconds),
        "f0_seconds_semantics": "observed_completed_F0_rows_only",
        "f0_observed_time_rows": len(observed_f0_seconds),
        "f0_right_censored_time_rows": len(right_censored_f0),
        "f0_right_censored_known_lower_bound_seconds": sum(
            known_censored_lower_bounds
        ),
        "f0_paired_runtime_overhead": _quantiles(observed_f0_seconds),
        "total_seconds": sum(float(row["total_seconds"]) for row in rows),
        "instance_timeout_exceeded": sum(
            bool(row.get("instance_timeout_exceeded")) for row in rows
        ),
        "hard_deadline_timeouts": sum(
            row.get("reason") == "INSTANCE_HARD_DEADLINE" for row in rows
        ),
    }


def _run_stage(args) -> dict[str, Any]:
    config_path = _inside(Path(args.config), PROJECT_ROOT)
    config = json.load(config_path.open(encoding="utf-8"))
    output_dir, model, dataset, selected, runtime = _prepare(config_path, config)
    stage = args.stage
    if stage == "boundary":
        census_summary_path = output_dir / "census/summary.json"
        if not census_summary_path.exists():
            raise RuntimeError("boundary stage requires the completed census")
        census_summary = json.load(census_summary_path.open(encoding="utf-8"))
        if census_summary["rows"] != 4 * int(config["sample_count"]):
            raise RuntimeError("boundary stage requires all census rows")
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": stage_dir / "results.jsonl",
        "csv": stage_dir / "results.csv",
        "summary": stage_dir / "summary.json",
        "log": stage_dir / f"{stage}.log",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing or (stage_dir / "witnesses").exists():
        raise RuntimeError(f"confirmatory stage refuses to overwrite {existing}")
    fields = CENSUS_FIELDS if stage == "census" else BOUNDARY_FIELDS
    rows: list[dict[str, Any]] = []
    with (
        paths["jsonl"].open("x", encoding="utf-8") as jsonl_handle,
        paths["csv"].open("x", newline="", encoding="utf-8") as csv_handle,
        paths["log"].open("x", encoding="utf-8") as log_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        expected = (
            len(selected) * len(config["fixed_epsilons"])
            if stage == "census"
            else len(selected)
        )
        _log(log_handle, f"START stage={stage} rows={expected} head={runtime['git_head']}")
        position = 0
        for selection in selected:
            epsilon_items = config["fixed_epsilons"] if stage == "census" else [None]
            for epsilon_item in epsilon_items:
                position += 1
                try:
                    if stage == "census":
                        row = _run_census_row(
                            model, dataset, selection, epsilon_item, config
                        )
                    else:
                        row = _run_boundary_with_deadline(
                            model=None,
                            dataset=None,
                            selection=selection,
                            stage_dir=stage_dir,
                            runtime=runtime,
                            config=config,
                        )
                except Exception as exc:
                    row = {
                        "sample_rank": selection["sample_rank"],
                        "dataset_index": selection["dataset_index"],
                        "epsilon_label": (
                            epsilon_item["label"] if epsilon_item else None
                        ),
                        "epsilon": epsilon_item["value"] if epsilon_item else None,
                        "status": "ERROR",
                        "reason": "EXPLICIT_NUMERICAL_OR_RUNNER_ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "total_seconds": 0.0,
                    }
                rows.append(row)
                _append_json(jsonl_handle, row)
                writer.writerow(_csv_row(row, fields))
                csv_handle.flush()
                os.fsync(csv_handle.fileno())
                _log(
                    log_handle,
                    f"ROW position={position}/{expected} rank={row['sample_rank']} "
                    f"status={row['status']} seconds={row.get('total_seconds', 0.0):.3f}",
                )
        summary = (
            _census_summary(rows) if stage == "census" else _boundary_summary(rows)
        )
        _write_json(paths["summary"], summary)
        _log(log_handle, f"DONE summary={json.dumps(summary, sort_keys=True)}")
    return {
        "stage": stage,
        "output_dir": str(stage_dir),
        "summary": summary,
        "manifest": {
            str(path.relative_to(stage_dir)): _sha256(path)
            for path in sorted(stage_dir.rglob("*"))
            if path.is_file()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=("census", "boundary"), required=True)
    args = parser.parse_args()
    print(json.dumps(_run_stage(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
