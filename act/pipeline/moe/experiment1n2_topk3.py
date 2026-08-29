# ===- act/pipeline/moe/experiment1n2_topk3.py - N2 Data Path -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""N2 staged verifier for normalized-sigmoid weighted top-3 MoEs.

This runner is deliberately separate from the top-2 F0 experiment.  It
enumerates every exact tie-inclusive unordered top-3 route set, retains the
set guard while propagating the three selected experts, and calls the generic
normalized weighted-top-k encoding.  Each linear safety row therefore uses
``k-1 == 2`` property-directed McCormick products.

A non-positive relaxation candidate is never reported as unsafe.  ``UNSAFE``
is emitted only after the candidate is replayed through the concrete full MoE
and changes the clean prediction inside the registered input box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from act.back_end.moe import (
    SAFE_WEIGHTED_RANGE,
    UNKNOWN_WEIGHTED_NUMERICAL,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    analyze_topk_sets,
    build_act_moe_program,
    build_weighted_topk_range,
    compute_normalized_topk_gate_box,
    condition_topk_set,
    guarded_input_topk_set,
    linear_safety_rows,
    load_output_moe_checkpoint,
    normalized_gate_support,
    shared_input_experts_hz,
    solve_weighted_topk_range,
)
from act.back_end.solver.solver_hz import SparseHZono
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
from act.pipeline.moe.experiment1f0 import _support_status
from act.util.path_config import get_torchvision_data_root
from act.util.stats import VerifyStatus


DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/experiment1n2_topk3_seed0_r2.json"
)
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
SAFE_GATE_ELIMINATION = "SAFE_GATE_ELIMINATION"
SAFE_N2_STAGED = "SAFE_N2_STAGED"


def _append_json_line(handle, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the checkpoint fields that make the frozen N2 target auditable."""
    model = config["model"]
    training = config["training"]
    return {
        "dataset": str(model["dataset"]),
        "factory_config": {
            "input_shape": list(model["input_shape"]),
            "num_classes": int(model["num_classes"]),
            "num_experts": int(model["num_experts"]),
            "top_k": int(model["top_k"]),
            "gate": str(model["gate"]),
            "router_hidden": list(model["router_hidden"]),
            "expert_hidden": list(model["expert_hidden"]),
            "seed": int(training["seed"]),
        },
        "training_config": {
            "balance_loss": str(training["balance_loss"]),
            "balance_coefficient": float(training["balance_coefficient"]),
            "validation_fraction": float(training["validation_fraction"]),
            "epochs": int(training["epochs"]),
            "batch_size": int(training["batch_size"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "seed": int(training["seed"]),
        },
    }


def validate_checkpoint_contract(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject checkpoints whose recorded provenance differs from the target."""
    expected = _expected_contract(config)
    observed = {
        "dataset": payload.get("dataset"),
        "factory_config": payload.get("factory_config"),
        "training_config": {
            key: (payload.get("training_config") or {}).get(key)
            for key in expected["training_config"]
        },
    }
    errors: list[str] = []
    if observed["dataset"] != expected["dataset"]:
        errors.append(
            f"dataset: expected {expected['dataset']!r}, got {observed['dataset']!r}"
        )
    observed_factory = observed["factory_config"] or {}
    for key, value in expected["factory_config"].items():
        if observed_factory.get(key) != value:
            errors.append(
                f"factory_config.{key}: expected {value!r}, "
                f"got {observed_factory.get(key)!r}"
            )
    observed_training = observed["training_config"]
    for key, value in expected["training_config"].items():
        if observed_training.get(key) != value:
            errors.append(
                f"training_config.{key}: expected {value!r}, "
                f"got {observed_training.get(key)!r}"
            )
    if errors:
        raise RuntimeError("checkpoint violates frozen N2 contract: " + "; ".join(errors))
    return {"expected": expected, "observed": observed, "matched": True}


def select_clean_correct_indices(
    model,
    dataset,
    ranks: Sequence[int],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> list[dict[str, int]]:
    """Select deterministic ranks from the dataset's clean-correct stream."""
    requested = tuple(int(value) for value in ranks)
    if not requested or any(value < 0 for value in requested):
        raise ValueError("clean-correct ranks must be non-negative")
    if len(set(requested)) != len(requested) or tuple(sorted(requested)) != requested:
        raise ValueError("clean-correct ranks must be unique and sorted")
    wanted = set(requested)
    largest = requested[-1]
    selected: list[dict[str, int]] = []
    clean_rank = 0
    offset = 0
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            for local in (predictions == labels).nonzero(as_tuple=False).flatten():
                if clean_rank in wanted:
                    selected.append(
                        {
                            "sample_rank": clean_rank,
                            "dataset_index": offset + int(local.item()),
                        }
                    )
                if clean_rank == largest:
                    return selected
                clean_rank += 1
            offset += int(labels.numel())
    raise RuntimeError(
        f"dataset contains only {clean_rank} clean-correct samples; "
        f"rank {largest} was requested"
    )


def prepare_selection_model(model, device: torch.device):
    """Move the reconstructed checkpoint module to the cohort device."""

    return model.to(device).eval()


def aggregate_property_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Aggregate one guarded route set without promoting relaxation witnesses."""
    if any(row.get("full_model_witness_valid") is True for row in rows):
        return "UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK
    if rows and all(row.get("status") == "SAFE" for row in rows):
        return "SAFE", SAFE_WEIGHTED_RANGE
    reasons = {row.get("reason") for row in rows}
    for reason in (
        UNKNOWN_WEIGHTED_SOLVER_LIMIT,
        UNKNOWN_WEIGHTED_NUMERICAL,
        UNKNOWN_WEIGHTED_RELAXATION,
    ):
        if reason in reasons:
            return "UNKNOWN", reason
    return "UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL


def aggregate_route_sets(
    rows: Sequence[Mapping[str, Any]],
    *,
    enumeration_exact: bool,
) -> tuple[str, str]:
    """Safety requires every legal tie-inclusive route set to be discharged."""
    if any(
        row.get("status") == "UNSAFE"
        and row.get("full_model_witness_valid") is True
        for row in rows
    ):
        return "UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK
    if not enumeration_exact:
        return "UNKNOWN", UNKNOWN_WEIGHTED_SOLVER_LIMIT
    if rows and all(row.get("status") == "SAFE" for row in rows):
        reasons = {row.get("reason") for row in rows}
        if reasons == {SAFE_GATE_ELIMINATION}:
            return "SAFE", SAFE_GATE_ELIMINATION
        if reasons == {SAFE_WEIGHTED_RANGE}:
            return "SAFE", SAFE_WEIGHTED_RANGE
        return "SAFE", SAFE_N2_STAGED
    reasons = {row.get("reason") for row in rows}
    for reason in (
        UNKNOWN_WEIGHTED_SOLVER_LIMIT,
        UNKNOWN_WEIGHTED_NUMERICAL,
        UNKNOWN_WEIGHTED_RELAXATION,
    ):
        if reason in reasons:
            return "UNKNOWN", reason
    return "UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL


def _candidate_replay(
    model,
    candidate: torch.Tensor | None,
    *,
    clean: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    clean_prediction: int,
) -> dict[str, Any]:
    checked = _forward_validate(
        model,
        candidate,
        lower=lower,
        upper=upper,
        clean_prediction=clean_prediction,
    )
    if candidate is None or candidate.unsqueeze(0).shape != clean.shape:
        checked["linf_distance"] = None
    else:
        checked["linf_distance"] = float(
            (candidate.unsqueeze(0) - clean).abs().max().item()
        )
    return checked


def _save_witness(
    output_dir: Path,
    *,
    sample_rank: int,
    route_set: Sequence[int],
    property_index: int,
    candidate: torch.Tensor,
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
    route_label = "_".join(str(int(value)) for value in route_set)
    relative = Path("witnesses") / (
        f"rank{sample_rank}_set{route_label}_property{property_index}.pt"
    )
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite witness {path}")
    torch.save(
        {"input": candidate.detach().cpu(), "metadata": dict(metadata)},
        path,
    )
    return str(relative), _sha256(path)


def _propagation_record(propagation) -> dict[str, Any]:
    return {
        "binary_width": propagation.binary_width,
        "unstable_relu": propagation.unstable_total,
        "elapsed": propagation.elapsed,
        "guarded_support": list(propagation.guarded_support),
    }


def verify_route_set(
    *,
    model,
    program,
    router,
    route_set: Sequence[int],
    properties,
    clean: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    clean_prediction: int,
    sample_rank: int,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify all property rows for one exact unordered top-3 route set."""
    started = time.monotonic()
    selected = tuple(sorted(int(value) for value in route_set))
    expected_products = int(model.spec.top_k) - 1
    if len(selected) != int(model.spec.top_k):
        raise ValueError("route set size differs from model top_k")
    conditioned_router = condition_topk_set(router.output_hz, selected).hz
    guarded_entry = guarded_input_topk_set(
        router.input_hz,
        router.output_hz,
        selected,
    ).hz
    if not isinstance(conditioned_router, SparseHZono) or not isinstance(
        guarded_entry, SparseHZono
    ):
        raise RuntimeError("N2 requires sparse HZ router and guarded input frames")

    support_config = HybridZConfig(
        max_input_dim=int(config["support"]["max_input_dim"]),
        guarded_support_enabled=bool(config["support"]["enabled"]),
        guarded_support_lp_neurons=int(config["support"]["lp_neurons"]),
        guarded_support_milp_neurons=int(config["support"]["milp_neurons"]),
        guarded_support_lp_time_limit=float(config["support"]["lp_time_limit"]),
        guarded_support_milp_time_limit=float(
            config["support"]["milp_time_limit"]
        ),
    )
    propagated = {
        expert: _propagate_component(
            program.experts[expert],
            entry_hz=guarded_entry,
            hybridz_config=support_config,
        )
        for expert in selected
    }
    gate_elimination_rows: list[dict[str, Any]] = []
    gate_witness = None
    for expert in selected:
        decision = _solve_output(
            propagated[expert],
            OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[clean_prediction]),
            input_shape=tuple(clean.shape),
            time_limit=float(config["solver"]["gate_elimination_seconds"]),
        )
        replay = _candidate_replay(
            model,
            decision.counterexample,
            clean=clean,
            lower=lower,
            upper=upper,
            clean_prediction=clean_prediction,
        )
        if replay["valid"]:
            witness_path, witness_sha256 = _save_witness(
                output_dir,
                sample_rank=sample_rank,
                route_set=selected,
                property_index=-1,
                candidate=decision.counterexample,
                metadata={
                    "stage": "gate_elimination",
                    "sample_rank": sample_rank,
                    "route_set": list(selected),
                    "expert": expert,
                    "clean_prediction": clean_prediction,
                    "counterexample_prediction": replay["prediction"],
                    "counterexample_topk_set": replay["topk_set"],
                },
            )
            gate_witness = {
                **replay,
                "path": witness_path,
                "sha256": witness_sha256,
            }
        gate_elimination_rows.append(
            {
                "expert": expert,
                "solver_status": decision.status.value,
                "status": (
                    "UNSAFE"
                    if replay["valid"]
                    else (
                        "SAFE"
                        if decision.status == VerifyStatus.CERTIFIED
                        else "UNKNOWN"
                    )
                ),
                "solver_reason": decision.metadata.get("reason"),
                "solver_gap": decision.metadata.get("mip_gap"),
                "candidate_recovered": decision.counterexample is not None,
                "full_model_witness_valid": bool(replay["valid"]),
                "counterexample_prediction": replay["prediction"],
                "counterexample_topk_set": replay["topk_set"],
                "candidate_linf_distance": replay["linf_distance"],
            }
        )
        if gate_witness is not None:
            break

    common = {
        "route_set": list(selected),
        "gate_family": model.spec.gate.value,
        "gate_family_supported": normalized_gate_support(model.spec.gate)[0],
        "gate_family_support_reason": normalized_gate_support(model.spec.gate)[1],
        "expected_property_products": expected_products,
        "expert_propagation": {
            str(expert): _propagation_record(propagated[expert])
            for expert in selected
        },
        "gate_elimination": gate_elimination_rows,
    }
    if gate_witness is not None:
        return {
            **common,
            "status": "UNSAFE",
            "reason": UNSAFE_FULL_FORWARD_FALLBACK,
            "resolved_stage": "gate_elimination_full_forward_replay",
            "fallback_invoked": False,
            "actual_property_products": 0,
            "property_rows": [],
            "full_model_witness_valid": True,
            "counterexample_prediction": gate_witness["prediction"],
            "counterexample_topk_set": gate_witness["topk_set"],
            "witness_path": gate_witness["path"],
            "witness_sha256": gate_witness["sha256"],
            "seconds": time.monotonic() - started,
        }
    if gate_elimination_rows and all(
        row["status"] == "SAFE" for row in gate_elimination_rows
    ):
        return {
            **common,
            "status": "SAFE",
            "reason": SAFE_GATE_ELIMINATION,
            "resolved_stage": "gate_elimination",
            "fallback_invoked": False,
            "actual_property_products": 0,
            "property_rows": [],
            "full_model_witness_valid": False,
            "counterexample_prediction": None,
            "counterexample_topk_set": None,
            "witness_path": None,
            "witness_sha256": None,
            "seconds": time.monotonic() - started,
        }

    merged = shared_input_experts_hz(
        guarded_entry,
        {expert: propagated[expert].output_hz for expert in selected},
    )
    gate_box = compute_normalized_topk_gate_box(
        conditioned_router,
        selected,
        model.spec.gate,
        time_limit=float(config["solver"]["gate_support_seconds"]),
        relax_binaries=True,
    )
    property_rows: list[dict[str, Any]] = []
    witness = None
    for property_index, (q, constant) in enumerate(properties):
        encoding = build_weighted_topk_range(
            merged,
            conditioned_router,
            gate_box,
            q,
            constant,
            difference_time_limit=float(
                config["solver"]["difference_support_seconds"]
            ),
        )
        product_count = len(encoding.term_bounds)
        if product_count != expected_products:
            raise RuntimeError(
                f"N2 expected {expected_products} products, got {product_count}"
            )
        decision = solve_weighted_topk_range(
            encoding,
            input_shape=tuple(clean.shape),
            time_limit=float(config["solver"]["property_seconds"]),
            tolerance=float(config["solver"]["safety_tolerance"]),
        )
        replay = _candidate_replay(
            model,
            decision.candidate_input,
            clean=clean,
            lower=lower,
            upper=upper,
            clean_prediction=clean_prediction,
        )
        witness_path = witness_sha256 = None
        if replay["valid"]:
            witness_path, witness_sha256 = _save_witness(
                output_dir,
                sample_rank=sample_rank,
                route_set=selected,
                property_index=property_index,
                candidate=decision.candidate_input,
                metadata={
                    "sample_rank": sample_rank,
                    "route_set": list(selected),
                    "property_index": property_index,
                    "clean_prediction": clean_prediction,
                    "counterexample_prediction": replay["prediction"],
                    "counterexample_topk_set": replay["topk_set"],
                },
            )
            witness = {
                **replay,
                "path": witness_path,
                "sha256": witness_sha256,
            }
        status = "UNSAFE" if replay["valid"] else decision.status
        reason = (
            UNSAFE_FULL_FORWARD_FALLBACK if replay["valid"] else decision.reason
        )
        property_rows.append(
            {
                "property_index": property_index,
                "status": status,
                "reason": reason,
                "product_count": product_count,
                "free_experts": list(encoding.free_experts),
                "anchor": encoding.anchor,
                "minimum": decision.minimum,
                "solver_certified_lower_bound": (
                    decision.solver_certified_lower_bound
                ),
                "solver_bound_kind": decision.solver_bound_kind,
                "solver_status": decision.solver_status,
                "solver_gap": decision.solver_gap,
                "solver_seconds": decision.elapsed,
                "term_bounds": [
                    {
                        "lambda": [term.lambda_lower, term.lambda_upper],
                        "difference": [
                            term.difference_lower,
                            term.difference_upper,
                        ],
                        "product": [term.product_lower, term.product_upper],
                    }
                    for term in encoding.term_bounds
                ],
                "difference_support": (
                    _support_status(encoding.difference_support)
                    if encoding.difference_support is not None
                    else None
                ),
                "candidate_recovered": decision.candidate_input is not None,
                "full_model_witness_valid": bool(replay["valid"]),
                "counterexample_prediction": replay["prediction"],
                "counterexample_topk_set": replay["topk_set"],
                "candidate_linf_distance": replay["linf_distance"],
                "witness_path": witness_path,
                "witness_sha256": witness_sha256,
            }
        )
        if witness is not None:
            break
    status, reason = aggregate_property_rows(property_rows)
    return {
        **common,
        "status": status,
        "reason": reason,
        "resolved_stage": (
            "weighted_topk_range" if status == "SAFE" else "unresolved"
        ),
        "fallback_invoked": True,
        "actual_property_products": expected_products,
        "gate_weight_lower": list(gate_box.lower),
        "gate_weight_upper": list(gate_box.upper),
        "gate_support": (
            _support_status(gate_box.score_support)
            if gate_box.score_support is not None
            else None
        ),
        "shared_input_continuous": merged.shared_continuous,
        "shared_input_binary": merged.shared_binary,
        "private_continuous": list(merged.private_continuous),
        "private_binary": list(merged.private_binary),
        "property_rows": property_rows,
        "full_model_witness_valid": witness is not None,
        "counterexample_prediction": witness["prediction"] if witness else None,
        "counterexample_topk_set": witness["topk_set"] if witness else None,
        "witness_path": witness["path"] if witness else None,
        "witness_sha256": witness["sha256"] if witness else None,
        "seconds": time.monotonic() - started,
    }


def run_sample(
    *,
    model,
    dataset,
    selection: Mapping[str, int],
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    rank = int(selection["sample_rank"])
    index = int(selection["dataset_index"])
    image, label = dataset[index]
    clean = image.unsqueeze(0).double()
    epsilon = float(config["verification"]["epsilon"])
    lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
    with torch.no_grad():
        clean_output, clean_route = model.forward_with_routing(clean)
    clean_prediction = int(clean_output.argmax(dim=1).item())
    if clean_prediction != int(label):
        raise RuntimeError(f"clean-correct rank {rank} is no longer correct")
    clean_set = sorted(int(value) for value in clean_route.indices[0].tolist())
    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[clean_prediction])
    properties = linear_safety_rows(output_spec, int(clean_output.shape[1]))
    program = build_act_moe_program(
        model,
        center=clean,
        lower=lower,
        upper=upper,
        output_spec=output_spec,
    )
    router = _propagate_component(program.router)
    if not isinstance(router.output_hz, SparseHZono) or not router.output_hz.exact:
        raise RuntimeError("N2 exact label requires an unrelaxed sparse router HZ")
    route_sets = analyze_topk_sets(
        router.output_hz,
        model.spec.top_k,
        time_limit_per_set=float(config["solver"]["route_set_seconds"]),
        router_exact=True,
    )
    route_rows: list[dict[str, Any]] = []
    if route_sets.exact:
        for selected in route_sets.feasible:
            try:
                route_row = verify_route_set(
                    model=model,
                    program=program,
                    router=router,
                    route_set=selected,
                    properties=properties,
                    clean=clean,
                    lower=lower,
                    upper=upper,
                    clean_prediction=clean_prediction,
                    sample_rank=rank,
                    output_dir=output_dir,
                    config=config,
                )
            except Exception as exc:
                route_row = {
                    "route_set": list(selected),
                    "status": "UNKNOWN",
                    "reason": UNKNOWN_WEIGHTED_NUMERICAL,
                    "full_model_witness_valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "property_rows": [],
                }
            route_rows.append(route_row)
            if route_row.get("full_model_witness_valid") is True:
                break
    status, reason = aggregate_route_sets(
        route_rows,
        enumeration_exact=route_sets.exact,
    )
    witness = next(
        (
            row
            for row in route_rows
            if row.get("full_model_witness_valid") is True
        ),
        None,
    )
    return {
        "sample_rank": rank,
        "dataset_index": index,
        "label": int(label),
        "clean_prediction": clean_prediction,
        "clean_topk_set": clean_set,
        "epsilon": epsilon,
        "status": status,
        "reason": reason,
        "route_semantics": "ANY_LEGAL_TOPK_UNORDERED_TIE_INCLUSIVE",
        "router_hz_exact": True,
        "exact_feasible_unordered_top3_sets": [
            list(values) for values in route_sets.feasible
        ],
        "exact_feasible_unordered_top3_set_count": len(route_sets.feasible),
        "route_set_enumeration_exact": route_sets.exact,
        "unresolved_route_sets": [list(values) for values in route_sets.unresolved],
        "gate_family": model.spec.gate.value,
        "gate_family_supported": normalized_gate_support(model.spec.gate)[0],
        "products_per_property": model.spec.top_k - 1,
        "route_sets": route_rows,
        "full_model_witness_valid": witness is not None,
        "counterexample_prediction": (
            witness.get("counterexample_prediction") if witness else None
        ),
        "counterexample_topk_set": (
            witness.get("counterexample_topk_set") if witness else None
        ),
        "witness_path": witness.get("witness_path") if witness else None,
        "witness_sha256": witness.get("witness_sha256") if witness else None,
        "total_seconds": time.monotonic() - started,
    }


def _load_cifar10(root: Path):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    return datasets.CIFAR10(
        root=str(root / "CIFAR10" / "raw"),
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fallback_properties = [
        property_row
        for row in rows
        for route in row.get("route_sets", [])
        for property_row in route.get("property_rows", [])
    ]
    return {
        "rows": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "all_unsafe_full_forward_validated": all(
            row["status"] != "UNSAFE"
            or row.get("full_model_witness_valid") is True
            for row in rows
        ),
        "route_sets": sum(
            int(row["exact_feasible_unordered_top3_set_count"]) for row in rows
        ),
        "fallback_property_rows": len(fallback_properties),
        "properties_use_exactly_two_products": bool(fallback_properties) and all(
            property_row.get("product_count") == 2
            for property_row in fallback_properties
        ),
        "two_product_path_observed": bool(fallback_properties),
    }


def independently_audit(output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read the flushed N2 artifacts and audit verdict semantics."""

    rows_path = output_dir / "results.jsonl"
    selection_path = output_dir / "sample_indices.json"
    raw_lines = rows_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(raw) for raw in raw_lines]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["rows"]
    issues: list[str] = []
    expected_ranks = [
        int(value) for value in config["cohort"]["clean_correct_ranks"]
    ]
    result_ranks = [int(row.get("sample_rank", -1)) for row in rows]
    selected_ranks = [int(row.get("sample_rank", -1)) for row in selection]
    if result_ranks != expected_ranks:
        issues.append("result_ranks_differ_from_frozen_cohort")
    if selected_ranks != expected_ranks:
        issues.append("selection_ranks_differ_from_frozen_cohort")
    if len(set(result_ranks)) != len(result_ranks):
        issues.append("duplicate_result_rank")
    for row in rows:
        rank = int(row.get("sample_rank", -1))
        if row.get("route_semantics") != "ANY_LEGAL_TOPK_UNORDERED_TIE_INCLUSIVE":
            issues.append(f"route_semantics:{rank}")
        if row.get("gate_family") != config["model"]["gate"]:
            issues.append(f"gate_family:{rank}")
        if int(row.get("products_per_property", -1)) != 2:
            issues.append(f"declared_product_count:{rank}")
        if row.get("status") == "UNSAFE" and row.get(
            "full_model_witness_valid"
        ) is not True:
            issues.append(f"unreplayed_unsafe:{rank}")
        if row.get("status") == "SAFE":
            if row.get("route_set_enumeration_exact") is not True:
                issues.append(f"safe_without_exact_route_sets:{rank}")
            if not row.get("route_sets") or not all(
                route.get("status") == "SAFE" for route in row["route_sets"]
            ):
                issues.append(f"safe_without_all_routes_safe:{rank}")
        for route in row.get("route_sets", []):
            if len(route.get("route_set", [])) != 3:
                issues.append(f"non_top3_route:{rank}")
            for property_row in route.get("property_rows", []):
                if int(property_row.get("product_count", -1)) != 2:
                    issues.append(f"fallback_product_count:{rank}")
                if property_row.get("status") == "UNSAFE" and property_row.get(
                    "full_model_witness_valid"
                ) is not True:
                    issues.append(f"unreplayed_property_unsafe:{rank}")
    recomputed = _summary(rows)
    return {
        "schema_version": 1,
        "result_jsonl_sha256": _sha256(rows_path),
        "result_line_sha256": [_sha256_bytes(raw) for raw in raw_lines],
        "selection_sha256": _sha256(selection_path),
        "recomputed_summary": recomputed,
        "issues": issues,
        "issue_count": len(issues),
        "passed": not issues,
    }


def run(args) -> dict[str, Any]:
    config_path = _inside(Path(args.config), PROJECT_ROOT)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    dataset_root = _inside(Path(config["dataset_root"]), WRITE_ROOT)
    if _git_value("branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("Experiment 1N2 requires feat/moe-route-verification")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("Experiment 1N2 requires a clean feature-branch worktree")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("Experiment 1N2 requires the act-py312 conda environment")
    configured_torchvision = Path(get_torchvision_data_root()).resolve()
    if configured_torchvision != dataset_root.resolve():
        raise RuntimeError("configured torchvision root differs from frozen N2 root")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"frozen N2 checkpoint is absent; training was not started: {checkpoint}"
        )
    if _sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("frozen N2 checkpoint hash changed")
    if output_dir.exists():
        raise RuntimeError(f"Experiment 1N2 refuses to overwrite {output_dir}")

    selection_device = torch.device(str(config["cohort"]["selection_device"]))
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    contract = validate_checkpoint_contract(payload, config)
    dataset = _load_cifar10(dataset_root)
    selection_model, _ = load_output_moe_checkpoint(
        checkpoint, map_location=selection_device
    )
    selection_model = prepare_selection_model(selection_model, selection_device)
    selection = select_clean_correct_indices(
        selection_model,
        dataset,
        config["cohort"]["clean_correct_ranks"],
        device=selection_device,
        batch_size=int(config["cohort"]["selection_batch_size"]),
    )
    del selection_model
    if selection_device.type == "cuda":
        torch.cuda.empty_cache()
    model.double().eval()

    output_dir.mkdir(parents=True)
    paths = {
        "config": output_dir / "runtime_config.json",
        "selection": output_dir / "sample_indices.json",
        "rows": output_dir / "results.jsonl",
        "summary": output_dir / "summary.json",
        "audit": output_dir / "independent_audit.json",
        "manifest": output_dir / "manifest.json",
    }
    runtime = {
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "git_head": _git_value("rev-parse", "HEAD"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_contract": contract,
        "dataset_root": str(dataset_root),
        "route_semantics": "ANY_LEGAL_TOPK_UNORDERED_TIE_INCLUSIVE",
        "status_semantics": {
            "SAFE": "all properties on all exact feasible route sets proved positive",
            "UNKNOWN": "incomplete enumeration, solver limit, numerical issue, or non-positive relaxation candidate",
            "UNSAFE": "only a concrete input replayed through the full MoE changes the clean prediction",
        },
        "config": config,
    }
    _write_json(paths["config"], runtime)
    _write_json(
        paths["selection"],
        {
            "selection_rule": "deterministic clean-correct ranks",
            "rows": selection,
        },
    )
    rows: list[dict[str, Any]] = []
    with paths["rows"].open("x", encoding="utf-8") as handle:
        for item in selection:
            row = run_sample(
                model=model,
                dataset=dataset,
                selection=item,
                output_dir=output_dir,
                config=config,
            )
            rows.append(row)
            _append_json_line(handle, row)
    summary = _summary(rows)
    _write_json(paths["summary"], summary)
    audit = independently_audit(output_dir, config)
    _write_json(paths["audit"], audit)
    if not audit["passed"]:
        raise RuntimeError(
            f"N2 independent audit failed with {audit['issue_count']} issues"
        )
    manifest = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != paths["manifest"]
    }
    manifest["manifest_payload_sha256"] = _sha256_bytes(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    )
    _write_json(paths["manifest"], manifest)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "independent_audit": audit,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
