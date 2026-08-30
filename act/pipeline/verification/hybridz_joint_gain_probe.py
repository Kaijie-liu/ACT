"""One-instance diagnostic for HybridZ-guided joint-gain BaB branching.

The probe deliberately bypasses dataset manifests and ground-truth labels.  It
loads exactly one ONNX/VNNLIB pair, runs the ordinary sound DualSolver BaB
verdict path, and records search telemetry for the optional layer-diverse
joint-group selector.  The selector has no proof authority; all selected ReLU
groups are expanded into their complete phase partitions by the verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    return str(value)


def _normalized_result_status(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value).strip().lower()


def _safe_promotion_route(
    status: object,
    *,
    property_separable_bab: bool,
) -> tuple[str, bool]:
    """Return the SAFE-promotion disposition without touching a validator."""

    normalized = _normalized_result_status(status)
    if not property_separable_bab:
        return "disabled", False
    if normalized == "certified":
        return "pending_certified_validation", True
    if normalized == "falsified":
        return "not_applicable_falsified", False
    if normalized == "unknown":
        return "not_applicable_unknown", False
    safe_label = normalized.replace(" ", "_") or "unavailable"
    return f"not_applicable_{safe_label}", False


def _discard_property_forest_live_capability(
    metadata: dict[str, Any],
) -> bool:
    """Remove an inapplicable or failed SAFE-only live capability."""

    return (
        metadata.pop("_property_forest_live_capability", None)
        is not None
    )


def _probe_authority_fields(
    status: object,
    *,
    safe_proof_receipt: Mapping[str, Any] | None,
    counterexample_receipt: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Derive conclusion authority from the matching verdict path only."""

    normalized = _normalized_result_status(status)
    safe_authority = bool(
        normalized == "certified"
        and safe_proof_receipt is not None
        and safe_proof_receipt.get("proof_authority") is True
    )
    counterexample_authority = bool(
        normalized == "falsified"
        and counterexample_receipt is not None
        and counterexample_receipt.get("proof_authority") is True
        and (
            counterexample_receipt.get("strict_replay") or {}
        ).get("valid_counterexample")
        is True
    )
    return {
        "proof_authority": bool(
            safe_authority or counterexample_authority
        ),
        "proof_authority_scope": (
            "property_forest_safe_live_run"
            if safe_authority
            else "strict_counterexample_replay"
            if counterexample_authority
            else "diagnostic_only"
        ),
        "safe_proof_authority": safe_authority,
        "counterexample_proof_authority": (
            counterexample_authority
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from act.back_end.bab.bab import (
        _check_input_specs_batched,
        _forward_for_violation_check,
        check_violations_batched,
        verify_bab_batched,
    )
    from act.back_end.bab.property_forest_authority import (
        new_property_forest_run_token,
        source_file_digests,
        validate_bab_safe_capability,
    )
    from act.back_end.config import build_vnncomp_bab_config
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.back_end.verifier import (
        gather_input_spec_layers,
        get_assert_layer,
    )
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.front_end.vnnlib_loader.create_specs import (
        create_specs_from_paths,
    )
    from act.pipeline.verification.torch2act import TorchToACT
    from act.pipeline.verification.strict_replay import make_strict_replay
    from act.util.device_manager import initialize_device

    # Hash before the first model/spec read.  A second hash after ACT
    # conversion closes the load-time TOCTOU window; the same snapshot is
    # sealed into the live verifier capability.
    source_paths = {
        "onnx": args.onnx,
        "vnnlib": args.vnnlib,
    }
    source_digests_before_run = source_file_digests(source_paths)
    initialize_device(args.device, args.dtype)
    started = time.time()
    spec_result = create_specs_from_paths(str(args.onnx), str(args.vnnlib))
    wrapped = list(synthesize_models_from_specs([spec_result]).values())
    if len(wrapped) != 1:
        raise ValueError(
            "joint-gain probe requires exactly one synthesized disjunct; "
            f"observed {len(wrapped)}"
        )
    net = TorchToACT(wrapped[0]).run()
    if source_file_digests(source_paths) != source_digests_before_run:
        raise RuntimeError(
            "ONNX/VNNLIB source changed while loading or converting"
        )
    config = build_vnncomp_bab_config(
        "gain",
        multi_split_levels=args.multi_split_levels,
        dual_n_iters=args.dual_iters,
        solver_tier="dual_alpha_eta",
    )
    config.joint_gain_groups = args.joint_gain_groups
    config.property_branch_focus = args.property_branch_focus
    config.property_separable_bab = args.property_separable_bab
    config.branch_requires_unstable_successor = (
        args.branch_requires_unstable_successor
    )
    config.frontier_contraction_target = args.frontier_contraction_target
    config.per_subproblem_refine = args.per_subproblem_refine
    config.per_subproblem_refine_rows_cap = (
        args.per_subproblem_refine_rows_cap
    )
    config.per_subproblem_refine_iters = args.per_subproblem_refine_iters
    config.per_subproblem_refine_layer_cap = (
        args.per_subproblem_refine_layer_cap
    )
    property_forest_run_token = (
        new_property_forest_run_token()
        if config.property_separable_bab
        else None
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    verdict_started = time.time()
    result = verify_bab_batched(
        net,
        solver_factory=TorchLPSolver,
        config=config,
        max_batch_size=args.max_batch_size,
        time_budget_s=args.time_budget,
        _property_forest_run_token=property_forest_run_token,
        _property_forest_source_digests=(
            source_digests_before_run
            if property_forest_run_token is not None
            else None
        ),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    completed = time.time()
    safe_proof_receipt: dict[str, Any] | None = None
    safe_promotion_errors: tuple[str, ...] = ()
    (
        safe_promotion_status,
        should_validate_safe,
    ) = _safe_promotion_route(
        result.status,
        property_separable_bab=bool(
            config.property_separable_bab
        ),
    )
    if should_validate_safe:
        if property_forest_run_token is None:
            safe_promotion_errors = ("missing_run_token",)
            safe_promotion_status = "rejected_missing_run_token"
        else:
            try:
                (
                    safe_proof_receipt,
                    safe_promotion_errors,
                ) = validate_bab_safe_capability(
                    result,
                    net=net,
                    solver_factory=TorchLPSolver,
                    config=config,
                    max_batch_size=args.max_batch_size,
                    time_budget_s=args.time_budget,
                    expected_dtype=args.dtype,
                    expected_device=args.device,
                    run_token=property_forest_run_token,
                    source_paths=source_paths,
                    source_digests_before_run=(
                        source_digests_before_run
                    ),
                )
                if (
                    safe_proof_receipt is not None
                    and safe_proof_receipt.get("proof_authority")
                    is True
                    and not safe_promotion_errors
                ):
                    safe_promotion_status = "validated_certified"
                else:
                    safe_promotion_status = "rejected_certified"
            except Exception as exc:
                _discard_property_forest_live_capability(
                    result.metadata
                )
                safe_proof_receipt = None
                safe_promotion_errors = (
                    f"validator_error:{type(exc).__name__}:{exc}",
                )
                safe_promotion_status = "validator_error_certified"
    else:
        _discard_property_forest_live_capability(result.metadata)
    counterexample_receipt: dict[str, Any] | None = None
    if result.counterexample is not None:
        candidate = (
            result.counterexample.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )
        candidate_batch = candidate.unsqueeze(0).to(
            device=next(
                (
                    value.device
                    for layer in net.layers
                    for value in layer.params.values()
                    if isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                ),
                torch.device(args.device),
            ),
            dtype=(
                torch.float64
                if args.dtype == "float64"
                else torch.float32
            ),
        )
        input_valid = _check_input_specs_batched(
            candidate_batch, gather_input_spec_layers(net)
        )
        assert_layer = get_assert_layer(net)
        raw_violation = check_violations_batched(
            net, candidate_batch, assert_layer
        )
        logits = _forward_for_violation_check(net, candidate_batch)
        replay_passed = bool(
            input_valid[0].item() and raw_violation[0].item()
        )
        if not replay_passed:
            raise RuntimeError(
                "FALSIFIED candidate failed independent full-spec replay"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        candidate_path = args.output.with_suffix(
            ".counterexample.npy"
        )
        np.save(
            candidate_path,
            candidate.numpy(),
            allow_pickle=False,
        )
        strict_replay = make_strict_replay(args.onnx, args.vnnlib)
        strict_receipt = strict_replay(candidate.numpy())
        if (
            strict_receipt.get("valid_counterexample") is not True
            or strict_receipt.get("model_sha256")
            != source_digests_before_run["onnx"]
            or strict_receipt.get("vnnlib_sha256")
            != source_digests_before_run["vnnlib"]
            or source_file_digests(source_paths)
            != source_digests_before_run
        ):
            raise RuntimeError(
                "FALSIFIED candidate failed independent ONNX Runtime + "
                "raw-VNNLIB replay or source binding"
            )
        strict_path = args.output.with_suffix(".strict_replay.json")
        strict_path.write_text(
            json.dumps(
                strict_receipt,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        logits_cpu = logits[0].detach().to(
            dtype=torch.float64, device="cpu"
        )
        counterexample_receipt = {
            "schema": "act.hybridz_bab_counterexample.v1",
            "proof_authority": True,
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "dtype": "float64",
            "shape": list(candidate.shape),
            "numel": int(candidate.numel()),
            "minimum": float(candidate.min().item()),
            "maximum": float(candidate.max().item()),
            "input_spec_valid": bool(input_valid[0].item()),
            "full_assert_violated": bool(raw_violation[0].item()),
            "replay_passed": replay_passed,
            "logits_sha256": hashlib.sha256(
                logits_cpu.numpy().tobytes(order="C")
            ).hexdigest(),
            "predicted_class": int(logits_cpu.argmax().item()),
            "strict_replay": {
                "authority": strict_receipt.get("authority"),
                "valid_counterexample": bool(
                    strict_receipt.get("valid_counterexample")
                ),
                "reason": strict_receipt.get("reason"),
                "tolerance": strict_receipt.get("tolerance"),
                "model_sha256": strict_receipt.get("model_sha256"),
                "vnnlib_sha256": strict_receipt.get(
                    "vnnlib_sha256"
                ),
                "input_sha256": (
                    strict_receipt.get("input") or {}
                ).get("actual_sha256"),
                "output_sha256": (
                    strict_receipt.get("output") or {}
                ).get("actual_sha256"),
                "receipt_path": str(strict_path),
                "receipt_sha256": _sha256(strict_path),
                "elapsed_seconds": strict_receipt.get(
                    "elapsed_seconds"
                ),
            },
        }
        if assert_layer.params.get("kind") == "TOP1_ROBUST":
            true_class = int(
                torch.as_tensor(
                    assert_layer.params["y_true"]
                ).reshape(-1)[0].item()
            )
            other = logits_cpu.clone()
            other[true_class] = -float("inf")
            rival = int(other.argmax().item())
            counterexample_receipt.update(
                {
                    "true_class": true_class,
                    "violating_rival": rival,
                    "top1_violation_margin": float(
                        logits_cpu[rival] - logits_cpu[true_class]
                    ),
                    "true_logit": float(logits_cpu[true_class]),
                    "rival_logit": float(logits_cpu[rival]),
                }
            )
    authority_fields = _probe_authority_fields(
        result.status,
        safe_proof_receipt=safe_proof_receipt,
        counterexample_receipt=counterexample_receipt,
    )
    return {
        "schema": "act.hybridz_joint_gain_probe.v2",
        **authority_fields,
        "property_forest_safe_promotion_status": (
            safe_promotion_status
        ),
        "property_forest_safe_proof": safe_proof_receipt,
        "property_forest_safe_promotion_errors": list(
            safe_promotion_errors
        ),
        "ground_truth_loaded": False,
        "verdict_source": "ordinary_sound_dual_bab",
        "inputs": {
            "onnx": str(args.onnx),
            "onnx_sha256": _sha256(args.onnx),
            "vnnlib": str(args.vnnlib),
            "vnnlib_sha256": _sha256(args.vnnlib),
        },
        "config": {
            "solver_tier": config.solver_tier,
            "dual_iters": int(config.dual_n_iters),
            "branching_method": config.branching_method,
            "multi_split_levels": int(config.multi_split_levels),
            "joint_gain_groups": int(config.joint_gain_groups),
            "property_branch_focus": config.property_branch_focus,
            "property_separable_bab": bool(
                config.property_separable_bab
            ),
            "branch_requires_unstable_successor": bool(
                config.branch_requires_unstable_successor
            ),
            "frontier_contraction_target": float(
                config.frontier_contraction_target
            ),
            "per_subproblem_refine": config.per_subproblem_refine,
            "per_subproblem_refine_rows_cap": int(
                config.per_subproblem_refine_rows_cap
            ),
            "per_subproblem_refine_iters": int(
                config.per_subproblem_refine_iters
            ),
            "per_subproblem_refine_layer_cap": int(
                config.per_subproblem_refine_layer_cap
            ),
            "max_batch_size": args.max_batch_size,
            "time_budget_seconds": float(args.time_budget),
            "device": args.device,
            "dtype": args.dtype,
        },
        "result": {
            "status": _json_value(result.status),
            "metadata": _json_value(result.metadata),
        },
        "counterexample": counterexample_receipt,
        "timing": {
            "setup_seconds": verdict_started - started,
            "verdict_seconds": completed - verdict_started,
            "total_seconds": completed - started,
        },
        "cuda_peak": {
            "available": bool(torch.cuda.is_available()),
            "allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("vnnlib", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--joint-gain-groups", type=int, default=1)
    parser.add_argument(
        "--property-branch-focus",
        choices=("sum", "worst"),
        default="sum",
    )
    parser.add_argument(
        "--property-separable-bab",
        action="store_true",
    )
    parser.add_argument(
        "--branch-requires-unstable-successor",
        action="store_true",
    )
    parser.add_argument(
        "--frontier-contraction-target",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--per-subproblem-refine",
        choices=("none", "tail", "all", "split_successors"),
        default="none",
    )
    parser.add_argument(
        "--per-subproblem-refine-rows-cap",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--per-subproblem-refine-iters",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--per-subproblem-refine-layer-cap",
        type=int,
        default=2,
    )
    parser.add_argument("--multi-split-levels", type=int, default=3)
    parser.add_argument("--dual-iters", type=int, default=40)
    parser.add_argument("--time-budget", type=float, default=50.0)
    parser.add_argument("--max-batch-size", default="auto")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    args = parser.parse_args()
    if args.joint_gain_groups < 1:
        parser.error("--joint-gain-groups must be positive")
    if args.multi_split_levels < 1:
        parser.error("--multi-split-levels must be positive")
    if args.dual_iters < 1:
        parser.error("--dual-iters must be positive")
    if args.time_budget <= 0:
        parser.error("--time-budget must be positive")
    if not 0.0 <= args.frontier_contraction_target <= 1.0:
        parser.error("--frontier-contraction-target must be in [0, 1]")
    if args.per_subproblem_refine_rows_cap < 1:
        parser.error("--per-subproblem-refine-rows-cap must be positive")
    if args.per_subproblem_refine_iters < 0:
        parser.error("--per-subproblem-refine-iters must be nonnegative")
    if args.per_subproblem_refine_layer_cap < 1:
        parser.error("--per-subproblem-refine-layer-cap must be positive")
    if args.max_batch_size != "auto":
        args.max_batch_size = int(args.max_batch_size)
        if args.max_batch_size < 1:
            parser.error("--max-batch-size must be positive or 'auto'")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = run_probe(args)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
