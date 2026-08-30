"""Prepare and evaluate the frozen ICML-2025 RT-ER B3 comparison.

The ACT-side prepare stage performs only exact affine-route analysis, guarded
coordinate-hull construction, and the paper-formula reimplementation.  It
serializes immutable branch boxes for a separate CROWN-environment worker; it
does not claim that a positive-margin CROWN filter is outward-rounded SAFE.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import typing
from typing import Any, Sequence

from typing_extensions import override

if not hasattr(typing, "override"):
    typing.override = override  # type: ignore[attr-defined]

import numpy as np
import torch
from torch import nn

from act.back_end.core import Bounds
from act.back_end.moe import guarded_hz_box_hull_highs, fold_affine_input_map
from act.back_end.solver.solver_hz import (
    hz_add_output_inequalities,
    hz_check_feasibility,
    sparse_hz_from_bounds,
)
from act.pipeline.moe.certificate_constants import (
    ConstantProvider,
    ConstantStatus,
    OutputReading,
    RouterReading,
    ScalarConstant,
    Theorem54Constants,
    author_unspecified_constants,
    evaluate_theorem54_paper_formula,
    hard_argmax_router_constants,
    official_cifar_resnet18_logit_bounds,
    raw_logit_output_upper_unspecified,
    sound_probability_expert_constant,
    sound_softmax_router_constants,
)
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_route_telemetry import (
    CIFAR_MEAN_255,
    CIFAR_STD_255,
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _load_official_model,
    fold_official_router,
)
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/icml2025_b3_seed0.json"
CROWN_PYTHON = Path("/data1/Kane/MOE/envs/alpha-beta-crown/bin/python")
CROWN_WORKER = PROJECT_ROOT / "act/pipeline/moe/icml2025_b3_crown.py"


class PixelNormalizedExpert(nn.Module):
    """Expose an official normalized-input expert over unit pixel inputs."""

    def __init__(self, expert: nn.Module) -> None:
        super().__init__()
        self.expert = expert
        self.register_buffer(
            "mean", torch.as_tensor(CIFAR_MEAN_255, dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.as_tensor(CIFAR_STD_255, dtype=torch.float32)[None, :, None, None]
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.expert((pixels * 255.0 - self.mean) / self.std)


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _nvidia_driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    versions = sorted(
        {row.strip() for row in completed.stdout.splitlines() if row.strip()}
    )
    if not versions:
        raise RuntimeError("nvidia-smi returned no driver identity")
    return ",".join(versions)


def _package_version_or_unavailable(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def certificate_identity(
    config: dict[str, Any],
    checkpoint: Path,
    model: nn.Module,
    telemetry_artifact_path: Path,
) -> dict[str, Any]:
    """Materialize the runtime, preprocessing, and artifact certificate identity."""

    import torchvision

    archive = (
        Path(get_torchvision_data_root()).resolve()
        / "CIFAR10/raw/cifar-10-python.tar.gz"
    )
    if not archive.is_relative_to(WRITE_ROOT.resolve()) or not archive.is_file():
        raise RuntimeError("CIFAR-10 archive identity is unavailable inside write root")
    frozen = config["certificate_identity"]
    identity = {
        "python_version": platform.python_version(),
        "torch_import_version": torch.__version__,
        "torchvision_import_version": torchvision.__version__,
        "torchvision_metadata_version": importlib.metadata.version("torchvision"),
        "numpy_version": np.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "nvidia_driver_version": _nvidia_driver_version(),
        "official_source_commit": OFFICIAL_COMMIT,
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_archive_sha256": _sha256(archive),
        "ordered_input_identity": {
            "split": "official torchvision CIFAR10 test order",
            "telemetry_per_input_sha256": _sha256(telemetry_artifact_path),
        },
        "preprocessing_graph": frozen["preprocessing_graph"],
        "preprocessing_dtypes": frozen["preprocessing_dtypes"],
        "normalization_constants": {
            "mean_255": CIFAR_MEAN_255.tolist(),
            "std_255": CIFAR_STD_255.tolist(),
        },
        "input_domain": frozen["input_domain"],
        "router_sha256": _module_sha256(model.router),
        "solver_and_outward_rounding_policy": {
            "versions": {
                name: _package_version_or_unavailable(name)
                for name in ("scipy", "highspy", "gurobipy")
            },
            "routing": config["routing"],
            "numerical": config["numerical"],
        },
    }
    missing = sorted(set(frozen["required_manifest_fields"]) - set(identity))
    if missing:
        raise RuntimeError(f"certificate identity is incomplete: {missing}")
    return identity


def audit_router_optimizer_state(
    model: nn.Module,
    payload: dict[str, Any],
    *,
    reference_model: nn.Module | None = None,
) -> dict[str, Any]:
    """Link checkpoint optimizer slots to named parameters in construction order."""

    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("official checkpoint lacks optimizer state")
    parameter_ids = [
        int(parameter_id)
        for group in optimizer.get("param_groups", [])
        for parameter_id in group.get("params", [])
    ]
    named_parameters = list(model.named_parameters())
    if len(parameter_ids) != len(named_parameters):
        raise RuntimeError(
            "optimizer parameter order cannot be linked to official named parameters"
        )
    state = optimizer.get("state", {})
    rows: list[dict[str, Any]] = []
    expert_state_entries = 0
    for (name, _parameter), parameter_id in zip(named_parameters, parameter_ids):
        slot = state.get(parameter_id, {})
        if name.startswith("experts.") and slot:
            expert_state_entries += 1
        if name.startswith("router."):
            step = slot.get("step")
            if isinstance(step, torch.Tensor):
                step = float(step.item())
            elif step is not None:
                step = float(step)
            rows.append(
                {
                    "name": name,
                    "optimizer_parameter_id": parameter_id,
                    "optimizer_state_keys": sorted(str(key) for key in slot),
                    "step": step,
                }
            )
    if not rows:
        raise RuntimeError("official model has no named router parameters")
    result: dict[str, Any] = {
        "optimizer_parameters": len(parameter_ids),
        "optimizer_state_entries": len(state),
        "expert_parameters_with_optimizer_state": expert_state_entries,
        "router_parameters": rows,
        "router_parameters_with_optimizer_state": sum(
            bool(row["optimizer_state_keys"]) for row in rows
        ),
        "router_sha256": _module_sha256(model.router),
    }
    if reference_model is not None:
        reference_state = reference_model.router.state_dict()
        current_state = model.router.state_dict()
        if reference_state.keys() != current_state.keys():
            raise RuntimeError("reference and final router state layouts differ")
        result.update(
            {
                "reference_router_sha256": _module_sha256(reference_model.router),
                "router_equal_reference": all(
                    torch.equal(current_state[name], reference_state[name])
                    for name in current_state
                ),
            }
        )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(path)


def select_boundary_cohort(
    clean_correct: np.ndarray,
    radius_uppers: np.ndarray,
    *,
    samples: int,
    multiplier: float,
    cap: float,
) -> np.ndarray:
    clean_correct = np.asarray(clean_correct, dtype=np.bool_).reshape(-1)
    radius_uppers = np.asarray(radius_uppers, dtype=np.float64).reshape(-1)
    if clean_correct.shape != radius_uppers.shape:
        raise ValueError("clean-correct and radius arrays differ")
    eligible = np.flatnonzero(
        clean_correct
        & np.isfinite(radius_uppers)
        & (radius_uppers >= 0.0)
        & (float(multiplier) * radius_uppers <= float(cap))
    )
    if eligible.size < int(samples):
        raise RuntimeError(
            f"only {eligible.size} clean-correct route-boundary inputs satisfy the frozen cap"
        )
    return eligible[: int(samples)].astype(np.int64)


def route_applicability_census(
    radius_lowers: np.ndarray,
    radius_uppers: np.ndarray,
    clean_correct: np.ndarray,
    epsilon_over_255: Sequence[float],
) -> dict[str, Any]:
    radius_lowers = np.asarray(radius_lowers, dtype=np.float64).reshape(-1)
    radius_uppers = np.asarray(radius_uppers, dtype=np.float64).reshape(-1)
    clean_correct = np.asarray(clean_correct, dtype=np.bool_).reshape(-1)
    if not (
        radius_lowers.shape == radius_uppers.shape == clean_correct.shape
    ):
        raise ValueError("route-applicability arrays differ")
    result: dict[str, Any] = {}
    for numerator in epsilon_over_255:
        epsilon = float(numerator) / 255.0
        stable = epsilon < radius_lowers
        reachable = radius_uppers <= epsilon
        undecided = ~(stable | reachable)
        result[str(numerator)] = {
            "epsilon": epsilon,
            "all_samples": {
                "denominator": int(stable.size),
                "route_stable": int(stable.sum()),
                "route_unstable": int(reachable.sum()),
                "numerically_undecided": int(undecided.sum()),
            },
            "clean_correct_samples": {
                "denominator": int(clean_correct.sum()),
                "route_stable": int((stable & clean_correct).sum()),
                "route_unstable": int((reachable & clean_correct).sum()),
                "numerically_undecided": int((undecided & clean_correct).sum()),
            },
        }
    return result


def top1_guard(
    weight: np.ndarray,
    bias: np.ndarray,
    expert: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tie-inclusive ``r_j-r_i <= 0`` rows for one legal expert."""

    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64).reshape(-1)
    expert = int(expert)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("router weight/bias shape mismatch")
    if not 0 <= expert < weight.shape[0]:
        raise IndexError("expert is outside router width")
    competitors = [value for value in range(weight.shape[0]) if value != expert]
    rows = np.stack([weight[value] - weight[expert] for value in competitors])
    rhs = np.asarray([bias[expert] - bias[value] for value in competitors])
    return rows, rhs


def formula_leaf(
    radius: float | None,
    status: ConstantStatus,
    *,
    route_lower: float,
    route_upper: float,
    smallest_registered_radius: float,
) -> str:
    if status in {
        ConstantStatus.NOT_FORMALLY_INSTANTIATED,
        ConstantStatus.DIAGNOSTIC_ONLY,
    }:
        return "L1_NOT_FORMALLY_INSTANTIATED"
    if status == ConstantStatus.NOT_APPLICABLE or radius is None:
        return "L1_NOT_APPLICABLE"
    radius = float(radius)
    if radius < float(smallest_registered_radius):
        return "L2_VACUOUS_AT_REGISTERED_RADII"
    if radius < float(route_lower):
        return "L3_HARD_ROUTE_APPLICABILITY_ESTABLISHED"
    if radius >= float(route_upper):
        return "L4_ASSUMPTION_NOT_ESTABLISHED"
    return "UNDECIDED_NUMERICAL_ROUTE_BRACKET_OVERLAP"


def _scalar(value: float, quantity: str, detail: str) -> ScalarConstant:
    return ScalarConstant(
        float(value),
        ConstantProvider.SOUND_GLOBAL_SPECTRAL,
        ConstantStatus.FORMAL_BOUND,
        quantity,
        detail,
    )


def official_constant_families(model, folded_router_weight: np.ndarray) -> dict[str, Any]:
    """Build both theorem readings without silently changing model semantics."""

    normalization_scale = float(np.max(255.0 / CIFAR_STD_255))
    router_logit_lipschitz = float(
        np.max(np.sum(np.abs(folded_router_weight), axis=1))
    )
    vector_bounds: list[float] = []
    row_bounds: list[list[float]] = []
    probability_lipschitz: list[ScalarConstant] = []
    probability_upper: list[ScalarConstant] = []
    raw_lipschitz: list[ScalarConstant] = []
    raw_upper: list[ScalarConstant] = []
    for expert_index, expert in enumerate(model.experts):
        vector, rows = official_cifar_resnet18_logit_bounds(expert.net)
        vector *= normalization_scale
        rows = tuple(value * normalization_scale for value in rows)
        vector_bounds.append(vector)
        row_bounds.append(list(rows))
        probability, upper = sound_probability_expert_constant(
            vector, expert_index=expert_index
        )
        probability_lipschitz.append(probability)
        probability_upper.append(upper)
        raw_lipschitz.append(
            _scalar(
                vector,
                f"L_R[{expert_index}]",
                "global pixel-domain official ResNet logit-vector bound",
            )
        )
        raw_upper.append(raw_logit_output_upper_unspecified(expert_index=expert_index))

    continuous = Theorem54Constants(
        sound_softmax_router_constants(
            router_logit_lipschitz, num_experts=len(model.experts)
        ),
        tuple(probability_lipschitz),
        tuple(probability_upper),
        OutputReading.PROBABILITY,
        RouterReading.CONTINUOUS_SOFTMAX,
    )
    hard_raw = Theorem54Constants(
        hard_argmax_router_constants(num_experts=len(model.experts)),
        tuple(raw_lipschitz),
        tuple(raw_upper),
        OutputReading.RAW_LOGIT,
        RouterReading.HARD_ARGMAX,
    )
    unspecified = Theorem54Constants(
        author_unspecified_constants(
            num_experts=len(model.experts), quantity_prefix="r_R"
        ),
        author_unspecified_constants(
            num_experts=len(model.experts), quantity_prefix="L_R"
        ),
        author_unspecified_constants(
            num_experts=len(model.experts), quantity_prefix="M_R"
        ),
        OutputReading.PROBABILITY,
        RouterReading.CONTINUOUS_SOFTMAX,
    )
    return {
        "continuous_probability": continuous,
        "released_hard_raw": hard_raw,
        "author_unspecified": unspecified,
        "manifest": {
            "normalization_linf_scale": normalization_scale,
            "router_logit_lipschitz": router_logit_lipschitz,
            "expert_logit_vector_lipschitz": vector_bounds,
            "expert_logit_row_lipschitz": row_bounds,
            "continuous_probability_semantics": (
                "formal constants for a continuous softmax-router/probability-expert "
                "reading; not the released hard-argmax/raw-logit program"
            ),
            "released_hard_raw_semantics": (
                "hard router is discontinuous at reachable ties and raw M_R is undisclosed"
            ),
        },
    }


def _constant_manifest(constants: Theorem54Constants) -> dict[str, Any]:
    return {
        "router_lipschitz": [asdict(item) for item in constants.router_lipschitz],
        "expert_lipschitz": [asdict(item) for item in constants.expert_lipschitz],
        "expert_output_upper": [asdict(item) for item in constants.expert_output_upper],
        "output_reading": constants.output_reading.value,
        "router_reading": constants.router_reading.value,
        "formal": constants.formal,
    }


def _paper_formula_rows(
    model,
    pixels: torch.Tensor,
    route_lowers: np.ndarray,
    route_uppers: np.ndarray,
    constants: dict[str, Any],
    *,
    smallest_registered_radius: float,
) -> list[dict[str, Any]]:
    mean = torch.as_tensor(CIFAR_MEAN_255, dtype=torch.float32)[None, :, None, None]
    std = torch.as_tensor(CIFAR_STD_255, dtype=torch.float32)[None, :, None, None]
    normalized = (pixels.float() * 255.0 - mean) / std
    with torch.no_grad():
        router_scores = model.router.gate(normalized.flatten(1))
        continuous_weights = torch.softmax(router_scores, dim=1)
        expert_logits = torch.stack(
            [expert(normalized) for expert in model.experts], dim=1
        )
        expert_probabilities = torch.softmax(expert_logits, dim=2)
        continuous_output = torch.sum(
            continuous_weights.unsqueeze(2) * expert_probabilities, dim=1
        )
        hard_routes = router_scores.argmax(dim=1)
        hard_output = expert_logits[
            torch.arange(expert_logits.shape[0]), hard_routes
        ]
    rows: list[dict[str, Any]] = []
    for slot in range(pixels.shape[0]):
        continuous_prediction = int(continuous_output[slot].argmax().item())
        hard_prediction = int(hard_output[slot].argmax().item())
        continuous = evaluate_theorem54_paper_formula(
            continuous_output[slot].tolist(),
            continuous_prediction,
            continuous_weights[slot].tolist(),
            constants["continuous_probability"],
        )
        one_hot = [0.0] * len(model.experts)
        one_hot[int(hard_routes[slot])] = 1.0
        hard = evaluate_theorem54_paper_formula(
            hard_output[slot].tolist(),
            hard_prediction,
            one_hot,
            constants["released_hard_raw"],
        )
        unspecified = evaluate_theorem54_paper_formula(
            continuous_output[slot].tolist(),
            continuous_prediction,
            continuous_weights[slot].tolist(),
            constants["author_unspecified"],
        )
        rows.append(
            {
                "continuous_probability": {
                    **asdict(continuous),
                    "leaf": formula_leaf(
                        continuous.radius,
                        continuous.status,
                        route_lower=float(route_lowers[slot]),
                        route_upper=float(route_uppers[slot]),
                        smallest_registered_radius=smallest_registered_radius,
                    ),
                    "semantic_scope": "continuous surrogate, not released hard dispatch",
                },
                "released_hard_raw": {
                    **asdict(hard),
                    "leaf": formula_leaf(
                        hard.radius,
                        hard.status,
                        route_lower=float(route_lowers[slot]),
                        route_upper=float(route_uppers[slot]),
                        smallest_registered_radius=smallest_registered_radius,
                    ),
                },
                "author_unspecified": {
                    **asdict(unspecified),
                    "leaf": formula_leaf(
                        unspecified.radius,
                        unspecified.status,
                        route_lower=float(route_lowers[slot]),
                        route_upper=float(route_uppers[slot]),
                        smallest_registered_radius=smallest_registered_radius,
                    ),
                },
                "continuous_prediction": continuous_prediction,
                "hard_prediction": hard_prediction,
                "hard_route": int(hard_routes[slot]),
            }
        )
    return rows


def _repo_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=OFFICIAL_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def prepare(config_path: Path, checkpoint: Path, telemetry_dir: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    checkpoint = _inside(checkpoint, WRITE_ROOT)
    telemetry_dir = _inside(telemetry_dir, WRITE_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"B3 prepare refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("B3 config is not frozen")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        "status", "--porcelain"
    ):
        raise RuntimeError("official repository identity/cleanliness gate failed")
    telemetry_summary_path = telemetry_dir / "summary.json"
    telemetry_artifact_path = telemetry_dir / "per_input.npz"
    telemetry = json.loads(telemetry_summary_path.read_text(encoding="utf-8"))
    if _sha256(checkpoint) != telemetry["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint differs from telemetry identity")
    if _sha256(telemetry_artifact_path) != telemetry["artifact"]["sha256"]:
        raise RuntimeError("per-input telemetry artifact changed")
    arrays = np.load(telemetry_artifact_path, allow_pickle=False)
    applicability = route_applicability_census(
        arrays["radius_lowers"],
        arrays["radius_uppers"],
        arrays["clean_correct"],
        config["applicability_epsilon_over_255"],
    )
    selection = config["selection"]
    multiplier = float(selection["route_radius_multiplier"])
    cap = float(selection["route_radius_cap_over_255"]) / 255.0
    indices = select_boundary_cohort(
        arrays["clean_correct"],
        arrays["radius_uppers"],
        samples=int(selection["samples"]),
        multiplier=multiplier,
        cap=cap,
    )
    route_lowers = arrays["radius_lowers"][indices].astype(np.float64)
    route_uppers = arrays["radius_uppers"][indices].astype(np.float64)
    epsilons = multiplier * route_uppers

    sys.dont_write_bytecode = True
    device = torch.device("cpu")
    model, payload = _load_official_model(checkpoint, device)
    checkpoint_epoch = int(payload.get("epoch", -1)) + 1
    expected_epoch = int(config["checkpoint"]["epoch"])
    if checkpoint_epoch != expected_epoch or int(telemetry.get("epoch", -1)) != expected_epoch:
        raise RuntimeError("B3 requires the frozen final epoch-130 checkpoint and telemetry")
    provenance = config["training_provenance"]
    reference_config = provenance["router_reference_checkpoint"]
    reference_checkpoint = _inside(Path(reference_config["path"]), WRITE_ROOT)
    if _sha256(reference_checkpoint) != reference_config["sha256"]:
        raise RuntimeError("B3 router-reference checkpoint changed")
    reference_model, reference_payload = _load_official_model(
        reference_checkpoint, device
    )
    if int(reference_payload.get("epoch", -1)) + 1 != int(
        reference_config["epoch"]
    ):
        raise RuntimeError("B3 router-reference epoch differs from config")
    optimizer_audit = audit_router_optimizer_state(
        model, payload, reference_model=reference_model
    )
    if provenance["require_zero_router_optimizer_state"] and int(
        optimizer_audit["router_parameters_with_optimizer_state"]
    ) != 0:
        raise RuntimeError("final router unexpectedly has optimizer state")
    if provenance["require_router_equal_reference"] and not bool(
        optimizer_audit["router_equal_reference"]
    ):
        raise RuntimeError("final router differs from frozen reference checkpoint")
    router = model.router.gate
    folded_weight, folded_bias = fold_official_router(
        router.weight.detach().cpu().double().numpy(),
        router.bias.detach().cpu().double().numpy(),
    )
    import torchvision.datasets as datasets

    dataset = datasets.CIFAR10(
        root=str(Path(get_torchvision_data_root()) / "CIFAR10/raw"),
        train=False,
        download=False,
    )
    raw = np.asarray(dataset.data, dtype=np.uint8)[indices]
    labels = np.asarray(dataset.targets, dtype=np.int64)[indices]
    centers = raw.transpose(0, 3, 1, 2).astype(np.float64) / 255.0
    lower = np.maximum(0.0, centers - epsilons[:, None, None, None])
    upper = np.minimum(1.0, centers + epsilons[:, None, None, None])
    constants = official_constant_families(model, folded_weight)
    formula_rows = _paper_formula_rows(
        model,
        torch.from_numpy(centers),
        route_lowers,
        route_uppers,
        constants,
        smallest_registered_radius=float(config["formula"]["smallest_radius_over_255"])
        / 255.0,
    )
    telemetry_predictions = arrays["predictions"][indices].astype(np.int64)
    telemetry_routes = arrays["clean_experts"][indices].astype(np.int64)
    telemetry_labels = arrays["labels"][indices].astype(np.int64)
    if not np.array_equal(labels, telemetry_labels):
        raise RuntimeError("selected dataset labels differ from telemetry")
    for slot, row in enumerate(formula_rows):
        if int(row["hard_prediction"]) != int(telemetry_predictions[slot]):
            raise RuntimeError("hard prediction differs from frozen telemetry")
        if int(row["hard_route"]) != int(telemetry_routes[slot]):
            raise RuntimeError("hard route differs from frozen telemetry")
        if int(row["hard_prediction"]) != int(labels[slot]):
            raise RuntimeError("selected B3 sample is no longer clean-correct")
        if float(epsilons[slot]) < float(route_uppers[slot]):
            raise RuntimeError("adaptive B3 radius is not proven route-unstable")

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "cohort.npz"
    artifact_hash = _save_npz(
        artifact_path,
        dataset_indices=indices,
        labels=labels,
        centers=centers,
        lower=lower,
        upper=upper,
        route_lowers=route_lowers,
        route_uppers=route_uppers,
        epsilons=epsilons,
    )
    hull_dir = output_dir / "hulls"
    hull_dir.mkdir()
    branches_path = output_dir / "branches.jsonl"
    branches: list[dict[str, Any]] = []
    with branches_path.open("x", encoding="utf-8") as branches_handle:
        for slot, dataset_index in enumerate(indices.tolist()):
            input_hz = sparse_hz_from_bounds(
                Bounds(
                    torch.from_numpy(lower[slot].reshape(1, -1)),
                    torch.from_numpy(upper[slot].reshape(1, -1)),
                ),
                frame_id=10_000_000 + int(dataset_index),
            )
            candidates: list[int] = []
            unresolved: list[int] = []
            for expert in range(folded_weight.shape[0]):
                guard_matrix, guard_rhs = top1_guard(folded_weight, folded_bias, expert)
                guarded = hz_add_output_inequalities(
                    input_hz,
                    torch.from_numpy(guard_matrix),
                    torch.from_numpy(guard_rhs),
                )
                feasibility = hz_check_feasibility(
                    guarded,
                    time_limit=float(config["routing"]["feasibility_time_limit_seconds"]),
                )
                record: dict[str, Any] = {
                    "sample_slot": slot,
                    "dataset_index": int(dataset_index),
                    "expert": expert,
                    "feasibility": feasibility.status,
                    "feasibility_seconds": feasibility.elapsed,
                    "feasibility_nodes": feasibility.nodes,
                    "guard_semantics": "tie-inclusive r_j-r_i <= 0 for every j != i",
                }
                if feasibility.status != "infeasible":
                    candidates.append(expert)
                    unresolved.extend([expert] if feasibility.status == "unknown" else [])
                    hull = guarded_hz_box_hull_highs(
                        guarded,
                        time_limit=float(config["routing"]["hull_time_limit_seconds"]),
                    )
                    hull_path = (
                        hull_dir / f"sample_{slot:03d}_expert_{expert}.npz"
                    )
                    hull_hash = _save_npz(
                        hull_path,
                        lower=hull.bounds.lb.numpy().reshape(3, 32, 32),
                        upper=hull.bounds.ub.numpy().reshape(3, 32, 32),
                    )
                    record.update(
                        {
                            "hull_artifact": str(hull_path),
                            "hull_artifact_sha256": hull_hash,
                            "hull_complete": hull.complete,
                            "hull_exact": hull.exact,
                            "hull_domain_status": hull.domain_status,
                            "hull_telemetry": hull.telemetry.as_dict(),
                        }
                    )
                branches.append(record)
                _append_json(branches_handle, record)
            formula_rows[slot].update(
                {
                    "sample_rank": slot,
                    "dataset_index": int(dataset_index),
                    "label": int(labels[slot]),
                    "route_radius_lower": float(route_lowers[slot]),
                    "route_radius_upper": float(route_uppers[slot]),
                    "epsilon": float(epsilons[slot]),
                    "route_invariance_baseline": "UNKNOWN_ROUTE_UNSTABLE",
                    "candidate_experts": candidates,
                    "candidate_feasibility_unresolved": unresolved,
                    "candidate_set_exact": not unresolved,
                }
            )
    result = {
        "schema_version": 1,
        "status": "PREPARED_CROWN_NOT_RUN",
        "label": telemetry["label"],
        "official_source": {"commit": OFFICIAL_COMMIT, "clone_clean": True},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "epoch": checkpoint_epoch,
        },
        "certificate_identity": certificate_identity(
            config, checkpoint, model, telemetry_artifact_path
        ),
        "training_provenance": {
            "router_reference_checkpoint": {
                **reference_config,
                "observed_epoch": int(reference_payload.get("epoch", -1)) + 1,
            },
            "optimizer_audit": optimizer_audit,
            "interpretation": (
                "released hard-dispatch training updates experts but leaves the router "
                "at its frozen initialization; B3 must not label it a learned router"
            ),
        },
        "telemetry": {
            "directory": str(telemetry_dir),
            "summary_sha256": _sha256(telemetry_summary_path),
            "per_input_sha256": _sha256(telemetry_artifact_path),
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path), "value": config},
        "selection": {
            "dataset_indices": indices.tolist(),
            "samples": len(indices),
            "policy": selection,
        },
        "route_applicability_census": applicability,
        "constant_families": {
            **constants["manifest"],
            "continuous_probability": _constant_manifest(
                constants["continuous_probability"]
            ),
            "released_hard_raw": _constant_manifest(constants["released_hard_raw"]),
            "author_unspecified": _constant_manifest(constants["author_unspecified"]),
        },
        "rows": formula_rows,
        "branches": branches,
        "artifact": {"path": str(artifact_path), "sha256": artifact_hash},
        "branches_artifact": {
            "path": str(branches_path),
            "sha256": _sha256(branches_path),
            "rows": len(branches),
            "flush_policy": "one fsync after every expert branch",
        },
        "status_semantics": {
            "crown_positive_margin": "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
            "crown_negative_bound": "UNKNOWN_NEVER_UNSAFE",
            "unsafe": "requires concrete full hard-dispatch replay",
        },
    }
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official repository became dirty during B3 prepare")
    _write_json(output_dir / "prepare.json", result)
    return result


def run_crown_worker(
    prepare_path: Path,
    output_path: Path,
    *,
    crown_python: Path = CROWN_PYTHON,
) -> int:
    prepare_path = _inside(prepare_path, WRITE_ROOT)
    output_path = _inside(output_path, WRITE_ROOT)
    crown_python = _inside(crown_python, WRITE_ROOT)
    if output_path.exists():
        raise RuntimeError(f"CROWN output already exists: {output_path}")
    command = [
        str(crown_python),
        str(CROWN_WORKER),
        "--prepare",
        str(prepare_path),
        "--output",
        str(output_path),
    ]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": (
            str(PROJECT_ROOT)
            + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
        ),
    }
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--telemetry-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    crown_parser = subparsers.add_parser("crown")
    crown_parser.add_argument("--prepare", type=Path, required=True)
    crown_parser.add_argument("--output", type=Path, required=True)
    crown_parser.add_argument("--crown-python", type=Path, default=CROWN_PYTHON)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.config, args.checkpoint, args.telemetry_dir, args.output_dir)
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        raise SystemExit(
            run_crown_worker(args.prepare, args.output, crown_python=args.crown_python)
        )


if __name__ == "__main__":
    main()
