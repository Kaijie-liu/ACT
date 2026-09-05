from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_init_bn_semantics import _forward_semantics, summarize_scores
from act.pipeline.moe.advmoe_router_bracket import (
    clean_margin_diagnostics,
    load_cifar10_test_archive,
    strong_pgd_route_flip,
)
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _resolve_recorded_path(path: str, repository: Path) -> Path:
    """Resolve an audit-recorded path without weakening its workspace gate."""
    recorded = Path(path)
    if not recorded.is_absolute():
        recorded = repository / recorded
    return recorded.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distribution requires a nonempty finite array")
    return {
        "minimum": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std(ddof=0)),
    }


def _load_state(
    spec: dict[str, Any], device: torch.device, workspace: Path
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    model, router, _moe_type = construct_official_init(int(spec.get("seed", 0)))
    identity: dict[str, Any] = {"label": spec["label"], "kind": spec["kind"]}
    if spec["kind"] == "CHECKPOINT":
        path = _inside(Path(spec["path"]), workspace)
        actual_hash = _sha256(path)
        if actual_hash != spec["sha256"]:
            raise RuntimeError(f"{spec['label']}: checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != int(spec["epoch"]):
            raise RuntimeError(f"{spec['label']}: checkpoint epoch mismatch")
        model.load_state_dict(payload["state_dict"])
        router.load_state_dict(payload["router"])
        identity.update(
            {
                "epoch": int(payload["epoch"]),
                "path": str(path),
                "sha256": actual_hash,
                "released_best_ra_percent": float(payload["best_acc"]),
                "released_best_sa_percent": float(payload["sa_record"]),
            }
        )
    else:
        identity["seed"] = int(spec["seed"])
    model = model.to(device)
    router = router.to(device)
    model.router = router
    nonfinite_router = [
        name
        for name, value in router.state_dict().items()
        if (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())
    ]
    if nonfinite_router:
        raise RuntimeError(
            f"{spec['label']}: router contains non-finite tensors: "
            f"{len(nonfinite_router)} state entries"
        )
    identity["model_state_sha256"] = state_dict_sha256(model)
    identity["router_state_sha256"] = state_dict_sha256(router)
    return model, router, identity


def _clean_accuracy(
    model: torch.nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            target = torch.from_numpy(labels[start : start + batch_size]).to(device)
            correct += int((model(batch).argmax(dim=1) == target).sum().item())
    return correct / len(inputs)


def _run_state(
    spec: dict[str, Any],
    inputs: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    workspace: Path,
) -> dict[str, Any]:
    model, router, identity = _load_state(spec, device, workspace)
    batch_size = int(config["batch_size"])
    eval_router = copy.deepcopy(router)
    train_router = copy.deepcopy(router)
    eval_scores, eval_before, eval_after = _forward_semantics(
        eval_router,
        inputs,
        device=device,
        batch_size=batch_size,
        train_mode=False,
    )
    train_scores, train_before, train_after = _forward_semantics(
        train_router,
        inputs,
        device=device,
        batch_size=batch_size,
        train_mode=True,
    )
    eval_summary = summarize_scores(eval_scores)
    train_summary = summarize_scores(train_scores)
    eval_summary.update({"bn_before": eval_before, "bn_after": eval_after})
    train_summary.update({"bn_before": train_before, "bn_after": train_after})

    subset_indices = np.asarray(config["diagnostic_subset_indices"], dtype=np.int64)
    subset = torch.from_numpy(inputs[subset_indices]).to(device)
    router.eval()
    with torch.no_grad():
        clean_routes = router(subset).argmax(dim=1)
    gradient = clean_margin_diagnostics(router, subset, clean_routes)
    attack_config = config["strong_pgd"]
    attack = strong_pgd_route_flip(
        router,
        subset,
        clean_routes,
        epsilon=float(attack_config["epsilon"]),
        steps=int(attack_config["steps"]),
        restarts=int(attack_config["restarts"]),
        step_divisor=float(attack_config["step_divisor"]),
        seed=int(attack_config["seed"]),
    )
    artifact_path = output_dir / f"{spec['label'].lower()}_raw.npz"
    with artifact_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            labels=labels.astype(np.int64),
            eval_scores=eval_scores.astype(np.float32),
            train_scores=train_scores.astype(np.float32),
            diagnostic_subset_indices=subset_indices,
            diagnostic_clean_routes=clean_routes.detach().cpu().numpy(),
            **{f"gradient_{key}": value for key, value in gradient.items()},
            **{
                f"attack_{key}": value
                for key, value in attack.items()
                if isinstance(value, np.ndarray)
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "identity": identity,
        "clean_accuracy_eval_semantics": _clean_accuracy(
            model, inputs, labels, device=device, batch_size=batch_size
        ),
        "EVAL_CURRENT_RUNNING_STATS": eval_summary,
        "TRAIN_ORDERED_TEST_BATCH_STATS": train_summary,
        "diagnostic_subset": {
            "indices": subset_indices.tolist(),
            "clean_route_counts": np.bincount(
                clean_routes.detach().cpu().numpy(), minlength=2
            ).astype(int).tolist(),
            "clean_margin": _distribution(gradient["clean_margin"]),
            "gradient_l1": _distribution(gradient["gradient_l1"]),
            "strong_pgd": {
                "epsilon": float(attack_config["epsilon"]),
                "steps": int(attack_config["steps"]),
                "restarts": int(attack_config["restarts"]),
                "route_flip_count": int(np.asarray(attack["success"], dtype=bool).sum()),
                "attacked_margin": _distribution(attack["attacked_margin"]),
                "margin_compression_fraction": _distribution(
                    attack["margin_compression_fraction"]
                ),
                "maximum_linf": float(np.asarray(attack["linf"]).max()),
                "interpretation": "attack witness search; zero flips never imply stability",
            },
        },
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
    }


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace_boundary"])
    config_path = _inside(config_path, workspace)
    act_repository = _inside(Path(config["act_repository"]), workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)
    source = _inside(Path(config["official_source"]["repository"]), workspace)
    archive = _inside(Path(config["dataset_archive"]), workspace)
    training_config = _inside(Path(config["training_config"]), workspace)
    training_audit_path = _inside(Path(config["training_audit"]), workspace)
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("unexpected config status")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if _git(act_repository, "branch", "--show-current") != config["required_branch"]:
        raise RuntimeError("ACT branch gate failed")
    if _git(act_repository, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree is dirty")
    if _git(source, "rev-parse", "HEAD") != config["official_source"]["commit"]:
        raise RuntimeError("official source commit mismatch")
    if _git(source, "rev-parse", "HEAD^{tree}") != config["official_source"]["tree"]:
        raise RuntimeError("official source tree mismatch")
    if _git(source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")
    if _sha256(archive) != config["dataset_archive_sha256"]:
        raise RuntimeError("dataset archive hash mismatch")
    training_audit = json.loads(training_audit_path.read_text(encoding="utf-8"))
    if training_audit.get("status") != "PASS" or training_audit.get("issues") != []:
        raise RuntimeError("training audit gate is not PASS with zero issues")
    recorded_training_config = training_audit.get("config", {}).get("path")
    if not isinstance(recorded_training_config, str) or _resolve_recorded_path(
        recorded_training_config, act_repository
    ) != training_config.resolve():
        raise RuntimeError("training audit points to a different training config")
    if training_audit.get("config", {}).get("sha256") != _sha256(training_config):
        raise RuntimeError("training config hash differs from accepted audit")
    if training_audit.get("official_source", {}).get("commit") != config["official_source"]["commit"]:
        raise RuntimeError("training audit official-source commit mismatch")
    if training_audit.get("official_source", {}).get("tree") != config["official_source"]["tree"]:
        raise RuntimeError("training audit official-source tree mismatch")
    device = torch.device(config["device"])
    free, total = torch.cuda.mem_get_info(device)
    required = int(float(config["minimum_free_gpu_memory_gib"]) * 1024**3)
    if free < required:
        raise RuntimeError("GPU memory gate failed")
    torch.set_num_threads(int(config["torch_threads"]))
    torch.set_num_interop_threads(1)
    inputs, labels = load_cifar10_test_archive(archive)
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    for spec in config["model_states"]:
        row = _run_state(
            spec, inputs, labels, config, output_dir, device, workspace
        )
        rows.append(row)
        _write_json(output_dir / f"{spec['label'].lower()}_summary.json", row)
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "dataset": {
            "archive": str(archive),
            "sha256": _sha256(archive),
            "ordered_test_samples": int(len(inputs)),
        },
        "execution": {
            "device": str(device),
            "free_gpu_gib_before": free / 1024**3,
            "total_gpu_gib": total / 1024**3,
            "elapsed_seconds": time.monotonic() - started,
        },
        "rows": rows,
        "interpretation": config["interpretation"],
        "official_source_clean_after": not bool(_git(source, "status", "--porcelain=v1")),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    compact = {
        row["identity"]["label"]: {
            "clean_accuracy": row["clean_accuracy_eval_semantics"],
            "eval_route_counts": row["EVAL_CURRENT_RUNNING_STATS"]["route_counts"],
            "train_route_counts": row["TRAIN_ORDERED_TEST_BATCH_STATS"]["route_counts"],
            "pgd_route_flips": row["diagnostic_subset"]["strong_pgd"]["route_flip_count"],
        }
        for row in result["rows"]
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
