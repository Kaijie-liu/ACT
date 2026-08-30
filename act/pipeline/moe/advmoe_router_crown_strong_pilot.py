"""Run the frozen AdvMoE strong-PGD plus scalable-CROWN init pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_router_bracket import (
    CROWN_PYTHON,
    WORKER_PATH,
    _write_json,
    aggregate_bracket,
    clean_margin_diagnostics,
    load_cifar10_test_archive,
    strong_pgd_route_flip,
)
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def distribution(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {key: None for key in ("min", "q25", "median", "q75", "max")}
    quantiles = np.quantile(finite, [0, 0.25, 0.5, 0.75, 1])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, quantiles)))


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    attack_device = str(config["attack"]["device"])
    if attack_device.startswith("cuda"):
        free_memory, _total_memory = torch.cuda.mem_get_info(torch.device(attack_device))
        required_free = int(config["resource_gate"]["minimum_free_gpu_memory_bytes"])
        if free_memory < required_free:
            raise RuntimeError(
                f"GPU resource gate: {free_memory} free bytes < {required_free} required"
            )
    output_dir.mkdir(parents=True)
    archive = _inside(Path(config["dataset"]["archive"]), MOE_ROOT)
    inputs_all, labels_all = load_cifar10_test_archive(archive)
    indices = [int(value) for value in config["sample_indices"]]
    inputs_np = inputs_all[indices]
    labels = labels_all[indices]
    device = torch.device(attack_device)
    model, router, _moe_type = construct_official_init(int(config["model_seed"]))
    del model
    router = router.to(device).eval()
    inputs = torch.from_numpy(inputs_np.copy()).to(device)
    with torch.no_grad():
        clean_scores = router(inputs)
        clean_routes = clean_scores.argmax(dim=1)
    clean = clean_margin_diagnostics(router, inputs, clean_routes)
    router_hash = state_dict_sha256(router)

    input_artifact = output_dir / "inputs.npz"
    np.savez_compressed(
        input_artifact,
        inputs=inputs_np,
        labels=labels,
        dataset_indices=np.asarray(indices, dtype=np.int64),
        clean_scores=clean_scores.detach().cpu().numpy(),
        clean_routes=clean_routes.detach().cpu().numpy(),
        **clean,
    )
    attack_rows: list[dict[str, Any]] = []
    best_endpoints = []
    attack_started = time.monotonic()
    for epsilon_slot, epsilon_value in enumerate(config["epsilons"]):
        epsilon = float(epsilon_value)
        started = time.monotonic()
        result = strong_pgd_route_flip(
            router,
            inputs,
            clean_routes,
            epsilon=epsilon,
            steps=int(config["attack"]["steps"]),
            restarts=int(config["attack"]["restarts"]),
            step_divisor=float(config["attack"]["step_divisor"]),
            seed=int(config["attack"]["seed"]) + epsilon_slot,
        )
        best_endpoints.append(result["adversarial"])
        attack_rows.append(
            {
                "epsilon": epsilon,
                "success": result["success"].astype(bool).tolist(),
                "replay_routes": result["replay_routes"].astype(int).tolist(),
                "clean_margin": result["clean_margin"].tolist(),
                "attacked_margin": result["attacked_margin"].tolist(),
                "margin_compression_fraction": result[
                    "margin_compression_fraction"
                ].tolist(),
                "linf": result["linf"].tolist(),
                "schedule": result["schedule"],
                "seconds": time.monotonic() - started,
            }
        )
    attack_seconds = time.monotonic() - attack_started
    endpoint_artifact = output_dir / "attack_endpoints.npz"
    np.savez_compressed(
        endpoint_artifact,
        adversarial=np.asarray(best_endpoints, dtype=np.float32),
        epsilons=np.asarray(config["epsilons"], dtype=np.float64),
        dataset_indices=np.asarray(indices, dtype=np.int64),
    )

    prepare = {
        "schema_version": 1,
        "status": "STRONG_ATTACK_COMPLETE_CROWN_PENDING",
        "config": config,
        "config_identity": {"path": str(config_path), "sha256": _sha256(config_path)},
        "dataset": {
            "archive": str(archive),
            "sha256": _sha256(archive),
            "ordered_test_size": len(inputs_all),
        },
        "router_sha256": router_hash,
        "input_artifact": {"path": str(input_artifact), "sha256": _sha256(input_artifact)},
        "attack_endpoint_artifact": {
            "path": str(endpoint_artifact),
            "sha256": _sha256(endpoint_artifact),
        },
        "attack_rows": attack_rows,
        "attack_seconds": attack_seconds,
        "clean_diagnostics": {
            "margin": distribution(clean["clean_margin"]),
            "gradient_l1": distribution(clean["gradient_l1"]),
            "gradient_l2": distribution(clean["gradient_l2"]),
            "gradient_linf": distribution(clean["gradient_linf"]),
        },
        "numerical_policy": {
            "attack_flip": "concrete literal-router witness",
            "crown_positive_bound": "numerical filter, not formal SAFE",
            "crown_negative_bound": "UNKNOWN",
        },
    }
    prepare_path = output_dir / "prepare.json"
    _write_json(prepare_path, prepare)
    bounds_path = output_dir / "crown_bounds.json"
    worker_log_path = output_dir / "crown_worker.log"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(config["bound_worker"]["torch_threads"])
    environment["MKL_NUM_THREADS"] = str(config["bound_worker"]["torch_threads"])
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with worker_log_path.open("x", encoding="utf-8") as worker_log:
        subprocess.run(
            [
                str(CROWN_PYTHON),
                str(WORKER_PATH),
                "--prepare",
                str(prepare_path),
                "--output",
                str(bounds_path),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=float(config["bound_worker"]["hard_timeout_seconds"]),
        )
        worker_log.flush()
        os.fsync(worker_log.fileno())
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    summaries, issues = aggregate_bracket(
        indices=indices,
        epsilons=[float(value) for value in config["epsilons"]],
        attack_rows=attack_rows,
        bound_rows=bounds["rows"],
        tolerance=float(config["numerical"]["safe_positive_margin"]),
        method=str(config["bound_worker"]["method"]),
    )
    attack_by_epsilon = {float(row["epsilon"]): row for row in attack_rows}
    bound_by_epsilon = {float(row["epsilon"]): row for row in bounds["rows"]}
    diagnostic_rows = []
    for summary in summaries:
        epsilon = float(summary["epsilon"])
        attack = attack_by_epsilon[epsilon]
        bound = bound_by_epsilon[epsilon]
        diagnostic_rows.append(
            {
                **summary,
                "attacked_margin": distribution(np.asarray(attack["attacked_margin"])),
                "margin_compression_fraction": distribution(
                    np.asarray(attack["margin_compression_fraction"])
                ),
                "crown_lower_bound": distribution(
                    np.asarray(bound["lower_bounds"], dtype=np.float64)
                ),
                "attack_seconds": float(attack["seconds"]),
                "crown_seconds": float(bound["seconds"]),
            }
        )
    final = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INIT_20_SAMPLE_STRONG_ATTACK_SPARSE_CROWN_ENGINEERING_PILOT",
        "parent_commit_required": config["parent_commit_required"],
        "artifacts": {
            "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
            "bounds": {"path": str(bounds_path), "sha256": _sha256(bounds_path)},
            "worker_log": {
                "path": str(worker_log_path),
                "sha256": _sha256(worker_log_path),
            },
            "inputs": {"path": str(input_artifact), "sha256": _sha256(input_artifact)},
            "attack_endpoints": {
                "path": str(endpoint_artifact),
                "sha256": _sha256(endpoint_artifact),
            },
        },
        "router_sha256": router_hash,
        "clean_diagnostics": prepare["clean_diagnostics"],
        "rows": diagnostic_rows,
        "raw_frontend_probe": bounds["raw_frontend_probe"],
        "adapter_equivalence": bounds["adapter_equivalence"],
        "batchnorm_deployment_identity": bounds["batchnorm_deployment_identity"],
        "bound_options": bounds["bound_options"],
        "peak_memory_bytes": bounds["peak_memory_bytes"],
        "numerical_scope": bounds["numerical_scope"],
        "claim_boundary": (
            "This init engineering pilot measures attack strength and sparse-CROWN "
            "bound scale. Numerical filters are not outward-rounded certificates; "
            "undecided rows do not establish stability or instability."
        ),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, final)
    if issues:
        raise RuntimeError(f"strong/CROWN pilot found {len(issues)} issue(s)")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
