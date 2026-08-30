"""Run frozen large-epsilon strong PGD on the AdvMoE init-router cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_router_bracket import strong_pgd_route_flip
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def attack_diagnostic_brackets(
    success: np.ndarray,
    epsilons: np.ndarray,
    known_no_flip_epsilon: float,
) -> list[dict[str, Any]]:
    success = np.asarray(success, dtype=bool)
    epsilons = np.asarray(epsilons, dtype=np.float64)
    if success.ndim != 2 or success.shape[0] != len(epsilons):
        raise ValueError("success must have shape [epsilon, sample]")
    if np.any(np.diff(epsilons) <= 0) or known_no_flip_epsilon >= epsilons[0]:
        raise ValueError("epsilon grid must increase above the known no-flip point")
    rows: list[dict[str, Any]] = []
    for sample in range(success.shape[1]):
        hits = np.flatnonzero(success[:, sample])
        if len(hits):
            first = int(hits[0])
            upper = float(epsilons[first])
            lower = (
                float(epsilons[first - 1])
                if first > 0
                else float(known_no_flip_epsilon)
            )
            status = "FLIP_FOUND"
        else:
            lower = float(epsilons[-1])
            upper = None
            status = "NO_FLIP_FOUND_THROUGH_MAXIMUM"
        rows.append(
            {
                "status": status,
                "largest_tested_without_found_flip": lower,
                "smallest_epsilon_with_found_flip": upper,
            }
        )
    return rows


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refuses to reuse {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("registered CUDA attack has no CUDA device")
    free, total = torch.cuda.mem_get_info()
    if free < int(config["minimum_free_memory_bytes"]):
        raise RuntimeError("free CUDA memory is below the attack resource gate")

    source_dir = _inside(Path(config["source_result_dir"]), MOE_ROOT)
    input_path = source_dir / "inputs.npz"
    source_prepare_path = source_dir / "prepare.json"
    source_prepare = json.loads(source_prepare_path.read_text(encoding="utf-8"))
    with np.load(input_path, allow_pickle=False) as arrays:
        inputs_np = arrays["inputs"].copy()
        ranks = arrays["dataset_indices"].astype(np.int64)
        clean_routes_np = arrays["clean_routes"].astype(np.int64)
        clean_margin = arrays["clean_margin"].astype(np.float64)
    _model, router, _moe_type = construct_official_init(int(config["model_seed"]))
    del _model
    if state_dict_sha256(router) != source_prepare["router_sha256"]:
        raise RuntimeError("router identity changed")
    device = str(config["device"])
    router = router.to(device).eval()
    inputs = torch.from_numpy(inputs_np).to(device)
    clean_routes = torch.from_numpy(clean_routes_np).long().to(device)
    epsilons = np.asarray(config["epsilon_over_255"], dtype=np.float64) / 255.0
    attack = config["attack"]
    endpoints: list[np.ndarray] = []
    success_rows: list[np.ndarray] = []
    result_rows: list[dict[str, Any]] = []
    for epsilon_slot, epsilon in enumerate(epsilons):
        result = strong_pgd_route_flip(
            router,
            inputs,
            clean_routes,
            epsilon=float(epsilon),
            steps=int(attack["steps"]),
            restarts=int(attack["restarts"]),
            step_divisor=float(attack["step_divisor"]),
            seed=int(attack["seed"]) + epsilon_slot,
        )
        endpoints.append(result["adversarial"])
        success = np.asarray(result["success"], dtype=bool)
        success_rows.append(success)
        result_rows.append(
            {
                "epsilon": float(epsilon),
                "epsilon_over_255": float(epsilon * 255.0),
                "success": success.tolist(),
                "success_count": int(success.sum()),
                "attacked_margin": np.asarray(
                    result["attacked_margin"], dtype=np.float64
                ).tolist(),
                "margin_compression_fraction": np.asarray(
                    result["margin_compression_fraction"], dtype=np.float64
                ).tolist(),
                "replay_routes": np.asarray(
                    result["replay_routes"], dtype=np.int64
                ).tolist(),
                "linf": np.asarray(result["linf"], dtype=np.float64).tolist(),
                "seconds": float(result["seconds"]),
            }
        )
    success_matrix = np.stack(success_rows)
    brackets = attack_diagnostic_brackets(
        success_matrix,
        epsilons,
        float(config["known_no_flip_epsilon_over_255"]) / 255.0,
    )
    for rank, bracket in zip(ranks, brackets):
        bracket["dataset_rank"] = int(rank)
        bracket["largest_tested_without_found_flip_x255"] = (
            bracket["largest_tested_without_found_flip"] * 255.0
        )
        upper = bracket["smallest_epsilon_with_found_flip"]
        bracket["smallest_epsilon_with_found_flip_x255"] = (
            None if upper is None else upper * 255.0
        )

    output_dir.mkdir(parents=True)
    endpoint_path = output_dir / "attack_endpoints.npz"
    np.savez_compressed(
        endpoint_path,
        adversarial=np.stack(endpoints),
        epsilons=epsilons,
        dataset_ranks=ranks,
    )
    csv_path = output_dir / "per_sample_brackets.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(brackets[0]))
        writer.writeheader()
        writer.writerows(brackets)
    summary = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "source": {
            "inputs": {"path": str(input_path), "sha256": _sha256(input_path)},
            "prepare": {
                "path": str(source_prepare_path),
                "sha256": _sha256(source_prepare_path),
            },
        },
        "router_sha256": state_dict_sha256(router),
        "resource_gate": {"free_bytes_before": int(free), "total_bytes": int(total)},
        "clean_margin": clean_margin.tolist(),
        "attack_rows": result_rows,
        "brackets": brackets,
        "artifacts": {
            "endpoints": {"path": str(endpoint_path), "sha256": _sha256(endpoint_path)},
            "brackets_csv": {"path": str(csv_path), "sha256": _sha256(csv_path)},
        },
        "interpretation": config["interpretation"],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
