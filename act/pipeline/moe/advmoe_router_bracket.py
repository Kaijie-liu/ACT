"""Prepare, attack, bound, and summarize an AdvMoE router bracket pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import subprocess
import tarfile
import time
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


CROWN_PYTHON = Path("/data1/Kane/MOE/envs/alpha-beta-crown/bin/python")
WORKER_PATH = PROJECT_ROOT / "act/pipeline/moe/advmoe_router_bound_worker.py"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_cifar10_test_archive(archive: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = _inside(archive, MOE_ROOT)
    with tarfile.open(archive, "r:gz") as handle:
        members = [m for m in handle.getmembers() if m.name.endswith("/test_batch")]
        if len(members) != 1:
            raise RuntimeError("CIFAR-10 archive does not contain one test_batch")
        extracted = handle.extractfile(members[0])
        if extracted is None:
            raise RuntimeError("unable to read CIFAR-10 test_batch")
        payload = pickle.load(extracted, encoding="bytes")
    raw = payload[b"data"]
    labels = payload.get(b"labels", payload.get(b"fine_labels"))
    inputs = raw.reshape(-1, 3, 32, 32).astype(np.float32) / np.float32(255.0)
    return inputs, np.asarray(labels, dtype=np.int64)


def pgd_route_flip(
    router: torch.nn.Module,
    inputs: torch.Tensor,
    clean_routes: torch.Tensor,
    *,
    epsilon: float,
    steps: int,
    restarts: int,
    step_size: float,
    seed: int,
) -> dict[str, Any]:
    router.eval()
    device = inputs.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    lower = torch.clamp(inputs - epsilon, 0, 1)
    upper = torch.clamp(inputs + epsilon, 0, 1)
    batch = torch.arange(len(inputs), device=device)
    competitors = 1 - clean_routes
    best = inputs.detach().clone()
    with torch.no_grad():
        scores = router(best)
        best_objective = scores[batch, competitors] - scores[batch, clean_routes]

    for restart in range(restarts):
        if restart == 0:
            adversarial = inputs.detach().clone()
        else:
            noise = torch.empty_like(inputs).uniform_(
                -epsilon, epsilon, generator=generator
            )
            adversarial = torch.max(torch.min(inputs + noise, upper), lower).detach()
        for _step in range(steps):
            adversarial.requires_grad_(True)
            scores = router(adversarial)
            objective = (
                scores[batch, competitors] - scores[batch, clean_routes]
            )
            gradient = torch.autograd.grad(objective.sum(), adversarial)[0]
            adversarial = adversarial.detach() + step_size * gradient.sign()
            adversarial = torch.max(torch.min(adversarial, upper), lower).detach()
            with torch.no_grad():
                scores = router(adversarial)
                objective = (
                    scores[batch, competitors] - scores[batch, clean_routes]
                )
                improve = objective > best_objective
                best_objective = torch.where(improve, objective, best_objective)
                best[improve] = adversarial[improve]

    with torch.no_grad():
        replay_scores = router(best)
        replay_routes = replay_scores.argmax(dim=1)
    success = replay_routes != clean_routes
    return {
        "success": success.detach().cpu().numpy(),
        "adversarial": best.detach().cpu().numpy(),
        "replay_routes": replay_routes.detach().cpu().numpy(),
        "objective": best_objective.detach().cpu().double().numpy(),
        "linf": (best - inputs)
        .abs()
        .flatten(1)
        .max(dim=1)
        .values.detach()
        .cpu()
        .double()
        .numpy(),
    }


def aggregate_bracket(
    *,
    indices: list[int],
    epsilons: list[float],
    attack_rows: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
    tolerance: float,
    method: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    attack_by_epsilon = {float(row["epsilon"]): row for row in attack_rows}
    bound_by_epsilon = {float(row["epsilon"]): row for row in bound_rows}
    summaries = []
    for epsilon in epsilons:
        attack = attack_by_epsilon[float(epsilon)]
        bound = bound_by_epsilon[float(epsilon)]
        success = np.asarray(attack["success"], dtype=bool)
        lower = np.asarray(bound.get("lower_bounds", []), dtype=np.float64)
        if bound.get("status") == "COMPLETED_NUMERICAL_FILTER":
            if len(lower) != len(indices):
                issues.append(f"epsilon={epsilon}: bound row length mismatch")
                positive = np.zeros(len(indices), dtype=bool)
            else:
                positive = np.isfinite(lower) & (lower >= tolerance)
        else:
            positive = np.zeros(len(indices), dtype=bool)
        overlap = success & positive
        if overlap.any():
            issues.append(
                f"epsilon={epsilon}: {int(overlap.sum())} attack witnesses conflict "
                "with positive numerical bound filters"
            )
        undecided = ~(success | positive)
        summaries.append(
            {
                "epsilon": float(epsilon),
                "samples": len(indices),
                "attack_confirmed_route_unstable": int(success.sum()),
                "positive_numerical_bound_filter": int(positive.sum()),
                "undecided_band": int(undecided.sum()),
                "conflicts": int(overlap.sum()),
                "formal_route_stable": 0,
                "formal_route_stable_reason": (
                    "backend lower bounds are not outward-rounded"
                ),
                "bound_method": method,
                "bound_error": bound.get("error"),
            }
        )
    return summaries, issues


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True)
    archive = _inside(Path(config["dataset"]["archive"]), MOE_ROOT)
    inputs_all, labels_all = load_cifar10_test_archive(archive)
    indices = [int(value) for value in config["sample_indices"]]
    inputs_np = inputs_all[indices]
    labels = labels_all[indices]
    device = torch.device(str(config["attack"]["device"]))
    model, router, _moe_type = construct_official_init(int(config["model_seed"]))
    del model
    router = router.to(device).eval()
    inputs = torch.from_numpy(inputs_np.copy()).to(device)
    with torch.no_grad():
        clean_scores = router(inputs)
        clean_routes = clean_scores.argmax(dim=1)
    router_hash = state_dict_sha256(router)

    input_artifact = output_dir / "inputs.npz"
    np.savez_compressed(
        input_artifact,
        inputs=inputs_np,
        labels=labels,
        dataset_indices=np.asarray(indices, dtype=np.int64),
        clean_scores=clean_scores.detach().cpu().numpy(),
        clean_routes=clean_routes.detach().cpu().numpy(),
    )
    attack_rows = []
    witness_examples = []
    witness_epsilon_slots = []
    witness_sample_slots = []
    for epsilon_slot, epsilon in enumerate(config["epsilons"]):
        epsilon = float(epsilon)
        started = time.monotonic()
        result = pgd_route_flip(
            router,
            inputs,
            clean_routes,
            epsilon=epsilon,
            steps=int(config["attack"]["steps"]),
            restarts=int(config["attack"]["restarts"]),
            step_size=epsilon / float(config["attack"]["step_divisor"]),
            seed=int(config["attack"]["seed"]) + epsilon_slot,
        )
        success_slots = np.flatnonzero(result["success"])
        for slot in success_slots:
            witness_examples.append(result["adversarial"][slot])
            witness_epsilon_slots.append(epsilon_slot)
            witness_sample_slots.append(int(slot))
        attack_rows.append(
            {
                "epsilon": epsilon,
                "success": result["success"].astype(bool).tolist(),
                "replay_routes": result["replay_routes"].astype(int).tolist(),
                "objective": result["objective"].tolist(),
                "linf": result["linf"].tolist(),
                "seconds": time.monotonic() - started,
            }
        )
    witness_artifact = output_dir / "witnesses.npz"
    np.savez_compressed(
        witness_artifact,
        adversarial=np.asarray(witness_examples, dtype=np.float32),
        epsilon_slots=np.asarray(witness_epsilon_slots, dtype=np.int64),
        sample_slots=np.asarray(witness_sample_slots, dtype=np.int64),
    )
    prepare = {
        "schema_version": 1,
        "status": "PREPARED_ATTACK_COMPLETE_BOUND_PENDING",
        "config": config,
        "config_identity": {"path": str(config_path), "sha256": _sha256(config_path)},
        "dataset": {
            "archive": str(archive),
            "sha256": _sha256(archive),
            "ordered_test_size": len(inputs_all),
        },
        "router_sha256": router_hash,
        "input_artifact": {"path": str(input_artifact), "sha256": _sha256(input_artifact)},
        "witness_artifact": {
            "path": str(witness_artifact),
            "sha256": _sha256(witness_artifact),
            "count": len(witness_examples),
        },
        "attack_rows": attack_rows,
        "numerical_policy": {
            "attack_flip": "concrete runtime witness",
            "positive_bound": "numerical filter, not formal SAFE",
            "negative_bound": "UNKNOWN",
        },
    }
    prepare_path = output_dir / "prepare.json"
    _write_json(prepare_path, prepare)
    worker_path = output_dir / "bounds.json"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config["bound_worker"]["torch_threads"])
    env["MKL_NUM_THREADS"] = str(config["bound_worker"]["torch_threads"])
    subprocess.run(
        [
            str(CROWN_PYTHON),
            str(WORKER_PATH),
            "--prepare",
            str(prepare_path),
            "--output",
            str(worker_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        timeout=float(config["bound_worker"]["hard_timeout_seconds"]),
    )
    bounds = json.loads(worker_path.read_text(encoding="utf-8"))
    summaries, issues = aggregate_bracket(
        indices=indices,
        epsilons=[float(value) for value in config["epsilons"]],
        attack_rows=attack_rows,
        bound_rows=bounds["rows"],
        tolerance=float(config["numerical"]["safe_positive_margin"]),
        method=str(config["bound_worker"]["method"]),
    )
    final = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INIT_20_SAMPLE_ENGINEERING_PILOT",
        "parent_commit_required": config["parent_commit_required"],
        "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        "bounds": {"path": str(worker_path), "sha256": _sha256(worker_path)},
        "summaries": summaries,
        "raw_frontend_probe": bounds["raw_frontend_probe"],
        "adapter_equivalence": bounds["adapter_equivalence"],
        "claim_boundary": (
            "This pilot validates the bracket harness. PGD flips are concrete route "
            "witnesses. IBP/CROWN positive margins remain numerical filters until an "
            "outward-rounded backend is supplied. It is not a full-test census."
        ),
    }
    final_path = output_dir / "summary.json"
    _write_json(final_path, final)
    if issues:
        raise RuntimeError(f"router bracket pilot found {len(issues)} issue(s)")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
