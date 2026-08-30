"""Compare AdvMoE initialization routing under two explicit BN semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
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


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std(ddof=0)),
    }


def _bn_identity(router: nn.Module) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for name, module in router.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        row = {
            "name": name,
            "training": bool(module.training),
            "running_mean_minimum": float(module.running_mean.min().item()),
            "running_mean_maximum": float(module.running_mean.max().item()),
            "running_variance_minimum": float(module.running_var.min().item()),
            "running_variance_maximum": float(module.running_var.max().item()),
            "num_batches_tracked": int(module.num_batches_tracked.item()),
        }
        layers.append(row)
        for tensor in (module.running_mean, module.running_var, module.num_batches_tracked):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {"layers": len(layers), "state_sha256": digest.hexdigest(), "rows": layers}


def _forward_semantics(
    router: nn.Module,
    inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    train_mode: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    router.train(train_mode)
    before = _bn_identity(router)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            scores.append(router(batch).detach().cpu().numpy())
    after = _bn_identity(router)
    return np.concatenate(scores), before, after


def summarize_scores(scores: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("AdvMoE route scores must have shape [N,2]")
    routes = scores.argmax(axis=1).astype(np.int64)
    counts = np.bincount(routes, minlength=2).astype(np.int64)
    shares = counts.astype(np.float64) / len(routes)
    nonzero = shares[shares > 0.0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    signed = scores[:, 0].astype(np.float64) - scores[:, 1].astype(np.float64)
    std = float(signed.std(ddof=0))
    return {
        "samples": int(len(routes)),
        "route_counts": counts.tolist(),
        "route_shares": shares.tolist(),
        "route_load_entropy": entropy,
        "effective_route_count": float(math.exp(entropy)),
        "exact_distribution_collapse": bool(int(counts.max()) == len(routes)),
        "collapse_target": int(counts.argmax()) if int(counts.max()) == len(routes) else None,
        "signed_score_difference": {
            **_distribution(signed),
            "absolute_mean_over_standard_deviation": (
                None if std == 0.0 else abs(float(signed.mean())) / std
            ),
        },
        "selected_margin": _distribution(np.abs(signed)),
    }


def aggregate_seed_rows(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    by_label: dict[str, Any] = {}
    for label in ("EVAL_DEFAULT_RUNNING_STATS", "TRAIN_ORDERED_TEST_BATCH_STATS"):
        selected = [row[label] for row in rows]
        collapsed = [bool(item["exact_distribution_collapse"]) for item in selected]
        targets = [item["collapse_target"] for item in selected if item["collapse_target"] is not None]
        maximum_shares = [max(item["route_shares"]) for item in selected]
        ratios = [item["signed_score_difference"]["absolute_mean_over_standard_deviation"] for item in selected]
        by_label[label] = {
            "seeds": seeds,
            "collapsed_seed_count": int(sum(collapsed)),
            "collapsed_seeds": [seed for seed, flag in zip(seeds, collapsed) if flag],
            "collapse_target_counts": {
                str(target): int(targets.count(target)) for target in sorted(set(targets))
            },
            "maximum_route_share": _distribution(np.asarray(maximum_shares)),
            "absolute_mean_over_standard_deviation": _distribution(np.asarray(ratios)),
        }
    return by_label


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refuses to reuse {output_dir}")
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("unexpected protocol status")
    torch.set_num_threads(int(config["torch_threads"]))
    torch.set_num_interop_threads(1)
    device = torch.device(str(config["device"]))
    archive = _inside(Path(config["dataset_archive"]), MOE_ROOT)
    inputs, labels = load_cifar10_test_archive(archive)
    seeds = [int(seed) for seed in config["seeds"]]
    if len(seeds) != 20 or len(set(seeds)) != 20 or seeds[0] != 1234:
        raise RuntimeError("K=20 seed policy changed")

    started = time.monotonic()
    seed_rows: list[dict[str, Any]] = []
    score_arrays: dict[str, np.ndarray] = {}
    for seed in seeds:
        model, eval_router, _moe_type = construct_official_init(seed)
        del model
        model, train_router, _moe_type = construct_official_init(seed)
        del model
        if state_dict_sha256(eval_router) != state_dict_sha256(train_router):
            raise RuntimeError(f"seed {seed}: semantic copies differ before forward")
        eval_scores, eval_before, eval_after = _forward_semantics(
            eval_router,
            inputs,
            device=device,
            batch_size=int(config["batch_size"]),
            train_mode=False,
        )
        train_scores, train_before, train_after = _forward_semantics(
            train_router,
            inputs,
            device=device,
            batch_size=int(config["batch_size"]),
            train_mode=True,
        )
        eval_summary = summarize_scores(eval_scores)
        train_summary = summarize_scores(train_scores)
        eval_summary["bn_before"] = eval_before
        eval_summary["bn_after"] = eval_after
        train_summary["bn_before"] = train_before
        train_summary["bn_after"] = train_after
        seed_rows.append(
            {
                "seed": seed,
                "router_initial_sha256": state_dict_sha256(eval_router),
                "EVAL_DEFAULT_RUNNING_STATS": eval_summary,
                "TRAIN_ORDERED_TEST_BATCH_STATS": train_summary,
            }
        )
        score_arrays[f"seed_{seed}_eval_scores"] = eval_scores.astype(np.float32)
        score_arrays[f"seed_{seed}_train_scores"] = train_scores.astype(np.float32)

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "scores.npz"
    with artifact_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            labels=labels.astype(np.int64),
            seeds=np.asarray(seeds, dtype=np.int64),
            **score_arrays,
        )
        handle.flush()
        os.fsync(handle.fileno())
    aggregate = aggregate_seed_rows(seed_rows, seeds)
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "dataset": {
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "samples": int(len(inputs)),
            "semantics": config["dataset_semantics"],
        },
        "execution": {
            "device": str(device),
            "torch_threads": int(config["torch_threads"]),
            "batch_size": int(config["batch_size"]),
            "elapsed_seconds": time.monotonic() - started,
        },
        "seed_roles": config["seed_roles"],
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "default_seed": seed_rows[0],
        "fresh_seed_aggregate": aggregate_seed_rows(seed_rows[1:], seeds[1:]),
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "interpretation": config["interpretation"],
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"aggregate": result["aggregate"], "default_seed": result["default_seed"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
