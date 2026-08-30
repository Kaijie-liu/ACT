"""Independently replay the AdvMoE K=20 dual-BN initialization census."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_init_bn_semantics import (
    _forward_semantics,
    aggregate_seed_rows,
    summarize_scores,
)
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(config_path: Path, result_dir: Path, output_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    result_dir = _inside(result_dir, MOE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact_path = result_dir / "scores.npz"
    issues: list[str] = []
    if summary.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw status changed")
    if summary.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash changed")
    if summary.get("artifact", {}).get("sha256") != _sha256(artifact_path):
        issues.append("score artifact hash changed")

    torch.set_num_threads(int(config["torch_threads"]))
    torch.set_num_interop_threads(1)
    inputs, labels = load_cifar10_test_archive(Path(config["dataset_archive"]))
    device = torch.device(str(config["device"]))
    seeds = [int(seed) for seed in config["seeds"]]
    replay_rows: list[dict[str, Any]] = []
    with np.load(artifact_path, allow_pickle=False) as stored:
        if not np.array_equal(stored["labels"], labels):
            issues.append("ordered labels changed")
        if not np.array_equal(stored["seeds"], np.asarray(seeds)):
            issues.append("seed order changed")
        for seed in seeds:
            model, eval_router, _moe_type = construct_official_init(seed)
            del model
            model, train_router, _moe_type = construct_official_init(seed)
            del model
            initial_hash = state_dict_sha256(eval_router)
            if initial_hash != state_dict_sha256(train_router):
                issues.append(f"seed {seed}: semantic copies differ before replay")
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
            if not np.array_equal(eval_scores, stored[f"seed_{seed}_eval_scores"]):
                issues.append(f"seed {seed}: eval scores changed")
            if not np.array_equal(train_scores, stored[f"seed_{seed}_train_scores"]):
                issues.append(f"seed {seed}: train scores changed")
            eval_summary = summarize_scores(eval_scores)
            train_summary = summarize_scores(train_scores)
            eval_summary["bn_before"], eval_summary["bn_after"] = eval_before, eval_after
            train_summary["bn_before"], train_summary["bn_after"] = train_before, train_after
            replay_rows.append(
                {
                    "seed": seed,
                    "router_initial_sha256": initial_hash,
                    "EVAL_DEFAULT_RUNNING_STATS": eval_summary,
                    "TRAIN_ORDERED_TEST_BATCH_STATS": train_summary,
                }
            )
    if replay_rows != summary.get("seed_rows"):
        issues.append("per-seed summaries changed")
    aggregate = aggregate_seed_rows(replay_rows, seeds)
    if aggregate != summary.get("aggregate"):
        issues.append("aggregate changed")
    fresh_aggregate = aggregate_seed_rows(replay_rows[1:], seeds[1:])
    if fresh_aggregate != summary.get("fresh_seed_aggregate"):
        issues.append("fresh-seed aggregate changed")
    if replay_rows[0] != summary.get("default_seed"):
        issues.append("default-seed summary changed")
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_AUDIT_ADV_MOE_INIT_K20_DUAL_BN_SEMANTICS",
        "config_sha256": _sha256(config_path),
        "summary_sha256": _sha256(summary_path),
        "artifact_sha256": _sha256(artifact_path),
        "aggregate": aggregate,
        "fresh_seed_aggregate": fresh_aggregate,
        "conclusion": (
            "The K=20 route shares and margin distributions replay under both "
            "registered BN semantics. Train-mode rows are ordered-test co-batch "
            "diagnostics, not literal augmented shuffled training batches."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.result_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
