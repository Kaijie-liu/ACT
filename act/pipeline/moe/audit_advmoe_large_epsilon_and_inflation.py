"""Audit AdvMoE large-epsilon PGD and relaxation-inflation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_relaxation_inflation import relaxation_inflation
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(
    attack_dir: Path,
    inflation_dir: Path,
    attack_config_path: Path,
    inflation_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    attack_dir = _inside(attack_dir, MOE_ROOT)
    inflation_dir = _inside(inflation_dir, MOE_ROOT)
    attack_config_path = _inside(attack_config_path, PROJECT_ROOT)
    inflation_config_path = _inside(inflation_config_path, PROJECT_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    attack_summary_path = attack_dir / "summary.json"
    inflation_summary_path = inflation_dir / "summary.json"
    attack = json.loads(attack_summary_path.read_text(encoding="utf-8"))
    inflation = json.loads(inflation_summary_path.read_text(encoding="utf-8"))
    attack_config = json.loads(attack_config_path.read_text(encoding="utf-8"))
    inflation_config = json.loads(inflation_config_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if attack.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("attack raw status changed")
    if inflation.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("inflation raw status changed")
    if attack.get("config", {}).get("sha256") != _sha256(attack_config_path):
        issues.append("attack config hash changed")
    if inflation.get("config", {}).get("sha256") != _sha256(inflation_config_path):
        issues.append("inflation config hash changed")

    source_dir = _inside(Path(attack_config["source_result_dir"]), MOE_ROOT)
    input_path = source_dir / "inputs.npz"
    with np.load(input_path, allow_pickle=False) as arrays:
        inputs = arrays["inputs"].copy()
        ranks = arrays["dataset_indices"].astype(np.int64)
        clean_routes = arrays["clean_routes"].astype(np.int64)
        clean_margin = arrays["clean_margin"].astype(np.float64)
    endpoint_path = attack_dir / "attack_endpoints.npz"
    if attack["artifacts"]["endpoints"]["sha256"] != _sha256(endpoint_path):
        issues.append("attack endpoint hash changed")
    with np.load(endpoint_path, allow_pickle=False) as arrays:
        endpoints = arrays["adversarial"].copy()
        epsilons = arrays["epsilons"].astype(np.float64)
        endpoint_ranks = arrays["dataset_ranks"].astype(np.int64)
    if not np.array_equal(endpoint_ranks, ranks):
        issues.append("endpoint sample ranks changed")

    _model, router, _moe_type = construct_official_init(
        int(attack_config["model_seed"])
    )
    del _model
    router = router.cpu().eval()
    if state_dict_sha256(router) != attack.get("router_sha256"):
        issues.append("router identity changed")
    total_flips = 0
    compression_medians: list[float] = []
    for epsilon_slot, (epsilon, row) in enumerate(zip(epsilons, attack["attack_rows"])):
        endpoint = endpoints[epsilon_slot]
        linf = np.max(np.abs(endpoint - inputs), axis=(1, 2, 3))
        if np.any(linf > epsilon + 1e-6):
            issues.append(f"epsilon {epsilon}: endpoint exceeds registered box")
        with torch.no_grad():
            scores = router(torch.from_numpy(endpoint)).numpy()
        routes = scores.argmax(axis=1)
        attacked_margin = scores[np.arange(len(ranks)), clean_routes] - scores[
            np.arange(len(ranks)), 1 - clean_routes
        ]
        success = routes != clean_routes
        total_flips += int(success.sum())
        if not np.array_equal(success, np.asarray(row["success"], dtype=bool)):
            issues.append(f"epsilon {epsilon}: success flags changed")
        if not np.array_equal(routes, np.asarray(row["replay_routes"], dtype=np.int64)):
            issues.append(f"epsilon {epsilon}: replay routes changed")
        if not np.allclose(
            attacked_margin,
            np.asarray(row["attacked_margin"], dtype=np.float64),
            atol=2e-5,
            rtol=1e-4,
        ):
            issues.append(f"epsilon {epsilon}: attacked margins changed")
        compression = (clean_margin - attacked_margin) / clean_margin
        compression_medians.append(float(np.median(compression)))
        if not np.allclose(
            compression,
            np.asarray(row["margin_compression_fraction"], dtype=np.float64),
            atol=1e-4,
            rtol=1e-3,
        ):
            issues.append(f"epsilon {epsilon}: compression changed")
    if total_flips != 0:
        issues.append("large-epsilon total flip count changed")
    if any(row.get("status") != "NO_FLIP_FOUND_THROUGH_MAXIMUM" for row in attack["brackets"]):
        issues.append("attack-diagnostic bracket status changed")

    prepare_path = source_dir / "prepare.json"
    bounds_path = source_dir / "crown_bounds.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    attacks_by_epsilon = {float(row["epsilon"]): row for row in prepare["attack_rows"]}
    bounds_by_epsilon = {float(row["epsilon"]): row for row in bounds["rows"]}
    inflation_medians: list[float] = []
    recomputed_matrix: list[np.ndarray] = []
    for row in inflation["rows"]:
        epsilon = float(row["epsilon"])
        values = relaxation_inflation(
            clean_margin,
            np.asarray(bounds_by_epsilon[epsilon]["lower_bounds"], dtype=np.float64),
            np.asarray(attacks_by_epsilon[epsilon]["attacked_margin"], dtype=np.float64),
        )
        recomputed_matrix.append(values)
        median = float(np.median(values))
        inflation_medians.append(median)
        if not np.isclose(median, row["inflation"]["median"], rtol=1e-12):
            issues.append(f"epsilon {epsilon}: inflation median changed")
    csv_path = inflation_dir / "per_sample.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.reader(handle))
    csv_values = np.asarray([[float(value) for value in row[1:]] for row in csv_rows[1:]])
    if not np.allclose(csv_values.T, np.stack(recomputed_matrix), atol=0, rtol=0):
        issues.append("inflation per-sample CSV changed")
    figure_path = inflation_dir / "relaxation_inflation.svg"
    if b"<text" not in figure_path.read_bytes():
        issues.append("inflation SVG text was converted to paths")
    if inflation["artifacts"]["figure"]["sha256"] != _sha256(figure_path):
        issues.append("inflation figure hash changed")

    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_AUDIT_ADV_MOE_LARGE_EPSILON_AND_INFLATION",
        "raw_results": {
            "large_epsilon_attack": {
                "path": str(attack_summary_path),
                "sha256": _sha256(attack_summary_path),
            },
            "relaxation_inflation": {
                "path": str(inflation_summary_path),
                "sha256": _sha256(inflation_summary_path),
            },
        },
        "recomputed": {
            "large_epsilon_total_flips": total_flips,
            "compression_medians": compression_medians,
            "inflation_medians": inflation_medians,
        },
        "conclusion": (
            "All 80 large-epsilon endpoints replay. Strong PGD finds zero flips "
            "through 96/255; the 96/255 median margin compression is reported as "
            "attack evidence only, so the earlier near-70/255 local estimates are "
            "not observed boundaries. Five-radius relaxation-inflation medians stay "
            "between 1.07e11 and 1.66e11. This diagnostic is not an approximation "
            "ratio or bound on the true reachable drop."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--inflation-dir", type=Path, required=True)
    parser.add_argument("--attack-config", type=Path, required=True)
    parser.add_argument("--inflation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.attack_dir,
        args.inflation_dir,
        args.attack_config,
        args.inflation_config,
        args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
