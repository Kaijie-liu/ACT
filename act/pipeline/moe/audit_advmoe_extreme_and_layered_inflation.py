"""Audit extreme-epsilon PGD and layered IBP/CROWN inflation artifacts."""

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
from act.pipeline.moe.advmoe_relaxation_inflation import relaxation_inflation
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(
    attack_config_path: Path,
    attack_dir: Path,
    inflation_config_path: Path,
    inflation_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    attack_config_path = _inside(attack_config_path, PROJECT_ROOT)
    attack_dir = _inside(attack_dir, MOE_ROOT)
    inflation_config_path = _inside(inflation_config_path, PROJECT_ROOT)
    inflation_dir = _inside(inflation_dir, MOE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    attack_config = json.loads(attack_config_path.read_text(encoding="utf-8"))
    attack_path = attack_dir / "summary.json"
    attack = json.loads(attack_path.read_text(encoding="utf-8"))
    inflation_config = json.loads(inflation_config_path.read_text(encoding="utf-8"))
    inflation_path = inflation_dir / "summary.json"
    inflation = json.loads(inflation_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if attack.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("extreme attack status changed")
    if attack.get("config", {}).get("sha256") != _sha256(attack_config_path):
        issues.append("extreme attack config hash changed")
    if attack.get("reuse_nested_endpoints") is not True:
        issues.append("nested endpoint reuse is not enabled")
    if inflation.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("layered inflation status changed")
    if inflation.get("config", {}).get("sha256") != _sha256(inflation_config_path):
        issues.append("layered inflation config hash changed")

    source_dir = _inside(Path(attack_config["source_result_dir"]), MOE_ROOT)
    input_path = source_dir / "inputs.npz"
    with np.load(input_path, allow_pickle=False) as arrays:
        inputs = arrays["inputs"].copy()
        ranks = arrays["dataset_indices"].astype(np.int64)
        clean_routes = arrays["clean_routes"].astype(np.int64)
        clean_margin = arrays["clean_margin"].astype(np.float64)
    endpoint_path = attack_dir / "attack_endpoints.npz"
    if attack.get("artifacts", {}).get("endpoints", {}).get("sha256") != _sha256(
        endpoint_path
    ):
        issues.append("extreme endpoint hash changed")
    with np.load(endpoint_path, allow_pickle=False) as arrays:
        endpoints = arrays["adversarial"].copy()
        epsilons = arrays["epsilons"].astype(np.float64)
        endpoint_ranks = arrays["dataset_ranks"].astype(np.int64)
    if not np.array_equal(endpoint_ranks, ranks):
        issues.append("extreme endpoint ranks changed")
    _model, router, _moe_type = construct_official_init(int(attack_config["model_seed"]))
    del _model
    router = router.cpu().eval()
    if state_dict_sha256(router) != attack.get("router_sha256"):
        issues.append("extreme attack router identity changed")
    total_flips = 0
    compression_medians: list[float] = []
    minimum_margins: list[float] = []
    previous_margin: np.ndarray | None = None
    for slot, (epsilon, row) in enumerate(zip(epsilons, attack["attack_rows"])):
        endpoint = endpoints[slot]
        linf = np.max(np.abs(endpoint - inputs), axis=(1, 2, 3))
        if np.any(linf > epsilon + 1e-6):
            issues.append(f"epsilon {epsilon}: endpoint exceeds box")
        with torch.no_grad():
            scores = router(torch.from_numpy(endpoint)).numpy()
        routes = scores.argmax(axis=1)
        margin = scores[np.arange(len(ranks)), clean_routes] - scores[
            np.arange(len(ranks)), 1 - clean_routes
        ]
        success = routes != clean_routes
        total_flips += int(success.sum())
        if not np.array_equal(success, np.asarray(row["success"], dtype=bool)):
            issues.append(f"epsilon {epsilon}: success flags changed")
        if not np.allclose(
            margin,
            np.asarray(row["attacked_margin"], dtype=np.float64),
            atol=2e-5,
            rtol=1e-4,
        ):
            issues.append(f"epsilon {epsilon}: attacked margins changed")
        if previous_margin is not None and np.any(margin > previous_margin + 2e-5):
            issues.append(f"epsilon {epsilon}: nested endpoint monotonicity failed")
        previous_margin = margin
        compression = (clean_margin - margin) / clean_margin
        compression_medians.append(float(np.median(compression)))
        minimum_margins.append(float(np.min(margin)))
    if total_flips != 0:
        issues.append("extreme attack total flip count changed")

    prepare_path = source_dir / "prepare.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    attacks = {float(row["epsilon"]): row for row in prepare["attack_rows"]}
    recomputed: dict[str, list[np.ndarray]] = {}
    recomputed_medians: dict[str, list[float]] = {}
    for registered in inflation_config["bound_sources"]:
        label = str(registered["label"])
        bound_path = _inside(Path(registered["path"]), MOE_ROOT)
        bounds = json.loads(bound_path.read_text(encoding="utf-8"))
        rows_by_epsilon = {float(row["epsilon"]): row for row in bounds["rows"]}
        matrices: list[np.ndarray] = []
        medians: list[float] = []
        for epsilon in sorted(set(attacks) & set(rows_by_epsilon)):
            values = relaxation_inflation(
                clean_margin,
                np.asarray(rows_by_epsilon[epsilon]["lower_bounds"], dtype=np.float64),
                np.asarray(attacks[epsilon]["attacked_margin"], dtype=np.float64),
            )
            matrices.append(values)
            medians.append(float(np.median(values)))
        recomputed[label] = matrices
        recomputed_medians[label] = medians
        raw_layer = next(layer for layer in inflation["layers"] if layer["label"] == label)
        raw_medians = [float(row["inflation"]["median"]) for row in raw_layer["rows"]]
        if not np.allclose(medians, raw_medians, atol=0, rtol=1e-12):
            issues.append(f"{label}: inflation medians changed")
    expected_ratio = np.asarray(recomputed_medians["IBP"]) / np.asarray(
        recomputed_medians["CROWN"]
    )
    raw_ratio = np.asarray(inflation["median_ratios"]["IBP_over_CROWN"])
    if not np.allclose(expected_ratio, raw_ratio, atol=0, rtol=1e-12):
        issues.append("IBP/CROWN median ratios changed")
    csv_path = inflation_dir / "per_sample.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.reader(handle))
    csv_values = np.asarray(
        [[float(value) for value in row[1:]] for row in csv_rows[1:]],
        dtype=np.float64,
    )
    expected_columns = np.concatenate(
        [np.stack(recomputed[label]).T for label in ("IBP", "CROWN")], axis=1
    )
    if not np.allclose(csv_values, expected_columns, atol=0, rtol=0):
        issues.append("layered inflation CSV changed")
    figure_path = inflation_dir / "relaxation_inflation.svg"
    if b"<text" not in figure_path.read_bytes():
        issues.append("layered inflation SVG text is not live")
    if inflation.get("artifacts", {}).get("figure", {}).get("sha256") != _sha256(
        figure_path
    ):
        issues.append("layered inflation figure hash changed")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_AUDIT_ADV_MOE_EXTREME_PGD_AND_LAYERED_INFLATION",
        "attack_summary_sha256": _sha256(attack_path),
        "inflation_summary_sha256": _sha256(inflation_path),
        "extreme_total_flips": total_flips,
        "extreme_compression_medians": compression_medians,
        "extreme_minimum_margins": minimum_margins,
        "inflation_medians": recomputed_medians,
        "ibp_over_crown_median_ratios": expected_ratio.tolist(),
        "conclusion": (
            "Strong PGD finds no route flip on the frozen 20 inputs even when "
            "epsilon=1 spans the entire pixel cube. This is attack non-discovery, "
            "not global constancy. CROWN reduces the IBP inflation diagnostic by "
            "about 5.2x while leaving an approximately 1e11 residual."
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
    parser.add_argument("--attack-config", type=Path, required=True)
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--inflation-config", type=Path, required=True)
    parser.add_argument("--inflation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.attack_config,
        args.attack_dir,
        args.inflation_config,
        args.inflation_dir,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
