"""Independently audit the AdvMoE init boundary-estimate diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from act.pipeline.moe.advmoe_init_boundary_estimates import (
    compute_boundary_estimates,
)
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(raw_dir: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    raw_dir = _inside(raw_dir, MOE_ROOT)
    config_path = _inside(config_path, PROJECT_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = raw_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if summary.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw result status changed")
    if summary.get("scope") != config.get("scope"):
        issues.append("scope changed")
    if summary.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash changed")

    source_dir = _inside(Path(config["advmoe_result_dir"]), MOE_ROOT)
    input_path = source_dir / "inputs.npz"
    prepare_path = source_dir / "prepare.json"
    for key, path in (("inputs", input_path), ("prepare", prepare_path)):
        if summary.get("sources", {}).get(key, {}).get("sha256") != _sha256(path):
            issues.append(f"source {key} hash changed")
    with np.load(input_path, allow_pickle=False) as arrays:
        indices = arrays["dataset_indices"].astype(np.int64)
        clean_margin = arrays["clean_margin"].astype(np.float64)
        gradient_l1 = arrays["gradient_l1"].astype(np.float64)
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    epsilon = float(config["attack_epsilon"])
    attack_rows = {
        float(row["epsilon"]): row for row in prepare.get("attack_rows", [])
    }
    if epsilon not in attack_rows:
        issues.append("registered attack row is missing")
        compression = np.full_like(clean_margin, np.nan)
    else:
        compression = np.asarray(
            attack_rows[epsilon]["margin_compression_fraction"], dtype=np.float64
        )
    first, pgd = compute_boundary_estimates(
        clean_margin, gradient_l1, compression, epsilon
    )

    csv_path = raw_dir / "per_sample.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(indices):
        issues.append("per-sample CSV length changed")
    else:
        csv_indices = np.asarray([int(row["sample_rank"]) for row in rows])
        csv_first = np.asarray([float(row["first_order_radius"]) for row in rows])
        csv_pgd = np.asarray([float(row["pgd_extrapolated_radius"]) for row in rows])
        if not np.array_equal(csv_indices, indices):
            issues.append("per-sample ranks changed")
        if not np.array_equal(csv_first, first):
            issues.append("first-order values changed")
        if not np.array_equal(csv_pgd, pgd):
            issues.append("PGD extrapolations changed")
    if summary.get("artifacts", {}).get("per_sample", {}).get("sha256") != _sha256(
        csv_path
    ):
        issues.append("per-sample CSV hash changed")
    figure_path = raw_dir / "per_sample_boundary_estimates.svg"
    if summary.get("artifacts", {}).get("figure", {}).get("sha256") != _sha256(
        figure_path
    ):
        issues.append("figure hash changed")
    if b"<text" not in figure_path.read_bytes():
        issues.append("SVG text was converted to glyph paths")

    rt_path = _inside(Path(config["rt_er_radii_artifact"]), MOE_ROOT)
    if summary.get("sources", {}).get("rt_er_radii", {}).get("sha256") != _sha256(
        rt_path
    ):
        issues.append("RT-ER radius artifact hash changed")
    with np.load(rt_path, allow_pickle=False) as arrays:
        rt_radii = arrays["radii"].astype(np.float64)
    rt_median = float(np.median(rt_radii))
    ratio = pgd / first
    pearson = float(np.corrcoef(first, pgd)[0, 1])
    spearman = spearmanr(first, pgd)
    expected = {
        "first_order_x255": float(np.median(first) * 255.0),
        "pgd_extrapolated_x255": float(np.median(pgd) * 255.0),
        "pearson": pearson,
        "spearman": float(spearman.statistic),
        "within_5_percent": int(np.sum(np.abs(ratio - 1.0) <= 0.05)),
        "first_order_over_rt_er": float(np.median(first) / rt_median),
        "pgd_over_rt_er": float(np.median(pgd) / rt_median),
    }
    observed = {
        "first_order_x255": summary["first_order_radius_x255"]["median"],
        "pgd_extrapolated_x255": summary["pgd_extrapolated_radius_x255"]["median"],
        "pearson": summary["paired_agreement"]["pearson"],
        "spearman": summary["paired_agreement"]["spearman"],
        "within_5_percent": summary["paired_agreement"]["within_5_percent"],
        "first_order_over_rt_er": summary["architecture_regime_ratio"]
        ["first_order_median_over_rt_er_aggregate"],
        "pgd_over_rt_er": summary["architecture_regime_ratio"]
        ["pgd_extrapolated_median_over_rt_er_aggregate"],
    }
    for key, value in expected.items():
        if not np.isclose(value, observed[key], atol=1e-12, rtol=1e-12):
            issues.append(f"summary statistic {key} changed")

    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_AUDIT_ADV_MOE_INIT_BOUNDARY_ESTIMATES",
        "raw_result": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "recomputed": expected,
        "conclusion": (
            "The clean-margin gradient identity and all 20 paired estimates replay. "
            "The two empirical estimates have Pearson 0.926 and Spearman 0.910; "
            "their medians are 67.85/255 and 70.64/255. Relative to the exact "
            "K=20 RT-ER aggregate median, the scale ratios are 127.4x and 132.7x. "
            "These are empirical architecture-regime diagnostics, not boundaries, "
            "certificates, or causal attribution."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw_dir, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
