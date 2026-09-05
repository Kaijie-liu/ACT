"""Summarize the audited AdvMoE Lagrangian development comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


WORKSPACE = Path("/data1/Kane/MOE")
POSITIVE = "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(WORKSPACE)
    return resolved


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "q25": None, "median": None,
                "q75": None, "p90": None, "maximum": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _paired_counts(left: list[bool], right: list[bool]) -> dict[str, int]:
    return {
        "both_negative": sum(not a and not b for a, b in zip(left, right)),
        "right_only": sum(not a and b for a, b in zip(left, right)),
        "left_only": sum(a and not b for a, b in zip(left, right)),
        "both_positive": sum(a and b for a, b in zip(left, right)),
    }


def _cluster_bootstrap(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(20260906)
    draws = generator.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return {
        "unit": "input sample with five radii retained as one cluster",
        "clusters": int(array.size),
        "point_estimate": float(array.mean()),
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260906,
        "percentile_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def analyze(config_path: Path, summary_path: Path, audit_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path)
    summary_path = _inside(summary_path)
    audit_path = _inside(audit_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("issues") != []:
        raise RuntimeError("development result has not passed independent audit")
    if summary.get("config", {}).get("sha256") != _sha256(config_path):
        raise RuntimeError("summary/config identity mismatch")
    if audit.get("result", {}).get("sha256") != _sha256(summary_path):
        raise RuntimeError("audit/summary identity mismatch")
    rows_path = _inside(Path(summary["rows"]["path"]))
    if _sha256(rows_path) != summary["rows"]["sha256"]:
        raise RuntimeError("row artifact hash mismatch")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if len(rows) != 100:
        raise RuntimeError("development analysis requires all 100 frozen rows")

    tolerance = float(config["numerical"]["safe_positive_margin"])
    lag_mu0_deltas: list[float] = []
    lag_unguarded_deltas: list[float] = []
    lag_separate_deltas: list[float] = []
    selected_multipliers: list[float] = []
    for row in rows:
        for branch in range(2):
            lag = np.asarray(
                row["lagrangian_guard_crown"][branch]["lower_bounds"],
                dtype=np.float64,
            )
            mu0 = np.asarray(
                row["lagrangian_mu0_graph_matched_crown"][branch]["lower_bounds"],
                dtype=np.float64,
            )
            unguarded = np.asarray(
                row["path_crown"][branch]["lower_bounds"], dtype=np.float64
            )
            separate = np.asarray(
                row["lagrangian_separate_interval"][branch]["lower_bounds"],
                dtype=np.float64,
            )
            if not all(value.shape == (9,) for value in (lag, mu0, unguarded, separate)):
                raise RuntimeError(f"malformed property rows: {row['row_id']}")
            lag_mu0_deltas.extend((lag - mu0).tolist())
            lag_unguarded_deltas.extend((lag - unguarded).tolist())
            lag_separate_deltas.extend((lag - separate).tolist())
            selected_multipliers.extend(
                float(value)
                for value in row["lagrangian_guard_crown"][branch][
                    "selected_multipliers"
                ]
            )

    def delta_summary(values: list[float]) -> dict[str, Any]:
        return {
            "strictly_improved_over_tolerance": sum(value > tolerance for value in values),
            "equal_within_tolerance": sum(abs(value) <= tolerance for value in values),
            "strictly_worse_over_tolerance": sum(value < -tolerance for value in values),
            "delta_distribution": _distribution(values),
        }

    lag_positive = [row["statuses"]["lagrangian_guard_ablation"] == POSITIVE for row in rows]
    mu0_positive = [row["statuses"]["lagrangian_mu0_graph_matched"] == POSITIVE for row in rows]
    unguarded_positive = [row["statuses"]["route_a_two_path"] == POSITIVE for row in rows]
    separate_positive = [row["statuses"]["lagrangian_separate_interval"] == POSITIVE for row in rows]
    eta_positive = [row["statuses"]["eta_guard_ablation"] == POSITIVE for row in rows]
    route_invariance_positive = [row["statuses"]["route_invariance"] == POSITIVE for row in rows]

    budget_methods = tuple(rows[0]["comparison"]["methods"])
    budget_positive = {
        method: [
            row["comparison"]["methods"][method]["budget_status"] == POSITIVE
            for row in rows
        ]
        for method in budget_methods
    }
    cost_distributions = {
        method: _distribution(
            [
                float(row["comparison"]["methods"][method]["accounted_wall_seconds"])
                for row in rows
            ]
        )
        for method in budget_methods
    }
    budget_overshoots = {
        method: sum(
            not bool(row["comparison"]["methods"][method]["within_budget"])
            for row in rows
        )
        for method in budget_methods
    }

    sample_slots = sorted({int(row["sample_slot"]) for row in rows})
    cluster_differences = []
    for slot in sample_slots:
        cluster = [row for row in rows if int(row["sample_slot"]) == slot]
        if len(cluster) != 5:
            raise RuntimeError("each development input must retain five radii")
        cluster_differences.append(
            float(
                np.mean(
                    [
                        (row["statuses"]["lagrangian_guard_ablation"] == POSITIVE)
                        - (row["statuses"]["route_a_two_path"] == POSITIVE)
                        for row in cluster
                    ]
                )
            )
        )

    route_flip = [
        int(row["attack"]["attacked_route"]) != int(row["clean_route"])
        for row in rows
    ]
    prediction_flip = [bool(row["attack"]["prediction_flip"]) for row in rows]
    route_changing_lag_positive = sum(
        changed and filtered for changed, filtered in zip(route_flip, lag_positive)
    )
    attribution = (
        "NO_COMPLETE_COVERAGE_GAIN_UNDER_FROZEN_PROTOCOL"
        if not any(lag and not base for lag, base in zip(lag_positive, mu0_positive))
        else "COMPLETE_COVERAGE_GAIN_OBSERVED_IN_DEVELOPMENT"
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "ADV_MOE_LAGRANGIAN_DEVELOPMENT_PAIRED_ANALYSIS_R1",
        "identity": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        },
        "denominator": {
            "input_clusters": len(sample_slots),
            "sample_radius_rows": len(rows),
            "property_rows": len(lag_mu0_deltas),
            "radii_over_255": config["radii_over_255"],
            "development_not_holdout": True,
        },
        "property_row_effect": {
            "lagrangian_vs_graph_matched_mu0": delta_summary(lag_mu0_deltas),
            "lagrangian_vs_unguarded_C_matrix": delta_summary(lag_unguarded_deltas),
            "shared_graph_vs_separate_intervals": delta_summary(lag_separate_deltas),
            "selected_multiplier_counts": dict(Counter(selected_multipliers)),
            "nonzero_selected_property_rows": sum(value != 0.0 for value in selected_multipliers),
        },
        "complete_sample_radius_effect": {
            "mechanism_positive_counts": {
                "route_invariance": sum(route_invariance_positive),
                "unguarded_two_path": sum(unguarded_positive),
                "eta_guard": sum(eta_positive),
                "lagrangian_mu0_graph_matched": sum(mu0_positive),
                "lagrangian_grid": sum(lag_positive),
                "lagrangian_separate_interval": sum(separate_positive),
            },
            "lagrangian_vs_graph_matched_mu0": _paired_counts(mu0_positive, lag_positive),
            "lagrangian_vs_unguarded": _paired_counts(unguarded_positive, lag_positive),
            "lagrangian_vs_separate_interval": _paired_counts(separate_positive, lag_positive),
            "cluster_primary_lagrangian_minus_unguarded": _cluster_bootstrap(
                cluster_differences
            ),
        },
        "cost_matched_effect": {
            "common_budget_seconds": float(
                config["comparison"][
                    "total_wall_budget_seconds_per_sample_radius_method"
                ]
            ),
            "positive_counts": {
                method: sum(values) for method, values in budget_positive.items()
            },
            "budget_overshoots": budget_overshoots,
            "accounted_wall_seconds": cost_distributions,
            "lagrangian_vs_unguarded": _paired_counts(
                budget_positive["unguarded_two_path"],
                budget_positive["lagrangian_grid"],
            ),
        },
        "core_endpoint": {
            "route_flip_witness_rows": sum(route_flip),
            "prediction_flip_witness_rows": sum(prediction_flip),
            "route_changing_lagrangian_positive_filters": route_changing_lag_positive,
            "formal_safe": 0,
            "positive_semantics": "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
        },
        "attribution": {
            "outcome": attribution,
            "permitted_statement": (
                "Under the frozen multiplier protocol and common budget, the "
                "Lagrangian compiler did not add complete numerical-filter coverage."
                if attribution == "NO_COMPLETE_COVERAGE_GAIN_UNDER_FROZEN_PROTOCOL"
                else "The frozen development protocol added complete numerical-filter coverage."
            ),
            "prohibited_inference": (
                "This result alone cannot distinguish CROWN relaxation error, "
                "finite multiplier search, and intrinsic fixed-multiplier "
                "sufficient-reduction gap."
            ),
        },
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path = _inside(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = analyze(
        arguments.config, arguments.summary, arguments.audit
    )
    _write(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
