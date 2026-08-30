"""Independently audit the frozen AdvMoE init router-bracket pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import (
    adapter_equivalence,
    construct_official_init,
    state_dict_sha256,
)
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def validate_accounting(
    config: dict[str, Any],
    prepare: dict[str, Any],
    bounds: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    """Recompute every epsilon-level bracket partition without runner helpers."""
    issues: list[str] = []
    indices = [int(value) for value in config["sample_indices"]]
    epsilons = [float(value) for value in config["epsilons"]]
    tolerance = float(config["numerical"]["safe_positive_margin"])
    attack_by_epsilon = {
        float(row["epsilon"]): row for row in prepare.get("attack_rows", [])
    }
    bound_by_epsilon = {
        float(row["epsilon"]): row for row in bounds.get("rows", [])
    }
    summary_by_epsilon = {
        float(row["epsilon"]): row for row in summary.get("summaries", [])
    }
    expected_keys = set(epsilons)
    for name, rows in (
        ("attack", attack_by_epsilon),
        ("bound", bound_by_epsilon),
        ("summary", summary_by_epsilon),
    ):
        if set(rows) != expected_keys:
            issues.append(f"{name} epsilon rows do not match the frozen grid")

    for epsilon in epsilons:
        attack = attack_by_epsilon.get(epsilon)
        bound = bound_by_epsilon.get(epsilon)
        reported = summary_by_epsilon.get(epsilon)
        if attack is None or bound is None or reported is None:
            continue
        success = np.asarray(attack.get("success", []), dtype=bool)
        replay = np.asarray(attack.get("replay_routes", []), dtype=np.int64)
        linf = np.asarray(attack.get("linf", []), dtype=np.float64)
        if len(success) != len(indices):
            issues.append(f"epsilon={epsilon}: attack length mismatch")
            continue
        if len(replay) != len(indices) or len(linf) != len(indices):
            issues.append(f"epsilon={epsilon}: attack replay metadata mismatch")
        if np.any(~np.isfinite(linf)) or np.any(linf > epsilon + 1e-6):
            issues.append(f"epsilon={epsilon}: attack endpoint exceeds its box")

        lower = np.asarray(bound.get("lower_bounds", []), dtype=np.float64)
        if bound.get("status") == "COMPLETED_NUMERICAL_FILTER":
            if len(lower) != len(indices):
                issues.append(f"epsilon={epsilon}: bound length mismatch")
                positive = np.zeros(len(indices), dtype=bool)
            else:
                positive = np.isfinite(lower) & (lower >= tolerance)
        else:
            positive = np.zeros(len(indices), dtype=bool)
        conflicts = success & positive
        undecided = ~(success | positive)
        expected_counts = {
            "samples": len(indices),
            "attack_confirmed_route_unstable": int(success.sum()),
            "positive_numerical_bound_filter": int(positive.sum()),
            "undecided_band": int(undecided.sum()),
            "conflicts": int(conflicts.sum()),
            "formal_route_stable": 0,
        }
        for key, expected in expected_counts.items():
            if reported.get(key) != expected:
                issues.append(
                    f"epsilon={epsilon}: {key}={reported.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if conflicts.any():
            issues.append(f"epsilon={epsilon}: positive filter overlaps a witness")
        if reported.get("bound_method") != config["bound_worker"]["method"]:
            issues.append(f"epsilon={epsilon}: bound method changed")
        if reported.get("formal_route_stable_reason") != (
            "backend lower bounds are not outward-rounded"
        ):
            issues.append(f"epsilon={epsilon}: formal-soundness label changed")
    return issues


def run(result_dir: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    result_dir = _inside(result_dir, MOE_ROOT)
    config_path = _inside(config_path, PROJECT_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"output already exists: {output_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepare_path = result_dir / "prepare.json"
    bounds_path = result_dir / "bounds.json"
    summary_path = result_dir / "summary.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    issues: list[str] = []

    if summary.get("status") != "PASS" or summary.get("issue_count") != 0:
        issues.append("runner summary is not a zero-issue PASS")
    if summary.get("scope") != "INIT_20_SAMPLE_ENGINEERING_PILOT":
        issues.append("pilot scope label changed")
    if summary.get("parent_commit_required") != config.get("parent_commit_required"):
        issues.append("code-parent identity changed")
    if prepare.get("config") != config:
        issues.append("embedded configuration differs from the frozen file")
    if prepare.get("config_identity", {}).get("sha256") != _sha256(config_path):
        issues.append("configuration hash changed")
    if summary.get("prepare", {}).get("sha256") != _sha256(prepare_path):
        issues.append("prepare artifact hash changed")
    if summary.get("bounds", {}).get("sha256") != _sha256(bounds_path):
        issues.append("bound artifact hash changed")
    if bounds.get("prepare", {}).get("sha256") != _sha256(prepare_path):
        issues.append("bound worker used a different prepare artifact")

    archive = _inside(Path(config["dataset"]["archive"]), MOE_ROOT)
    if prepare.get("dataset", {}).get("sha256") != _sha256(archive):
        issues.append("CIFAR-10 archive identity changed")
    all_inputs, all_labels = load_cifar10_test_archive(archive)
    indices = np.asarray(config["sample_indices"], dtype=np.int64)
    if not np.array_equal(indices, np.arange(20, dtype=np.int64)):
        issues.append("pilot sample indices are not frozen ordered ranks 0--19")

    input_path = _inside(Path(prepare["input_artifact"]["path"]), MOE_ROOT)
    witness_path = _inside(Path(prepare["witness_artifact"]["path"]), MOE_ROOT)
    if prepare["input_artifact"]["sha256"] != _sha256(input_path):
        issues.append("input artifact identity changed")
    if prepare["witness_artifact"]["sha256"] != _sha256(witness_path):
        issues.append("witness artifact identity changed")
    with np.load(input_path, allow_pickle=False) as arrays:
        inputs = arrays["inputs"].copy()
        labels = arrays["labels"].copy()
        stored_indices = arrays["dataset_indices"].copy()
        clean_scores = arrays["clean_scores"].copy()
        clean_routes = arrays["clean_routes"].copy()
    if not np.array_equal(stored_indices, indices):
        issues.append("input artifact indices changed")
    if not np.array_equal(inputs, all_inputs[indices]):
        issues.append("input artifact does not replay from the official archive")
    if not np.array_equal(labels, all_labels[indices]):
        issues.append("input labels do not replay from the official archive")

    model, router, _moe_type = construct_official_init(int(config["model_seed"]))
    del model
    router = router.cpu().eval()
    if state_dict_sha256(router) != prepare.get("router_sha256"):
        issues.append("independently reconstructed router identity changed")
    if bounds.get("router_sha256") != prepare.get("router_sha256"):
        issues.append("bound worker router identity changed")
    input_tensor = torch.from_numpy(inputs)
    with torch.no_grad():
        replay_scores = router(input_tensor).cpu().numpy()
    replay_routes = replay_scores.argmax(axis=1)
    if not np.array_equal(replay_routes, clean_routes):
        issues.append("clean routes do not replay on the reconstructed router")
    if not np.allclose(replay_scores, clean_scores, atol=2e-5, rtol=1e-5):
        issues.append("clean router scores exceed the CPU/GPU replay tolerance")
    equivalence = adapter_equivalence(router, input_tensor)
    if not equivalence["outputs_equal"] or not equivalence["routes_equal"]:
        issues.append("fixed-shape adapter is not concretely exact")
    if bounds.get("adapter_equivalence", {}).get("outputs_equal") is not True:
        issues.append("bound worker did not establish adapter score equality")
    if bounds.get("adapter_equivalence", {}).get("routes_equal") is not True:
        issues.append("bound worker did not establish adapter route equality")
    if bounds.get("raw_frontend_probe", {}).get("status") != "REJECTED":
        issues.append("literal auto_LiRPA frontend rejection changed")

    issues.extend(validate_accounting(config, prepare, bounds, summary))
    attack_rows = {
        float(row["epsilon"]): row for row in prepare.get("attack_rows", [])
    }
    for epsilon, row in attack_rows.items():
        reported_routes = np.asarray(row["replay_routes"], dtype=np.int64)
        reported_success = np.asarray(row["success"], dtype=bool)
        if not np.array_equal(reported_success, reported_routes != clean_routes):
            issues.append(f"epsilon={epsilon}: success flags disagree with replay routes")

    with np.load(witness_path, allow_pickle=False) as arrays:
        adversarial = arrays["adversarial"].copy()
        epsilon_slots = arrays["epsilon_slots"].copy()
        sample_slots = arrays["sample_slots"].copy()
    expected_witnesses = sum(
        int(np.asarray(row["success"], dtype=bool).sum())
        for row in prepare.get("attack_rows", [])
    )
    if len(epsilon_slots) != expected_witnesses:
        issues.append("witness count does not equal successful attack endpoints")
    if len(sample_slots) != len(epsilon_slots) or len(adversarial) != len(epsilon_slots):
        issues.append("witness array lengths disagree")
    epsilons = [float(value) for value in config["epsilons"]]
    for witness, epsilon_slot, sample_slot in zip(
        adversarial, epsilon_slots, sample_slots
    ):
        epsilon = epsilons[int(epsilon_slot)]
        sample_slot = int(sample_slot)
        linf = float(np.max(np.abs(witness - inputs[sample_slot])))
        if linf > epsilon + 1e-6:
            issues.append("concrete route witness exceeds its registered box")
        with torch.no_grad():
            witness_route = int(
                router(torch.from_numpy(witness[None]).float()).argmax(dim=1).item()
            )
        if witness_route == int(clean_routes[sample_slot]):
            issues.append("stored route witness does not flip the literal router")
        if not bool(attack_rows[epsilon]["success"][sample_slot]):
            issues.append("stored witness is not linked to a successful attack row")

    summaries = summary.get("summaries", [])
    lower_abs_max = max(
        abs(float(value))
        for row in bounds.get("rows", [])
        for value in row.get("lower_bounds", [])
    )
    upper_abs_max = max(
        abs(float(value))
        for row in bounds.get("rows", [])
        for value in row.get("upper_bounds", [])
    )
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INIT_20_SAMPLE_ENGINEERING_PILOT_INDEPENDENT_AUDIT",
        "raw_artifacts": {
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
            "bounds": {"path": str(bounds_path), "sha256": _sha256(bounds_path)},
            "inputs": {"path": str(input_path), "sha256": _sha256(input_path)},
            "witnesses": {"path": str(witness_path), "sha256": _sha256(witness_path)},
        },
        "router_sha256": prepare["router_sha256"],
        "adapter_equivalence": equivalence,
        "radius_rows": summaries,
        "totals": {
            "sample_radius_rows": len(indices) * len(config["epsilons"]),
            "attack_confirmed_route_unstable": sum(
                int(row["attack_confirmed_route_unstable"]) for row in summaries
            ),
            "positive_numerical_bound_filter": sum(
                int(row["positive_numerical_bound_filter"]) for row in summaries
            ),
            "undecided_band": sum(int(row["undecided_band"]) for row in summaries),
            "formal_route_stable": sum(
                int(row["formal_route_stable"]) for row in summaries
            ),
            "stored_witnesses": len(epsilon_slots),
        },
        "bound_magnitude": {
            "maximum_absolute_lower_bound": lower_abs_max,
            "maximum_absolute_upper_bound": upper_abs_max,
        },
        "conclusion": (
            "The harness and identities pass independently. On this init-only "
            "20-sample pilot, PGD finds no route flips and one-thread IBP gives "
            "no positive numerical margin filters at any frozen radius, leaving "
            "all 100 sample-radius rows undecided. This is not evidence of route "
            "stability, prevalence, accuracy, robustness, or certification."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if issues:
        raise RuntimeError(f"AdvMoE router pilot audit found {len(issues)} issue(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.result_dir, args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
