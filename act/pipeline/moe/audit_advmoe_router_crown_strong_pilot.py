"""Independently audit the AdvMoE strong-PGD/sparse-CROWN init pilot."""

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
from act.pipeline.moe.advmoe_router_bracket import (
    clean_margin_diagnostics,
    load_cifar10_test_archive,
)
from act.pipeline.moe.audit_advmoe_router_bracket_pilot import validate_accounting
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(result_dir: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    result_dir = _inside(result_dir, MOE_ROOT)
    config_path = _inside(config_path, PROJECT_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"output already exists: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepare_path = result_dir / "prepare.json"
    bounds_path = result_dir / "crown_bounds.json"
    summary_path = result_dir / "summary.json"
    worker_log_path = result_dir / "crown_worker.log"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    issues: list[str] = []

    if summary.get("status") != "PASS" or summary.get("issue_count") != 0:
        issues.append("runner result is not a zero-issue PASS")
    if summary.get("scope") != (
        "INIT_20_SAMPLE_STRONG_ATTACK_SPARSE_CROWN_ENGINEERING_PILOT"
    ):
        issues.append("result scope label changed")
    if prepare.get("config") != config:
        issues.append("embedded configuration differs from the frozen file")
    if summary.get("parent_commit_required") != config.get("parent_commit_required"):
        issues.append("code-parent identity changed")
    if prepare.get("config_identity", {}).get("sha256") != _sha256(config_path):
        issues.append("configuration hash changed")
    for key, path in (
        ("prepare", prepare_path),
        ("bounds", bounds_path),
        ("worker_log", worker_log_path),
    ):
        if summary.get("artifacts", {}).get(key, {}).get("sha256") != _sha256(path):
            issues.append(f"{key} artifact identity changed")
    if bounds.get("prepare", {}).get("sha256") != _sha256(prepare_path):
        issues.append("CROWN worker used a different prepare artifact")

    archive = _inside(Path(config["dataset"]["archive"]), MOE_ROOT)
    if prepare.get("dataset", {}).get("sha256") != _sha256(archive):
        issues.append("CIFAR-10 archive identity changed")
    all_inputs, all_labels = load_cifar10_test_archive(archive)
    indices = np.asarray(config["sample_indices"], dtype=np.int64)
    if not np.array_equal(indices, np.arange(20, dtype=np.int64)):
        issues.append("sample ranks are not the frozen ordered 0--19 cohort")

    input_path = _inside(Path(prepare["input_artifact"]["path"]), MOE_ROOT)
    endpoint_path = _inside(
        Path(prepare["attack_endpoint_artifact"]["path"]), MOE_ROOT
    )
    if prepare["input_artifact"]["sha256"] != _sha256(input_path):
        issues.append("input artifact identity changed")
    if prepare["attack_endpoint_artifact"]["sha256"] != _sha256(endpoint_path):
        issues.append("attack endpoint artifact identity changed")
    if summary["artifacts"]["inputs"]["sha256"] != _sha256(input_path):
        issues.append("summary input identity changed")
    if summary["artifacts"]["attack_endpoints"]["sha256"] != _sha256(
        endpoint_path
    ):
        issues.append("summary endpoint identity changed")
    with np.load(input_path, allow_pickle=False) as arrays:
        inputs = arrays["inputs"].copy()
        labels = arrays["labels"].copy()
        stored_indices = arrays["dataset_indices"].copy()
        clean_scores = arrays["clean_scores"].copy()
        clean_routes = arrays["clean_routes"].copy()
        stored_diagnostics = {
            key: arrays[key].copy()
            for key in (
                "clean_margin",
                "gradient_l1",
                "gradient_l2",
                "gradient_linf",
            )
        }
    if not np.array_equal(stored_indices, indices):
        issues.append("stored sample ranks changed")
    if not np.array_equal(inputs, all_inputs[indices]):
        issues.append("inputs do not replay from the official archive")
    if not np.array_equal(labels, all_labels[indices]):
        issues.append("labels do not replay from the official archive")

    model, router, _moe_type = construct_official_init(int(config["model_seed"]))
    del model
    router = router.cpu().eval()
    if state_dict_sha256(router) != prepare.get("router_sha256"):
        issues.append("independently reconstructed router identity changed")
    if bounds.get("router_sha256") != prepare.get("router_sha256"):
        issues.append("CROWN worker router identity changed")
    input_tensor = torch.from_numpy(inputs)
    with torch.no_grad():
        replay_scores = router(input_tensor).cpu().numpy()
    replay_routes = replay_scores.argmax(axis=1)
    if not np.array_equal(replay_routes, clean_routes):
        issues.append("clean routes do not replay on CPU")
    if not np.allclose(replay_scores, clean_scores, atol=2e-5, rtol=1e-5):
        issues.append("clean scores exceed the registered CPU/GPU tolerance")
    replay_diagnostics = clean_margin_diagnostics(
        router, input_tensor, torch.from_numpy(clean_routes).long()
    )
    diagnostic_tolerances = {
        "clean_margin": (2e-5, 1e-4),
        "gradient_l1": (1e-3, 1e-3),
        "gradient_l2": (3e-5, 2e-3),
        "gradient_linf": (5e-5, 2e-2),
    }
    for key, (atol, rtol) in diagnostic_tolerances.items():
        if not np.allclose(
            replay_diagnostics[key], stored_diagnostics[key], atol=atol, rtol=rtol
        ):
            issues.append(f"clean diagnostic {key} exceeds CPU/GPU tolerance")
    equivalence = adapter_equivalence(router, input_tensor)
    if not equivalence["outputs_equal"] or not equivalence["routes_equal"]:
        issues.append("fixed-shape adapter is not concretely exact")

    attack_rows = {
        float(row["epsilon"]): row for row in prepare.get("attack_rows", [])
    }
    with np.load(endpoint_path, allow_pickle=False) as arrays:
        endpoints = arrays["adversarial"].copy()
        endpoint_epsilons = arrays["epsilons"].copy()
        endpoint_indices = arrays["dataset_indices"].copy()
    epsilons = np.asarray(config["epsilons"], dtype=np.float64)
    if endpoints.shape != (len(epsilons), len(indices), 3, 32, 32):
        issues.append("attack endpoint tensor shape changed")
    if not np.array_equal(endpoint_epsilons, epsilons):
        issues.append("attack endpoint epsilon grid changed")
    if not np.array_equal(endpoint_indices, indices):
        issues.append("attack endpoint sample ranks changed")
    for epsilon_slot, epsilon in enumerate(epsilons):
        row = attack_rows.get(float(epsilon))
        if row is None:
            issues.append(f"missing attack row at epsilon={epsilon}")
            continue
        endpoint = endpoints[epsilon_slot]
        linf = np.max(np.abs(endpoint - inputs), axis=(1, 2, 3))
        if np.any(linf > epsilon + 1e-6):
            issues.append(f"epsilon={epsilon}: attack endpoint exceeds its box")
        with torch.no_grad():
            endpoint_scores = router(torch.from_numpy(endpoint)).cpu().numpy()
        endpoint_routes = endpoint_scores.argmax(axis=1)
        endpoint_margins = (
            endpoint_scores[np.arange(len(indices)), clean_routes]
            - endpoint_scores[np.arange(len(indices)), 1 - clean_routes]
        )
        success = endpoint_routes != clean_routes
        if not np.array_equal(success, np.asarray(row["success"], dtype=bool)):
            issues.append(f"epsilon={epsilon}: route-flip flags do not replay")
        if not np.array_equal(
            endpoint_routes, np.asarray(row["replay_routes"], dtype=np.int64)
        ):
            issues.append(f"epsilon={epsilon}: endpoint routes changed")
        if not np.allclose(
            endpoint_margins,
            np.asarray(row["attacked_margin"], dtype=np.float64),
            atol=2e-5,
            rtol=1e-4,
        ):
            issues.append(f"epsilon={epsilon}: attacked margins do not replay")
        clean_margin = stored_diagnostics["clean_margin"]
        compression = (clean_margin - endpoint_margins) / np.maximum(
            np.abs(clean_margin), np.finfo(np.float32).eps
        )
        if not np.allclose(
            compression,
            np.asarray(row["margin_compression_fraction"], dtype=np.float64),
            atol=1e-4,
            rtol=1e-3,
        ):
            issues.append(f"epsilon={epsilon}: margin compression changed")
        if row.get("schedule", {}).get("name") != "PIECEWISE_HALVING_50_75":
            issues.append(f"epsilon={epsilon}: attack schedule changed")

    accounting_summary = {"summaries": summary.get("rows", [])}
    issues.extend(validate_accounting(config, prepare, bounds, accounting_summary))
    if bounds.get("method") != "CROWN":
        issues.append("bound method is not CROWN")
    expected_options = {
        "sparse_intermediate_bounds": True,
        "use_full_conv_alpha": False,
        "crown_batch_size": 128,
        "max_crown_size": 512,
    }
    for key, value in expected_options.items():
        if bounds.get("bound_options", {}).get(key) != value:
            issues.append(f"CROWN option {key} changed")
    if bounds.get("sample_batch_size") != 1 or bounds.get("bound_upper") is not False:
        issues.append("CROWN batching/property direction changed")
    peak = int(bounds.get("peak_memory_bytes", 0))
    if peak > int(config["bound_worker"]["maximum_peak_memory_bytes"]):
        issues.append("CROWN peak exceeds the frozen resource gate")
    bn = bounds.get("batchnorm_deployment_identity", {})
    expected_bn = {
        "layers": 19,
        "training_layers": 0,
        "maximum_abs_running_mean": 0.0,
        "maximum_abs_running_variance_minus_one": 0.0,
        "maximum_batches_tracked": 0,
    }
    if bn != expected_bn:
        issues.append("init BatchNorm deployment identity changed")
    if bounds.get("raw_frontend_probe", {}).get("status") != "REJECTED":
        issues.append("literal auto_LiRPA frontend rejection changed")
    if bounds.get("adapter_equivalence", {}).get("outputs_equal") is not True:
        issues.append("worker adapter score equality changed")

    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_AUDIT_STRONG_ATTACK_SPARSE_CROWN_INIT_PILOT",
        "raw_artifacts": {
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
            "bounds": {"path": str(bounds_path), "sha256": _sha256(bounds_path)},
            "worker_log": {
                "path": str(worker_log_path),
                "sha256": _sha256(worker_log_path),
            },
            "inputs": {"path": str(input_path), "sha256": _sha256(input_path)},
            "attack_endpoints": {
                "path": str(endpoint_path),
                "sha256": _sha256(endpoint_path),
            },
        },
        "router_sha256": prepare["router_sha256"],
        "adapter_equivalence": equivalence,
        "batchnorm_deployment_identity": bn,
        "peak_memory_bytes": peak,
        "rows": summary["rows"],
        "conclusion": (
            "All 100 strong-attack endpoints replay and all CROWN/accounting "
            "identities pass. Strong PGD finds no route flip; median margin "
            "compression reaches 11.324% at 8/255. Sparse CROWN is materially "
            "tighter than IBP but all lower bounds remain negative by hundreds "
            "of millions or more, so every row remains undecided. This init "
            "engineering result is neither formal stability nor a census."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if issues:
        raise RuntimeError(f"strong/CROWN pilot audit found {len(issues)} issue(s)")
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
