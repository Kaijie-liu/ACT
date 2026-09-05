"""Independent audit for the frozen AdvMoE first-NaN diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def audit(
    config_r1_path: Path,
    result_r1_path: Path,
    config_r2_path: Path,
    result_r2_path: Path,
    log_r2_path: Path,
    config_r3_path: Path,
    result_r3_path: Path,
    log_r3_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    config_r1 = _load(config_r1_path)
    result_r1 = _load(result_r1_path)
    config_r2 = _load(config_r2_path)
    result_r2 = _load(result_r2_path)
    config_r3 = _load(config_r3_path)
    result_r3 = _load(result_r3_path)
    workspace = Path(config_r1["workspace_boundary"]).resolve()
    repository = Path(config_r1["act_repository"]).resolve()
    source = Path(config_r1["official_source"]["repository"]).resolve()
    for path in (
        config_r1_path,
        result_r1_path,
        config_r2_path,
        result_r2_path,
        log_r2_path,
        config_r3_path,
        result_r3_path,
        log_r3_path,
        output_path,
        repository,
        source,
    ):
        path.resolve().relative_to(workspace)

    issues: list[str] = []
    if output_path.exists():
        raise FileExistsError(output_path)
    if _git(repository, "branch", "--show-current") != config_r1["required_branch"]:
        issues.append("ACT branch differs from the frozen branch")
    if _git(repository, "status", "--porcelain=v1"):
        issues.append("ACT worktree was not clean before audit output")
    if _git(source, "status", "--porcelain=v1"):
        issues.append("official source clone is dirty")
    if _git(source, "rev-parse", "HEAD") != config_r1["official_source"]["commit"]:
        issues.append("official source commit mismatch")
    if config_r2.get("predecessor", {}).get("sha256") != _sha256(result_r1_path):
        issues.append("r2 does not bind the observed r1 result")

    if result_r1.get("status") != "COMPLETED_NONFINITE_DETECTED":
        issues.append("r1 did not stop on a non-finite state")
    initial_r1 = result_r1.get("initial_router", {})
    if not initial_r1 or not all(
        group.get("all_finite") is True for group in initial_r1.values()
    ):
        issues.append("r1 router state was not wholly finite at initialization")
    expected_stages = [
        *(config_r1["stages"] * 2),
        *config_r1["stages"][:3],
    ]
    observed_stages = [row.get("stage") for row in result_r1.get("phase_log", [])]
    if observed_stages != expected_stages:
        issues.append("r1 phase sequence does not isolate the third router update")
    first = result_r1.get("first_nonfinite") or {}
    if first.get("stage") != "BEFORE_ROUTER_OPTIMIZER_STEP":
        issues.append("r1 first non-finite stage is not before router optimizer step")
    if first.get("zero_based_batch_index") != 2:
        issues.append("r1 first non-finite state is not on zero-based batch 2")
    quick = first.get("quick", {})
    expected_quick = {
        "parameters": True,
        "buffers": True,
        "gradients": False,
        "optimizer_state": True,
    }
    if quick != expected_quick:
        issues.append("r1 failure-family classification differs from expectation")
    gradient = first.get("details", {}).get("gradients", {})
    if gradient.get("elements") != 269202 or gradient.get("finite_elements") != 0:
        issues.append("r1 does not show all 269,202 router gradients as non-finite")
    if result_r1.get("final_main_without_router", {}).get("all_finite") is not True:
        issues.append("r1 main model outside the router is not finite")

    if result_r2.get("status") != "COMPLETED_AUTOGRAD_ANOMALY":
        issues.append("r2 did not stop on an autograd anomaly")
    runtime_failure = result_r2.get("execution", {}).get("runtime_failure") or {}
    if "XlogyBackward0" not in runtime_failure.get("message", ""):
        issues.append("r2 anomaly is not localized to XlogyBackward0")
    log_r2 = log_r2_path.read_text(encoding="utf-8")
    if "XlogyBackward0" not in log_r2 or "train_moe.py\", line 143" not in log_r2:
        issues.append("r2 log does not bind XlogyBackward0 to router KL construction")
    if "train_moe.py\", line 156" not in runtime_failure.get("traceback", ""):
        issues.append("r2 traceback does not bind the anomaly to router loss backward")
    forward_rows = result_r2.get("router_forward_log", [])
    if not forward_rows or not all(
        row.get("input", {}).get("all_finite") is True
        and row.get("output", {}).get("all_finite") is True
        for row in forward_rows
    ):
        issues.append("r2 has a non-finite router input or forward output before backward")
    batch_two_abs = [
        float(row["output"]["finite_max_abs"])
        for row in forward_rows
        if row.get("zero_based_batch_index") == 2
    ]
    if not batch_two_abs or max(batch_two_abs) < 1000.0:
        issues.append("r2 does not reproduce the extreme finite batch-2 logits")

    if config_r3.get("predecessor", {}).get("sha256") != _sha256(result_r2_path):
        issues.append("r3 does not bind the observed r2 result")
    if result_r3.get("status") != "COMPLETED_AUTOGRAD_ANOMALY":
        issues.append("r3 did not reproduce the autograd anomaly")
    log_r3 = log_r3_path.read_text(encoding="utf-8")
    if "XlogyBackward0" not in log_r3 or "train_moe.py\", line 143" not in log_r3:
        issues.append("r3 log does not reproduce the Xlogy router-KL failure")
    forward_r3 = result_r3.get("router_forward_log", [])
    if not forward_r3 or not all(
        row.get("output", {}).get("all_finite") is True for row in forward_r3
    ):
        issues.append("r3 has a non-finite router forward before the KL failure")
    zero_rows = [
        row
        for row in forward_r3
        if int(row.get("output", {}).get("softmax_zero_elements", 0)) > 0
    ]
    if not zero_rows:
        issues.append("r3 does not observe exact softmax underflow")
        first_zero = None
    else:
        first_zero = zero_rows[0]
        if first_zero.get("zero_based_batch_index") != 2:
            issues.append("r3 softmax underflow does not first occur on batch 2")
        earlier_rows = forward_r3[: int(first_zero.get("call_index", 0))]
        if any(
            int(row.get("output", {}).get("softmax_zero_elements", 0)) > 0
            for row in earlier_rows
        ):
            issues.append("r3 reports softmax underflow before its first-zero row")

    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": "ADV_MOE_OFFICIAL_TRAINER_FIRST_NONFINITE_DIAGNOSIS_AUDIT",
        "artifacts": {
            "config_r1": {"path": str(config_r1_path), "sha256": _sha256(config_r1_path)},
            "result_r1": {"path": str(result_r1_path), "sha256": _sha256(result_r1_path)},
            "config_r2": {"path": str(config_r2_path), "sha256": _sha256(config_r2_path)},
            "result_r2": {"path": str(result_r2_path), "sha256": _sha256(result_r2_path)},
            "log_r2": {"path": str(log_r2_path), "sha256": _sha256(log_r2_path)},
            "config_r3": {"path": str(config_r3_path), "sha256": _sha256(config_r3_path)},
            "result_r3": {"path": str(result_r3_path), "sha256": _sha256(result_r3_path)},
            "log_r3": {"path": str(log_r3_path), "sha256": _sha256(log_r3_path)},
        },
        "act": {
            "branch": _git(repository, "branch", "--show-current"),
            "head": _git(repository, "rev-parse", "HEAD"),
        },
        "official_source": {
            "commit": _git(source, "rev-parse", "HEAD"),
            "tree": _git(source, "rev-parse", "HEAD^{tree}"),
            "clean": not bool(_git(source, "status", "--porcelain=v1")),
        },
        "finding": {
            "zero_based_batch_index": first.get("zero_based_batch_index"),
            "stage": first.get("stage"),
            "router_gradient_elements": gradient.get("elements"),
            "finite_router_gradient_elements": gradient.get("finite_elements"),
            "parameters_finite_before_step": quick.get("parameters"),
            "buffers_finite_before_step": quick.get("buffers"),
            "optimizer_state_finite_before_step": quick.get("optimizer_state"),
            "autograd_operation": "XlogyBackward0",
            "all_router_forwards_finite": bool(forward_rows)
            and all(row["output"]["all_finite"] for row in forward_rows),
            "maximum_absolute_finite_router_logit": max(batch_two_abs)
            if batch_two_abs
            else None,
            "first_softmax_underflow_call": (
                int(first_zero["call_index"]) if first_zero is not None else None
            ),
            "first_softmax_underflow_zero_elements": (
                int(first_zero["output"]["softmax_zero_elements"])
                if first_zero is not None
                else None
            ),
            "first_softmax_underflow_pair_gap": (
                float(first_zero["output"]["maximum_pair_gap"])
                if first_zero is not None
                else None
            ),
        },
        "issues": issues,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-r1", type=Path, required=True)
    parser.add_argument("--result-r1", type=Path, required=True)
    parser.add_argument("--config-r2", type=Path, required=True)
    parser.add_argument("--result-r2", type=Path, required=True)
    parser.add_argument("--log-r2", type=Path, required=True)
    parser.add_argument("--config-r3", type=Path, required=True)
    parser.add_argument("--result-r3", type=Path, required=True)
    parser.add_argument("--log-r3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(
        arguments.config_r1,
        arguments.result_r1,
        arguments.config_r2,
        arguments.result_r2,
        arguments.log_r2,
        arguments.config_r3,
        arguments.result_r3,
        arguments.log_r3,
        arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
