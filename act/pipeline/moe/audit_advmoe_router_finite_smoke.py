"""Independent audit of the AdvMoE softmax-underflow bridge smoke."""

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
    output_path: Path,
) -> dict[str, Any]:
    config_r1 = _load(config_r1_path)
    result_r1 = _load(result_r1_path)
    config_r2 = _load(config_r2_path)
    result_r2 = _load(result_r2_path)
    workspace = Path(config_r2["workspace_boundary"]).resolve()
    repository = Path(config_r2["act_repository"]).resolve()
    source = Path(config_r2["official_source"]["repository"]).resolve()
    for path in (
        config_r1_path,
        result_r1_path,
        config_r2_path,
        result_r2_path,
        output_path,
        repository,
        source,
    ):
        path.resolve().relative_to(workspace)
    if output_path.exists():
        raise FileExistsError(output_path)

    issues: list[str] = []
    if _git(repository, "branch", "--show-current") != config_r2["required_branch"]:
        issues.append("ACT branch mismatch")
    if _git(repository, "status", "--porcelain=v1"):
        issues.append("ACT worktree was dirty before audit output")
    if _git(source, "status", "--porcelain=v1"):
        issues.append("official source clone is dirty")
    if _git(source, "rev-parse", "HEAD") != config_r2["official_source"]["commit"]:
        issues.append("official source commit mismatch")
    if config_r2.get("predecessor", {}).get("sha256") != _sha256(result_r1_path):
        issues.append("r2 does not bind the preserved r1 failure")

    if result_r1.get("status") != "COMPLETED_AUTOGRAD_ANOMALY":
        issues.append("r1 is not the expected anomaly-mode failure")
    bridge_r1 = result_r1.get("softmax_underflow_gradient_bridge") or {}
    if bridge_r1.get("replaced_elements") != 0:
        issues.append("r1 unexpectedly applied the bridge before anomaly preemption")
    if config_r1.get("autograd_anomaly_detection") is not True:
        issues.append("r1 does not record anomaly detection")

    expected_steps = int(config_r2["maximum_batches"])
    execution = result_r2.get("execution", {})
    if result_r2.get("status") != "COMPLETED_NO_NONFINITE_WITHIN_BUDGET":
        issues.append("r2 did not finish its finite-state budget")
    if execution.get("completed_main_steps") != expected_steps:
        issues.append("r2 main-step count mismatch")
    if execution.get("completed_router_steps") != expected_steps:
        issues.append("r2 router-step count mismatch")
    if execution.get("runtime_failure") is not None:
        issues.append("r2 reports a runtime failure")
    phase_log = result_r2.get("phase_log", [])
    if len(phase_log) != expected_steps * len(config_r2["stages"]):
        issues.append("r2 phase-log length mismatch")
    if not phase_log or not all(row.get("all_finite") is True for row in phase_log):
        issues.append("r2 contains a non-finite phase")
    initial = result_r2.get("initial_router", {})
    final = result_r2.get("final_router", {})
    if not initial or not all(group.get("all_finite") is True for group in initial.values()):
        issues.append("r2 initial router state is not wholly finite")
    if not final or not all(group.get("all_finite") is True for group in final.values()):
        issues.append("r2 final router state is not wholly finite")
    bridge_r2 = result_r2.get("softmax_underflow_gradient_bridge") or {}
    if int(bridge_r2.get("replaced_elements", 0)) <= 0:
        issues.append("r2 did not exercise the underflow bridge")
    forward_log = result_r2.get("router_forward_log", [])
    if not forward_log or not all(
        row.get("input", {}).get("all_finite") is True
        and row.get("output", {}).get("all_finite") is True
        for row in forward_log
    ):
        issues.append("r2 contains a non-finite router forward")
    maximum_gap = max(
        (float(row["output"].get("maximum_pair_gap", 0.0)) for row in forward_log),
        default=0.0,
    )
    if maximum_gap <= 320.0:
        issues.append("r2 does not cross the diagnosed extreme-logit regime")

    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": "ADV_MOE_SOFTMAX_UNDERFLOW_BRIDGE_FINITE_SMOKE_AUDIT",
        "label": config_r2["label"],
        "artifacts": {
            "config_r1": {"path": str(config_r1_path), "sha256": _sha256(config_r1_path)},
            "result_r1": {"path": str(result_r1_path), "sha256": _sha256(result_r1_path)},
            "config_r2": {"path": str(config_r2_path), "sha256": _sha256(config_r2_path)},
            "result_r2": {"path": str(result_r2_path), "sha256": _sha256(result_r2_path)},
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
        "smoke": {
            "main_steps": execution.get("completed_main_steps"),
            "router_steps": execution.get("completed_router_steps"),
            "finite_phase_checks": len(phase_log),
            "bridge_replaced_elements": bridge_r2.get("replaced_elements"),
            "bridge_gradient_hook_calls": bridge_r2.get("gradient_hook_calls"),
            "maximum_router_pair_gap": maximum_gap,
            "final_router_parameter_elements": final.get("parameters", {}).get("elements"),
            "final_router_gradient_elements": final.get("gradients", {}).get("elements"),
            "final_router_optimizer_elements": final.get("optimizer_state", {}).get("elements"),
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
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(
        arguments.config_r1,
        arguments.result_r1,
        arguments.config_r2,
        arguments.result_r2,
        arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
