"""Fail-closed rehearsal and unattended landing for the B1 reproduction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from act.pipeline.moe.baseline_icml2025_b1_smoke import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)
from act.pipeline.moe.baseline_icml2025_b1_supervisor import _checkpoint_epoch


RT_ER_PYTHON = MOE_ROOT / "envs/rt-er-blackwell/bin/python"


class RetryableGpuLandingError(RuntimeError):
    """A transient GPU-capacity failure that the watcher may retry."""


def _artifact_subprocess_environment() -> dict[str, str]:
    """Keep read-only imports from dirtying the pinned official clone."""
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def gpu_memory_bytes(device_index: int = 0) -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    row = result.stdout.strip().splitlines()
    if len(row) != 1:
        raise RuntimeError("GPU resource query returned an unexpected row count")
    free_mib, total_mib = (int(value.strip()) for value in row[0].split(","))
    return free_mib * 1024 * 1024, total_mib * 1024 * 1024


def require_gpu_resource(protocol: dict[str, Any]) -> dict[str, int]:
    gate = protocol["gpu_resource_gate"]
    free_bytes, total_bytes = gpu_memory_bytes(int(gate["device_index"]))
    required = int(gate["minimum_free_memory_bytes"])
    if free_bytes < required:
        raise RetryableGpuLandingError(
            f"GPU resource gate: {free_bytes} free bytes < {required} required"
        )
    return {
        "free_memory_bytes": free_bytes,
        "total_memory_bytes": total_bytes,
        "minimum_free_memory_bytes": required,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_completed_epoch(
    progress: dict[str, Any], epoch: int
) -> dict[str, Any]:
    matches = [row for row in progress.get("completed", []) if row.get("epoch") == epoch]
    if len(matches) != 1:
        raise RuntimeError(f"expected one completed epoch {epoch}, found {len(matches)}")
    row = matches[0]
    checkpoint = _inside(Path(row["checkpoint"]))
    metrics = _inside(Path(row["metrics"]))
    telemetry_dir = _inside(Path(row["telemetry"]))
    telemetry_summary = telemetry_dir / "summary.json"
    for path in (checkpoint, metrics, telemetry_summary):
        if not path.is_file():
            raise RuntimeError(f"landing input is missing: {path}")
    if _sha256(checkpoint) != row.get("checkpoint_sha256"):
        raise RuntimeError("landing checkpoint hash changed")
    if _sha256(metrics) != row.get("metrics_sha256"):
        raise RuntimeError("landing metrics hash changed")
    if _sha256(telemetry_summary) != row.get("telemetry_summary_sha256"):
        raise RuntimeError("landing telemetry hash changed")
    if _checkpoint_epoch(checkpoint) != epoch:
        raise RuntimeError("landing checkpoint epoch changed")
    telemetry = json.loads(telemetry_summary.read_text(encoding="utf-8"))
    if telemetry.get("epoch") != epoch:
        raise RuntimeError("landing telemetry epoch changed")
    if telemetry.get("checkpoint", {}).get("sha256") != row.get("checkpoint_sha256"):
        raise RuntimeError("landing telemetry checkpoint identity changed")
    return {
        **row,
        "telemetry_summary": str(telemetry_summary),
        "telemetry_checkpoint_identity_passed": True,
    }


def endpoint_decisions(
    standard_accuracy: float,
    pgd50_accuracy: float,
    interpretation: dict[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    standard_rule = interpretation["primary_standard_accuracy_rule"]
    robust_rule = interpretation["secondary_pgd50_rule"]
    standard_inside = (
        float(standard_rule["inclusive_interval_percent"][0])
        <= standard_accuracy
        <= float(standard_rule["inclusive_interval_percent"][1])
    )
    robust_inside = (
        float(robust_rule["inclusive_interval_percent"][0])
        <= pgd50_accuracy
        <= float(robust_rule["inclusive_interval_percent"][1])
    )
    if int(seed) == 0:
        pipeline_claim_status = (
            "EXISTENCE_SUPPORTED_BY_SEED0_NO_SEED1_REQUIRED"
            if standard_inside
            else "SEED1_REQUIRED_BEFORE_PIPELINE_LEVEL_FAILURE_WORDING"
        )
        seed1_followup_required = not standard_inside
    elif int(seed) == 1:
        pipeline_claim_status = (
            "EXISTENCE_SUPPORTED_BY_SEED1_AFTER_SEED0_MISS"
            if standard_inside
            else "TWO_REGISTERED_RUNS_MISS_SA_INTERVAL_PIPELINE_LEVEL_WORDING_ALLOWED"
        )
        seed1_followup_required = False
    else:
        raise ValueError("B1 endpoint decisions are registered only for seeds 0 and 1")
    return {
        "seed": int(seed),
        "standard_accuracy_percent": standard_accuracy,
        "standard_accuracy_branch": standard_rule["inside"] if standard_inside else standard_rule["outside"],
        "pgd50_accuracy_percent": pgd50_accuracy,
        "pgd50_accuracy_branch": robust_rule["inside"] if robust_inside else robust_rule["outside"],
        "pipeline_claim_status": pipeline_claim_status,
        "seed1_followup_required": seed1_followup_required,
        "thresholds_changed_after_observation": False,
    }


def run_rehearsal(protocol_path: Path) -> dict[str, Any]:
    protocol_path = _inside(protocol_path, PROJECT_ROOT)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_root = _inside(Path(protocol["run_root"]))
    progress_path = run_root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    epoch = int(protocol["rehearsal_epoch"])
    completed = validate_completed_epoch(progress, epoch)
    result = {
        "schema_version": 1,
        "status": "PASSED_REHEARSAL",
        "scope": "B1_UNATTENDED_LANDING_EPOCH50_REHEARSAL",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "progress": {"path": str(progress_path), "sha256": _sha256(progress_path)},
        "completed_epoch": completed,
        "excluded_by_protocol": protocol["rehearsal_exclusions"],
        "conclusion": (
            "The unattended hook recovered and validated the immutable checkpoint, "
            "metrics, and telemetry identity chain without running final evaluation, "
            "gate interpretation, commit, or push."
        ),
    }
    output = run_root / "landing/rehearsal_epoch050/B1_LANDING_REHEARSAL.json"
    _write_json(output, result)
    return result


def _retryable_cuda_failure(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-65536:].lower()
    return any(
        marker in tail
        for marker in (
            "cuda out of memory",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
        )
    )


def _run_endpoint(
    protocol: dict[str, Any], protocol_path: Path, checkpoint: Path
) -> tuple[Path, Path, dict[str, int]]:
    run_root = _inside(Path(protocol["run_root"]))
    resource = require_gpu_resource(protocol)
    attempts_root = run_root / "landing/final_endpoint_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    prior = sorted(path for path in attempts_root.glob("attempt_*"))
    attempt_root = attempts_root / f"attempt_{len(prior) + 1:03d}"
    attempt_root.mkdir()
    endpoint_dir = attempt_root / "endpoint"
    audit_path = attempt_root / "endpoint_audit.json"
    endpoint_log = attempt_root / "endpoint.log"
    command = [
        str(RT_ER_PYTHON),
        "-m",
        "act.pipeline.moe.baseline_icml2025_b1_endpoint",
        "--protocol",
        str(protocol_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(endpoint_dir),
        "--device",
        "cuda",
    ]
    with endpoint_log.open("xb") as handle:
        try:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=_artifact_subprocess_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            if _retryable_cuda_failure(endpoint_log):
                raise RetryableGpuLandingError(
                    f"endpoint attempt {attempt_root.name} exhausted GPU memory"
                ) from error
            raise
    audit_log = attempt_root / "endpoint_audit.log"
    audit_command = [
        str(RT_ER_PYTHON),
        "-m",
        "act.pipeline.moe.audit_baseline_icml2025_b1_endpoint",
        "--protocol",
        str(protocol_path),
        "--endpoint-dir",
        str(endpoint_dir),
        "--output",
        str(audit_path),
    ]
    with audit_log.open("xb") as handle:
        try:
            subprocess.run(
                audit_command,
                cwd=PROJECT_ROOT,
                env=_artifact_subprocess_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            if _retryable_cuda_failure(audit_log):
                raise RetryableGpuLandingError(
                    f"audit attempt {attempt_root.name} exhausted GPU memory"
                ) from error
            raise
    return endpoint_dir, audit_path, resource


def _render_report(landing: dict[str, Any]) -> str:
    decisions = landing["endpoint_decisions"]
    endpoint = landing["endpoint"]
    seed = int(decisions["seed"])
    prior = landing.get("prior_seed0_landing")
    prior_line = (
        f"- Frozen seed-0 landing SHA-256: `{prior['sha256']}`\n" if prior else ""
    )
    conclusion = (
        "If seed 0 misses the SA interval, seed 1 is required before any "
        "pipeline-level failure wording."
        if seed == 0
        else "Seed 1 is the frozen follow-up required by the seed-0 miss; the "
        "pipeline-level branch follows the two registered outcomes without "
        "changing the endpoint or thresholds."
    )
    return f"""# B1 landed: official-code RT-ER seed {seed}

The frozen 130-epoch official-code, Blackwell-compatible dependency
reproduction completed and passed the unattended identity and endpoint audit.

- Ordered full-test standard accuracy: `{decisions['standard_accuracy_percent']:.4f}%`
- Ordered full-test PGD-50 accuracy: `{decisions['pgd50_accuracy_percent']:.4f}%`
- Standard-accuracy branch: `{decisions['standard_accuracy_branch']}`
- PGD-50 branch: `{decisions['pgd50_accuracy_branch']}`
- Pipeline-claim status: `{decisions['pipeline_claim_status']}`
- Full-model replayed attack endpoints: `{landing['endpoint_audit']['samples_replayed']}`
- Endpoint audit issues: `{landing['endpoint_audit']['issue_count']}`
- Epoch-130 checkpoint SHA-256: `{landing['epoch130']['checkpoint_sha256']}`
- Endpoint summary SHA-256: `{endpoint['summary_sha256']}`
{prior_line}

The original thresholds remain unchanged. Matching or missing them is a
single-seed reproduction outcome under the disclosed compatibility environment.
{conclusion} This result does not establish author-checkpoint identity,
theorem applicability, or a general claim about the paper's method.
"""


def _commit_and_push(protocol: dict[str, Any], landing: dict[str, Any]) -> dict[str, str]:
    branch = str(protocol["branch"])
    remote = str(protocol["remote"])
    if _git("branch", "--show-current") != branch:
        raise RuntimeError("landing hook is not on the feature branch")
    if _git("status", "--porcelain"):
        raise RuntimeError("landing hook requires a clean worktree")
    subprocess.run(
        ["git", "fetch", remote, branch], cwd=PROJECT_ROOT, check=True
    )
    if _git("rev-parse", "HEAD") != _git("rev-parse", f"{remote}/{branch}"):
        raise RuntimeError("landing hook refuses a local/remote branch divergence")
    seed = int(protocol.get("seed", 0))
    tracked_json = PROJECT_ROOT / (
        f"act/pipeline/moe/results/baseline/icml2025_rt_er_b1_landed_seed{seed}.json"
    )
    report = PROJECT_ROOT / f"paper/results/b1_landed_seed{seed}.md"
    _write_json(tracked_json, landing)
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        raise RuntimeError(f"refuses to overwrite {report}")
    with report.open("x", encoding="utf-8") as handle:
        handle.write(_render_report(landing))
        handle.flush()
        os.fsync(handle.fileno())
    expected = {
        f"?? {tracked_json.relative_to(PROJECT_ROOT)}",
        f"?? {report.relative_to(PROJECT_ROOT)}",
    }
    observed = set(_git("status", "--porcelain", "--untracked-files=all").splitlines())
    if observed != expected:
        raise RuntimeError(f"unexpected landing worktree changes: {sorted(observed)}")
    subprocess.run(
        ["git", "add", str(tracked_json.relative_to(PROJECT_ROOT)), str(report.relative_to(PROJECT_ROOT))],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"Record landed RT-ER B1 seed {seed} endpoint"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    push_error: subprocess.CalledProcessError | None = None
    for _attempt in range(3):
        try:
            subprocess.run(
                ["git", "push", remote, f"HEAD:{branch}"], cwd=PROJECT_ROOT, check=True
            )
            push_error = None
            break
        except subprocess.CalledProcessError as error:
            push_error = error
            time.sleep(30)
    if push_error is not None:
        raise push_error
    return {"commit": _git("rev-parse", "HEAD"), "remote": f"{remote}/{branch}"}


def run_final(protocol_path: Path) -> dict[str, Any]:
    protocol_path = _inside(protocol_path, PROJECT_ROOT)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seed = int(protocol.get("seed", 0))
    run_root = _inside(Path(protocol["run_root"]))
    supervisor_summary_path = run_root / "summary.json"
    supervisor = json.loads(supervisor_summary_path.read_text(encoding="utf-8"))
    if supervisor.get("status") != "PASSED":
        raise RuntimeError("B1 supervisor has not landed PASSED")
    expected_epochs = list(range(10, 131, 10))
    if [row.get("epoch") for row in supervisor.get("completed", [])] != expected_epochs:
        raise RuntimeError("B1 final checkpoint schedule is incomplete")
    validated_schedule = [
        validate_completed_epoch(supervisor, epoch) for epoch in expected_epochs
    ]
    epoch130 = validated_schedule[-1]
    endpoint_dir, audit_path, gpu_resource = _run_endpoint(
        protocol, protocol_path, Path(epoch130["checkpoint"])
    )
    endpoint_summary_path = endpoint_dir / "summary.json"
    endpoint = json.loads(endpoint_summary_path.read_text(encoding="utf-8"))
    endpoint_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if endpoint_audit.get("status") != "PASS" or endpoint_audit.get("issue_count") != 0:
        raise RuntimeError("B1 endpoint independent audit failed")
    interpretation_path = _inside(Path(protocol["endpoint_interpretation"]), PROJECT_ROOT)
    interpretation = json.loads(interpretation_path.read_text(encoding="utf-8"))
    prior_seed0_landing = None
    if seed == 1:
        prior_spec = protocol.get("prior_seed0_landing")
        if not isinstance(prior_spec, dict):
            raise RuntimeError("seed1 landing requires the frozen seed0 landing identity")
        prior_path = _inside(Path(prior_spec["path"]), PROJECT_ROOT)
        if _sha256(prior_path) != prior_spec.get("sha256"):
            raise RuntimeError("frozen seed0 landing identity changed")
        prior_value = json.loads(prior_path.read_text(encoding="utf-8"))
        if not prior_value.get("endpoint_decisions", {}).get("seed1_followup_required"):
            raise RuntimeError("seed0 landing did not require the registered seed1 follow-up")
        prior_seed0_landing = {"path": str(prior_path), "sha256": _sha256(prior_path)}
    decisions = endpoint_decisions(
        float(endpoint["standard_accuracy_percent"]),
        float(endpoint["pgd50_accuracy_percent"]),
        interpretation,
        seed=seed,
    )
    rehearsal_path = run_root / "landing/rehearsal_epoch050/B1_LANDING_REHEARSAL.json"
    rehearsal_failure_path = run_root / "landing/rehearsal_epoch050/REHEARSAL_FAILED.json"
    rehearsal = {
        "status": "PASSED" if rehearsal_path.is_file() else "FAILED_OR_MISSING",
        "path": str(rehearsal_path if rehearsal_path.is_file() else rehearsal_failure_path),
        "sha256": (
            _sha256(rehearsal_path)
            if rehearsal_path.is_file()
            else (_sha256(rehearsal_failure_path) if rehearsal_failure_path.is_file() else None)
        ),
    }
    landing = {
        "schema_version": 1,
        "status": "PASSED",
        "scope": f"B1_LANDED_OFFICIAL_RT_ER_SEED{seed}",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "supervisor_summary": {
            "path": str(supervisor_summary_path),
            "sha256": _sha256(supervisor_summary_path),
        },
        "epoch130": epoch130,
        "validated_checkpoint_schedule": validated_schedule,
        "rehearsal": rehearsal,
        "gpu_resource_gate": gpu_resource,
        "endpoint": {
            "summary": str(endpoint_summary_path),
            "summary_sha256": _sha256(endpoint_summary_path),
            "artifact_sha256": endpoint["artifact"]["sha256"],
        },
        "endpoint_audit": endpoint_audit,
        "endpoint_interpretation": {
            "path": str(interpretation_path),
            "sha256": _sha256(interpretation_path),
        },
        "endpoint_decisions": decisions,
        "prior_seed0_landing": prior_seed0_landing,
        "thresholds_modified": False,
        "generated_unix_seconds": time.time(),
    }
    raw_landing = run_root / "landing/B1_LANDED_summary.json"
    _write_json(raw_landing, landing)
    git_result = _commit_and_push(protocol, landing)
    completion = {**landing, "landing_git": git_result}
    _write_json(run_root / "landing/B1_LANDED_completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("rehearsal", "final"), required=True)
    args = parser.parse_args()
    result = run_rehearsal(args.protocol) if args.mode == "rehearsal" else run_final(args.protocol)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
