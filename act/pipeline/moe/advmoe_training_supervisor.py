from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import torch


GIB = 1024**3


class NonfiniteCheckpointError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def build_training_command(config: Mapping[str, Any]) -> list[str]:
    run = config["run"]
    source = Path(config["official_source"]["repository"])
    environment = Path(config["environment"]["path"])
    official_arguments = [
        "--dataset",
        run["dataset"],
        "--arch",
        run["arch"],
        "--n-expert",
        str(run["n_expert"]),
        "--ratio",
        str(run["ratio"]),
        "--seed",
        str(run["seed"]),
        "--epochs",
        str(run["epochs"]),
        "--batch-size",
        str(run["batch_size"]),
        "--test-batch-size",
        str(run["test_batch_size"]),
        "--num-workers",
        str(run["num_workers_argument"]),
        "--optimizer",
        run["optimizer"],
        "--momentum",
        str(run["momentum"]),
        "--wd",
        str(run["weight_decay"]),
        "--lr",
        str(run["learning_rate"]),
        "--lr-schedule",
        run["learning_rate_schedule"],
        "--router-optimizer",
        run["router_optimizer_argument"],
        "--router-lr",
        str(run["router_learning_rate_argument"]),
        "--router-lr-schedule",
        run["router_learning_rate_schedule_argument"],
        "--epsilon",
        str(run["epsilon"]),
        "--num-steps",
        str(run["train_attack_steps"]),
        "--step-size",
        str(run["train_attack_step_size"]),
        "--epsilon-test",
        str(run["test_epsilon"]),
        "--num-steps-test",
        str(run["test_attack_steps"]),
        "--step-size-test",
        str(run["test_attack_step_size"]),
        "--alpha",
        str(run["alpha"]),
        "--beta",
        str(run["beta"]),
        "--data-dir",
        config["dataset"]["run_data_root"],
        "--exp-identifier",
        run.get("experiment_identifier", "official_seed0_r1"),
    ]
    compatibility = config.get("compatibility_variant")
    if compatibility is None:
        return [
            str(environment / "bin" / "python"),
            str(source / "train_moe.py"),
            *official_arguments,
        ]
    if compatibility.get("softmax_underflow_gradient_bridge") is not True:
        raise RuntimeError("unknown compatibility variant")
    return [
        str(environment / "bin" / "python"),
        "-m",
        "act.pipeline.moe.advmoe_compatibility_train",
        "--workspace",
        config["workspace_boundary"],
        "--official-source",
        str(source),
        "--summary",
        str(Path(run["root"]) / "compatibility_bridge_summary.json"),
        "--",
        *official_arguments,
    ]


def _find_live_checkpoint(run_root: Path) -> Path | None:
    matches = list(run_root.glob("results/training/train_moe/**/checkpoint/checkpoint.pth.tar"))
    if len(matches) > 1:
        raise RuntimeError(f"multiple live checkpoints found: {matches}")
    return matches[0] if matches else None


def _all_floating_tensors_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value).all().item())
        return True
    if isinstance(value, dict):
        return all(_all_floating_tensors_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_floating_tensors_finite(child) for child in value)
    return True


def _checkpoint_epoch(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    epoch = int(payload["epoch"])
    if epoch <= 0:
        raise RuntimeError(f"invalid checkpoint epoch: {epoch}")
    for key in ("state_dict", "router", "optimizer", "router_optimizer"):
        if key not in payload:
            raise RuntimeError(f"checkpoint missing {key}")
        if not _all_floating_tensors_finite(payload[key]):
            raise NonfiniteCheckpointError(
                f"checkpoint epoch {epoch} contains non-finite {key} state"
            )
    return epoch


def snapshot_checkpoint(live: Path, snapshots: Path) -> dict[str, Any] | None:
    snapshots.mkdir(parents=True, exist_ok=True)
    try:
        before = live.stat()
        with tempfile.NamedTemporaryFile(
            dir=snapshots,
            prefix=".live-checkpoint-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copy2(live, temporary)
        after = live.stat()
        signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if signature_before != signature_after:
            temporary.unlink(missing_ok=True)
            return None
        epoch = _checkpoint_epoch(temporary)
        copied_hash = _sha256(temporary)
    except NonfiniteCheckpointError:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise
    except (EOFError, OSError, RuntimeError, KeyError, ValueError, pickle.UnpicklingError):
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        return None
    destination = snapshots / f"epoch_{epoch:03d}.pth.tar"
    if destination.exists():
        existing_hash = _sha256(destination)
        temporary.unlink(missing_ok=True)
        if copied_hash != existing_hash:
            raise RuntimeError(
                f"epoch {epoch} was rewritten with different checkpoint content"
            )
        return {
            "epoch": epoch,
            "path": str(destination),
            "sha256": existing_hash,
            "size_bytes": destination.stat().st_size,
            "existing": True,
            "floating_state_finite": True,
        }
    os.replace(temporary, destination)
    return {
        "epoch": epoch,
        "path": str(destination),
        "sha256": copied_hash,
        "size_bytes": destination.stat().st_size,
        "existing": False,
        "floating_state_finite": True,
    }


def _preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = Path(config["workspace_boundary"]).resolve()
    act_repo = Path(__file__).resolve().parents[3]
    source = Path(config["official_source"]["repository"])
    run_root = Path(config["run"]["root"])
    for path in (config_path, act_repo, source, run_root):
        path.resolve().relative_to(workspace)
    if _git(act_repo, "branch", "--show-current") != "feat/moe-route-verification":
        raise RuntimeError("ACT is not on the feature branch")
    if _git(act_repo, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree is dirty")
    head = _git(act_repo, "rev-parse", "HEAD")
    origin = _git(act_repo, "rev-parse", "origin/feat/moe-route-verification")
    if head != origin:
        raise RuntimeError("ACT feature branch is not synchronized with origin")
    if _git(source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")
    if _git(source, "rev-parse", "HEAD") != config["official_source"]["commit"]:
        raise RuntimeError("official source commit mismatch")
    environment_python = Path(config["environment"]["path"]) / "bin" / "python"
    if not environment_python.is_file():
        raise RuntimeError("isolated environment is missing")
    compatibility = config.get("compatibility_variant")
    if compatibility is None:
        smoke = act_repo / "act/pipeline/moe/results/baseline/advmoe_training_smoke_seed0_r1.json"
    else:
        smoke = Path(compatibility["finite_smoke_audit"])
        smoke.resolve().relative_to(workspace)
        if _sha256(smoke) != compatibility["finite_smoke_audit_sha256"]:
            raise RuntimeError("compatibility smoke audit hash mismatch")
    smoke_payload = _load(smoke)
    if smoke_payload.get("status") != "PASS":
        raise RuntimeError("training smoke gate is not PASS")
    if compatibility is None and smoke_payload.get("training_unlocked") is not True:
        raise RuntimeError("official training smoke does not unlock training")
    data_link = Path(config["dataset"]["run_data_root"]) / "CIFAR10/cifar-10-batches-py"
    if data_link.resolve() != Path(config["dataset"]["source_root"]).resolve():
        raise RuntimeError("CIFAR10 data identity mismatch")
    disk_free = shutil.disk_usage(workspace).free
    disk_required = int(config["preflight_gates"]["minimum_free_disk_gib"] * GIB)
    if disk_free < disk_required:
        raise RuntimeError("disk resource gate failed")
    free_gpu, total_gpu = torch.cuda.mem_get_info(0)
    gpu_required = int(config["preflight_gates"]["minimum_free_gpu_memory_gib"] * GIB)
    if free_gpu < gpu_required:
        raise RuntimeError("GPU resource gate failed")
    if (run_root / "results").exists():
        raise RuntimeError("run output already exists")
    return {
        "act_head": head,
        "official_commit": _git(source, "rev-parse", "HEAD"),
        "official_tree": _git(source, "rev-parse", "HEAD^{tree}"),
        "smoke_manifest_sha256": _sha256(smoke),
        "free_disk_gib": disk_free / GIB,
        "free_gpu_gib": free_gpu / GIB,
        "total_gpu_gib": total_gpu / GIB,
    }


def run_supervisor(config_path: Path, progress_path: Path) -> dict[str, Any]:
    config = _load(config_path)
    run_root = Path(config["run"]["root"])
    snapshots = run_root / "checkpoint_snapshots"
    preflight = _preflight(config_path, config)
    command = build_training_command(config)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_CACHE_DIR": "/data1/Kane/MOE/cache/pip",
            "HF_HOME": "/data1/Kane/MOE/cache/huggingface",
            "TORCH_HOME": "/data1/Kane/MOE/cache/torch",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        }
    )
    started = time.time()
    progress: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "command": command,
        "preflight": preflight,
        "started_unix_seconds": started,
        "checkpoints": [],
    }
    _atomic_json(progress_path, progress)
    process = subprocess.Popen(command, cwd=run_root, env=environment)
    recorded: dict[int, dict[str, Any]] = {}
    try:
        while process.poll() is None:
            live = _find_live_checkpoint(run_root)
            if live is not None:
                snapshot = snapshot_checkpoint(live, snapshots)
                if snapshot is not None and snapshot["epoch"] not in recorded:
                    recorded[snapshot["epoch"]] = snapshot
                    progress["checkpoints"] = [recorded[key] for key in sorted(recorded)]
                    progress["latest_epoch"] = max(recorded)
                    progress["last_update_unix_seconds"] = time.time()
                    _atomic_json(progress_path, progress)
            time.sleep(2.0)
        return_code = int(process.wait())
        live = _find_live_checkpoint(run_root)
        if live is not None:
            snapshot = snapshot_checkpoint(live, snapshots)
            if snapshot is not None:
                recorded[snapshot["epoch"]] = snapshot
        expected_epochs = set(range(1, int(config["run"]["epochs"]) + 1))
        observed_epochs = set(recorded)
        compatibility_summary = None
        compatibility_ok = True
        if config.get("compatibility_variant") is not None:
            compatibility_path = run_root / "compatibility_bridge_summary.json"
            if compatibility_path.is_file():
                compatibility_summary = _load(compatibility_path)
                compatibility_ok = (
                    compatibility_summary.get("status") == "PASSED"
                    and compatibility_summary.get("official_source", {}).get("clean_after")
                    is True
                    and int(
                        compatibility_summary.get("bridge", {}).get(
                            "replaced_elements", 0
                        )
                    )
                    > 0
                )
            else:
                compatibility_ok = False
        status = (
            "PASSED"
            if return_code == 0
            and observed_epochs == expected_epochs
            and compatibility_ok
            else "FAILED"
        )
        summary = {
            **progress,
            "status": status,
            "return_code": return_code,
            "ended_unix_seconds": time.time(),
            "runtime_seconds": time.time() - started,
            "checkpoints": [recorded[key] for key in sorted(recorded)],
            "missing_checkpoint_epochs": sorted(expected_epochs - observed_epochs),
            "compatibility_bridge": compatibility_summary,
            "official_clone_clean_after": not bool(
                _git(Path(config["official_source"]["repository"]), "status", "--porcelain=v1")
            ),
        }
        _atomic_json(progress_path, summary)
        if status != "PASSED":
            raise RuntimeError(
                f"training failed: return_code={return_code}, "
                f"missing_epochs={summary['missing_checkpoint_epochs']}, "
                f"compatibility_ok={compatibility_ok}"
            )
        return summary
    except BaseException as error:
        if process.poll() is None:
            process.terminate()
        progress.update(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "last_update_unix_seconds": time.time(),
                "checkpoints": [recorded[key] for key in sorted(recorded)],
            }
        )
        _atomic_json(progress_path, progress)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.progress.exists():
        raise FileExistsError(arguments.progress)
    print(json.dumps(run_supervisor(arguments.config, arguments.progress), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
