from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import torch


REQUIRED_CHECKPOINT_KEYS = {
    "epoch",
    "state_dict",
    "router",
    "best_acc",
    "sa_record",
    "optimizer",
    "router_optimizer",
}
METRIC_PATTERN = re.compile(
    r"Epoch (\d+), SA:\s+([0-9.]+)%, RA:\s+([0-9.]+)%\. "
    r"\[best performance \(RA\):\s+([0-9.]+), \(SA\):\s+([0-9.]+)\]"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def floating_tensor_summary(value: Any) -> dict[str, int | bool]:
    tensors: list[torch.Tensor] = []

    def visit(current: Any) -> None:
        if torch.is_tensor(current):
            if current.is_floating_point() or current.is_complex():
                tensors.append(current)
        elif isinstance(current, dict):
            for child in current.values():
                visit(child)
        elif isinstance(current, (list, tuple)):
            for child in current:
                visit(child)

    visit(value)
    elements = sum(tensor.numel() for tensor in tensors)
    finite_elements = sum(
        int(torch.isfinite(tensor).sum().item()) for tensor in tensors
    )
    nan_elements = sum(int(torch.isnan(tensor).sum().item()) for tensor in tensors)
    inf_elements = sum(int(torch.isinf(tensor).sum().item()) for tensor in tensors)
    nonfinite_tensors = sum(
        not bool(torch.isfinite(tensor).all().item()) for tensor in tensors
    )
    return {
        "tensors": len(tensors),
        "elements": elements,
        "finite_elements": finite_elements,
        "nan_elements": nan_elements,
        "inf_elements": inf_elements,
        "nonfinite_tensors": nonfinite_tensors,
        "all_finite": finite_elements == elements,
    }


def audit(config_path: Path, progress_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    progress = _load_json(progress_path)
    workspace = Path(config["workspace_boundary"]).resolve()
    act_repo = Path(__file__).resolve().parents[3]
    run_root = Path(config["run"]["root"])
    source = Path(config["official_source"]["repository"])
    for path in (config_path, progress_path, act_repo, run_root, source):
        path.resolve().relative_to(workspace)

    issues: list[str] = []
    if progress.get("status") != "PASSED":
        issues.append("progress status is not PASSED")
    if progress.get("return_code") != 0:
        issues.append("trainer return code is not zero")
    if progress.get("missing_checkpoint_epochs") != []:
        issues.append("progress reports missing checkpoint epochs")
    if progress.get("official_clone_clean_after") is not True:
        issues.append("progress does not attest a clean official clone after training")
    if progress.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash differs from the preflight identity")
    if _git(source, "status", "--porcelain=v1"):
        issues.append("official source clone is dirty")
    if _git(source, "rev-parse", "HEAD") != config["official_source"]["commit"]:
        issues.append("official source commit changed")

    records = progress.get("checkpoints", [])
    expected_epochs = list(range(1, int(config["run"]["epochs"]) + 1))
    observed_epochs = [int(record["epoch"]) for record in records]
    if observed_epochs != expected_epochs:
        issues.append("checkpoint record epochs are not exactly consecutive")

    verified_records: list[dict[str, Any]] = []
    recovered_metadata: list[dict[str, Any]] = []
    nonfinite_epochs: dict[str, list[int]] = {
        "main_without_router": [],
        "embedded_router": [],
        "router": [],
        "optimizer": [],
        "router_optimizer": [],
    }
    snapshot_root = run_root / "checkpoint_snapshots"
    for record in records:
        epoch = int(record["epoch"])
        path = Path(record["path"])
        expected_path = snapshot_root / f"epoch_{epoch:03d}.pth.tar"
        if path.resolve() != expected_path.resolve():
            issues.append(f"epoch {epoch}: unexpected snapshot path")
            continue
        if not path.is_file():
            issues.append(f"epoch {epoch}: snapshot is missing")
            continue
        actual_hash = _sha256(path)
        actual_size = path.stat().st_size
        if actual_hash != record.get("sha256"):
            issues.append(f"epoch {epoch}: snapshot hash mismatch")
        if record.get("size_bytes") is None and record.get("existing") is True:
            recovered_metadata.append(
                {
                    "epoch": epoch,
                    "field": "size_bytes",
                    "value": actual_size,
                    "reason": "legacy final existing-snapshot record omitted size; hash and file were independently verified",
                }
            )
        elif actual_size != record.get("size_bytes"):
            issues.append(f"epoch {epoch}: snapshot size mismatch")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # audit records backend decoding failures verbatim
            issues.append(f"epoch {epoch}: checkpoint load failed: {type(error).__name__}")
            continue
        missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(payload))
        if missing:
            issues.append(f"epoch {epoch}: missing checkpoint keys {missing}")
        if int(payload.get("epoch", -1)) != epoch:
            issues.append(f"epoch {epoch}: embedded epoch mismatch")
        state_dict = payload.get("state_dict", {})
        main_without_router = {
            name: tensor
            for name, tensor in state_dict.items()
            if not name.startswith("router.") and ".router." not in name
        }
        embedded_router = {
            name: tensor
            for name, tensor in state_dict.items()
            if name.startswith("router.") or ".router." in name
        }
        finiteness = {
            "main_without_router": floating_tensor_summary(main_without_router),
            "embedded_router": floating_tensor_summary(embedded_router),
            "router": floating_tensor_summary(payload.get("router", {})),
            "optimizer": floating_tensor_summary(payload.get("optimizer", {})),
            "router_optimizer": floating_tensor_summary(
                payload.get("router_optimizer", {})
            ),
        }
        for group, summary in finiteness.items():
            if summary["all_finite"] is not True:
                nonfinite_epochs[group].append(epoch)
        verified_records.append(
            {
                "epoch": epoch,
                "sha256": actual_hash,
                "size_bytes": actual_size,
                "finiteness": finiteness,
            }
        )

    for group, epochs in nonfinite_epochs.items():
        if epochs:
            issues.append(
                f"{group} contains non-finite tensors in {len(epochs)} "
                f"checkpoints (first epoch {epochs[0]}, last epoch {epochs[-1]})"
            )

    result_roots = list(run_root.glob("results/training/train_moe/*"))
    if len(result_roots) != 1:
        issues.append(f"expected one released result root, found {len(result_roots)}")
        result_root = None
    else:
        result_root = result_roots[0]

    metric_rows: list[dict[str, float | int]] = []
    best_snapshot_epochs: list[int] = []
    final_hash = None
    best_hash = None
    if result_root is not None:
        setup_log = result_root / "setup.log"
        for match in METRIC_PATTERN.finditer(setup_log.read_text(encoding="utf-8")):
            epoch, sa, ra, best_ra, best_sa = match.groups()
            metric_rows.append(
                {
                    "epoch": int(epoch),
                    "sa_percent": float(sa),
                    "ra_percent": float(ra),
                    "best_ra_percent": float(best_ra),
                    "best_sa_percent": float(best_sa),
                }
            )
        if [row["epoch"] for row in metric_rows] != list(range(len(expected_epochs))):
            issues.append("setup log does not contain exactly one ordered row per training epoch")
        live = result_root / "checkpoint/checkpoint.pth.tar"
        best = result_root / "checkpoint/model_best.pth.tar"
        if not live.is_file() or not best.is_file():
            issues.append("released final or best checkpoint is missing")
        else:
            final_hash = _sha256(live)
            best_hash = _sha256(best)
            if records and final_hash != records[-1].get("sha256"):
                issues.append("released final checkpoint does not match epoch-100 snapshot")
            best_snapshot_epochs = [
                int(record["epoch"])
                for record in records
                if record.get("sha256") == best_hash
            ]
            if len(best_snapshot_epochs) != 1:
                issues.append("released best checkpoint does not match exactly one snapshot")

    best_row = max(metric_rows, key=lambda row: float(row["ra_percent"])) if metric_rows else None
    if best_row is not None and best_snapshot_epochs:
        if best_snapshot_epochs[0] != int(best_row["epoch"]) + 1:
            issues.append("best checkpoint epoch does not match maximum logged RA")
        if not math.isclose(
            float(best_row["ra_percent"]),
            float(best_row["best_ra_percent"]),
            abs_tol=0.005,
        ):
            issues.append("best metric row is internally inconsistent")

    return {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "progress": {"path": str(progress_path), "sha256": _sha256(progress_path)},
        "act": {
            "branch": _git(act_repo, "branch", "--show-current"),
            "head": _git(act_repo, "rev-parse", "HEAD"),
            "worktree_clean": not bool(_git(act_repo, "status", "--porcelain=v1")),
        },
        "official_source": {
            "commit": _git(source, "rev-parse", "HEAD"),
            "tree": _git(source, "rev-parse", "HEAD^{tree}"),
            "clean": not bool(_git(source, "status", "--porcelain=v1")),
        },
        "checkpoint_count": len(records),
        "verified_checkpoint_count": len(verified_records),
        "checkpoint_finiteness": {
            "all_finite": not any(nonfinite_epochs.values()),
            "nonfinite_epochs_by_group": nonfinite_epochs,
            "records": verified_records,
        },
        "recovered_metadata": recovered_metadata,
        "final_checkpoint_sha256": final_hash,
        "best_checkpoint_sha256": best_hash,
        "best_checkpoint_epoch": best_snapshot_epochs[0] if len(best_snapshot_epochs) == 1 else None,
        "metric_row_count": len(metric_rows),
        "last_metric": metric_rows[-1] if metric_rows else None,
        "best_metric": best_row,
        "runtime_seconds": progress.get("runtime_seconds"),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.progress)
    _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
