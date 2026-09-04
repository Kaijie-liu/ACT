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
        if actual_size != record.get("size_bytes"):
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
        verified_records.append(
            {
                "epoch": epoch,
                "sha256": actual_hash,
                "size_bytes": actual_size,
            }
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
