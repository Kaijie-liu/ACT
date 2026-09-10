"""Interleaved four-method follow-up; no historical timing is reused."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _git_value
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.run_staged_verifier_confirmatory import _load_jsonl

METHODS = ("staged", "route_invariance", "monolithic_f0", "tier1_only")
DEFAULT = PROJECT_ROOT / "act/pipeline/moe/configs/paired_followup_r1.json"


def source_identity():
    """Allow documentation commits between smoke/full, not executable changes."""
    names = _git_value("ls-files", "act").splitlines()
    digest = hashlib.sha256()
    for name in sorted(n for n in names if n.endswith(".py")):
        digest.update(f"{name}:{_sha256(PROJECT_ROOT / name)}\n".encode())
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def schedule(selection, ranks):
    """Rotate method position by sample, and model position by sample."""
    jobs = []
    models = sorted(selection["models"])
    for rank in ranks:
        for offset in range(len(models)):
            model = models[(rank + offset) % len(models)]
            shift = (rank + models.index(model)) % len(METHODS)
            for position in range(len(METHODS)):
                method = METHODS[(shift + position) % len(METHODS)]
                jobs.append({"job_id": f"rank{rank}_{model}_{method}",
                             "rank": rank, "model": model, "method": method,
                             "position": position,
                             "dataset_index": selection["samples"][rank]["dataset_index"]})
    return jobs


def method_config(base, method, budget):
    cfg = copy.deepcopy(base)
    cfg["comparison_method"] = method
    if method == "tier1_only":
        # Same low-budget pass; escalation can spend the remaining outer budget.
        cfg["tier1"]["solver"]["escalation_budget_per_branch"] = budget
    if method == "monolithic_f0":
        # A joint property can consume the remaining outer budget.
        cfg["f0"]["solver"]["property_seconds"] = budget
    return cfg


def summarize(rows, expected):
    models = {}
    for model in sorted({row["model"] for row in rows}):
        part = [row for row in rows if row["model"] == model]
        by_method = {method: {r["rank"]: r for r in part if r["method"] == method}
                     for method in METHODS}
        details = {}
        for method, values in by_method.items():
            details[method] = {
                "rows": len(values), "statuses": dict(Counter(r["status"] for r in values.values())),
                "total_wall_seconds": sum(r["wall_seconds"] for r in values.values()),
            }
        pairs = {}
        for baseline in METHODS[1:]:
            common = sorted(set(by_method["staged"]) & set(by_method[baseline]))
            result = {"paired_rows": len(common)}
            for outcome, statuses in (("safe", {"SAFE"}), ("solved", {"SAFE", "UNSAFE"})):
                gained, lost = [], []
                for rank in common:
                    a = by_method["staged"][rank]["status"] in statuses
                    b = by_method[baseline][rank]["status"] in statuses
                    if a and not b:
                        gained.append(rank)
                    if b and not a:
                        lost.append(rank)
                result[outcome] = {"staged_only_ranks": gained, "baseline_only_ranks": lost,
                                   "net_gain": len(gained) - len(lost)}
            pairs[baseline] = result
        models[model] = {"methods": details, "paired": pairs}
    return {"classification": "OBSERVED_COHORT_PAIRED_FOLLOWUP",
            "expected_jobs": expected, "observed_jobs": len(rows),
            "complete": len(rows) == expected, "models": models,
            "claim_boundary": "Same-family observed cohort; not certified accuracy or holdout. Audit checks structure, not independent SAFE re-proving."}


def run(path, *, smoke=False, resume=False):
    config = json.loads(path.read_text())
    if _git_value("branch", "--show-current") != "feat/moe-route-verification" or _git_value("status", "--porcelain"):
        raise RuntimeError("clean feature branch required")
    if Path(sys.executable).resolve() != Path(config["python"]).resolve():
        raise RuntimeError("act-py312 required")
    output = _inside(Path(config["smoke_output"] if smoke else config["output"]), WRITE_ROOT)
    if resume:
        if not output.is_dir() or not (output / "runtime.json").is_file():
            raise RuntimeError("resume requires an existing run and runtime identity")
    elif output.exists():
        raise RuntimeError("refusing to overwrite run")
    # Resumption is not authority to bypass the independent smoke gate.
    if not smoke:
        from act.pipeline.moe.audit_paired_followup import audit
        smoke_root = _inside(Path(config["smoke_output"]), WRITE_ROOT)
        smoke_audit = audit(smoke_root)
        if smoke_audit["status"] != "PASS":
            raise RuntimeError("complete, independently audited smoke required")
        smoke_runtime = json.loads((smoke_root / "runtime.json").read_text())
        if (smoke_runtime["config_sha256"] != _sha256(path) or
                smoke_runtime["source_sha256"] != source_identity()):
            raise RuntimeError("smoke used different executable sources/config")
    selection_path = _inside(Path(config["selection"]), PROJECT_ROOT)
    base_path = _inside(Path(config["base_config"]), PROJECT_ROOT)
    for artifact, expected in ((selection_path, config["selection_sha256"]),
                               (base_path, config["base_config_sha256"])):
        if _sha256(artifact) != expected:
            raise RuntimeError(f"identity changed: {artifact}")
    selection = json.loads(selection_path.read_text())
    for model in selection["models"].values():
        checkpoint = _inside(Path(model["checkpoint"]), WRITE_ROOT)
        if _sha256(checkpoint) != model["checkpoint_sha256"]:
            raise RuntimeError("checkpoint hash changed")
    base = json.loads(base_path.read_text())
    ranks = config["smoke_ranks"] if smoke else list(range(100))
    jobs = schedule(selection, ranks)
    budget = float(config["budget_seconds"])
    identity = {"config_sha256": _sha256(path), "git_head": _git_value("rev-parse", "HEAD"),
                "source_sha256": source_identity(),
                "smoke": smoke, "jobs": jobs, "config": config}
    if output.exists():
        if not resume:
            raise RuntimeError("refusing to overwrite run")
        previous = json.loads((output / "runtime.json").read_text())
        if {k: v for k, v in previous.items() if k != "git_head"} != {
                k: v for k, v in identity.items() if k != "git_head"}:
            raise RuntimeError("resume identity/code changed")
    else:
        output.mkdir(parents=True)
        save(output / "runtime.json", identity)
        for method in METHODS:
            save(output / f"{method}.json", method_config(base, method, budget))
    for method in METHODS:
        if json.loads((output / f"{method}.json").read_text()) != method_config(base, method, budget):
            raise RuntimeError("method config drift")
    rows = _load_jsonl(output / "rows.jsonl")
    if [r["job_id"] for r in rows] != [j["job_id"] for j in jobs[:len(rows)]]:
        raise RuntimeError("results are not a scheduled prefix")
    environment = os.environ.copy()
    environment.update(ACT_TORCHVISION_DATA_ROOT=str(PROJECT_ROOT / "data/torchvision"),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for job in jobs[len(rows):]:
        root = output / job["job_id"]
        root.mkdir(exist_ok=True)
        attempt = len(list(root.glob("attempt*"))) + 1
        directory = root / f"attempt{attempt}"
        directory.mkdir()
        model = selection["models"][job["model"]]
        command = [config["python"], "-m", "act.pipeline.moe.staged_verifier",
                   "--checkpoint", model["checkpoint"], "--dataset-index", str(job["dataset_index"]),
                   "--epsilon", repr(selection["request"]["epsilon"]),
                   "--config", str(output / f"{job['method']}.json"),
                   "--output-dir", str(directory / "package"),
                   "--progress-path", str(directory / "progress.json")]
        started = time.monotonic()
        expired, code = False, None
        with (directory / "worker.log").open("x") as log:
            try:
                result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=budget)
                code = result.returncode
            except subprocess.TimeoutExpired:
                expired = True
        wall = time.monotonic() - started
        row = {**job, "attempt": attempt, "wall_seconds": wall, "budget_seconds": budget,
               "load_average_after": list(os.getloadavg()),
               "return_code": code, "outer_timeout": expired, "package": None,
               "status": "TIMEOUT" if expired else "ERROR", "reason": "HARD_DEADLINE" if expired else "WORKER_ERROR"}
        package = directory / "package"
        if not expired and code == 0:
            audit = audit_evidence_package(package, replay_unsafe=True)
            save(directory / "audit.json", audit)
            if audit["status"] != "PASS":
                raise RuntimeError(f"audit failed: {directory}")
            evidence = json.loads((package / "evidence.json").read_text())
            row.update(status=evidence["verdict"]["status"], reason=evidence["verdict"]["reason"],
                       package=str(package), manifest_sha256=_sha256(package / "manifest.json"))
        progress = directory / "progress.json"
        try:
            row["last_progress"] = json.loads(progress.read_text()) if progress.exists() else None
        except json.JSONDecodeError:
            row["last_progress"] = None
            row["progress_parse_failed"] = True
        save(directory / "terminal.json", row)
        with (output / "rows.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        rows.append(row)
        save(output / "summary.partial.json", summarize(rows, len(jobs)))
        print(f"{len(rows)}/{len(jobs)} {job['job_id']} {row['status']}", flush=True)
        if row["status"] == "ERROR":
            raise RuntimeError(f"worker failed; retained {directory}")
    save(output / "summary.json", summarize(rows, len(jobs)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(args.config, smoke=args.smoke, resume=args.resume))


if __name__ == "__main__":
    main()
