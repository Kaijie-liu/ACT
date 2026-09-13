"""Frozen two-arm observed-cohort relationship experiment, separate from SCH100."""
import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _git_value
from act.pipeline.moe.paired_followup import save, source_identity
from act.pipeline.moe.schedule_confirmation import inspect_row

DEFAULT = PROJECT_ROOT / "act/pipeline/moe/configs/relation_ablation_r1.json"
ARMS = ("shared", "independent")
HASHES = {"shared": "26e9ffef88ad257fbc079bdd50df75b78c9625e379a3adcd20b396b495acb4c7",
          "independent": "67f7e5814ecb57a86c36c21f869998945604c61f68e3855c3a36dea5e15f063c"}
SELECTION = "e001940bf28f64c997dc1f947dd2fcda515f4051985836756c25843b89dcd987"


def artifacts(config, deep=False):
    if (config["classification"] != "POST_CONFIRMATION_OBSERVED_RELATION_ABLATION_R1"
            or config["sample_count"] != 10 or config["budget_seconds"] != 300
            or set(config["methods"]) != set(ARMS)):
        raise ValueError("frozen design changed")
    path = _inside(Path(config["selection"]), PROJECT_ROOT)
    if config["selection_sha256"] != SELECTION or _sha256(path) != SELECTION:
        raise ValueError("observed selection changed")
    selection = json.loads(path.read_text())
    selection["samples"] = selection["samples"][:10]
    configs = {}
    for arm in ARMS:
        row = config["methods"][arm]; path = _inside(Path(row["path"]), PROJECT_ROOT)
        if row["sha256"] != HASHES[arm] or _sha256(path) != HASHES[arm]:
            raise ValueError("method changed")
        configs[arm] = json.loads(path.read_text())
    left, right = [copy.deepcopy(configs[a]) for a in ARMS]
    if left["f0"].pop("expert_relation") != "shared_input" or right["f0"].pop("expert_relation") != "independent_inputs" or left != right:
        raise ValueError("not a single-factor relation comparison")
    if deep:
        data = selection["dataset"]
        if _sha256(_inside(Path(data["raw_test_batch"]), WRITE_ROOT)) != data["raw_test_batch_sha256"]:
            raise ValueError("dataset drift")
        for subject in selection["models"].values():
            if _sha256(_inside(Path(subject["checkpoint"]), WRITE_ROOT)) != subject["checkpoint_sha256"]:
                raise ValueError("checkpoint drift")
    return selection, configs


def jobs(selection, smoke):
    result = []; models = sorted(selection["models"])
    for rank, sample in enumerate(selection["smoke_samples" if smoke else "samples"]):
        for offset in range(len(models)):
            model = models[(rank+offset) % len(models)]
            for position in range(2):
                arm = ARMS[(rank+models.index(model)+position) % 2]
                result.append({"rank": rank, "model": model, "method": arm, "position": position,
                               "dataset_index": sample["dataset_index"], "job_id": f"rank{rank}_{model}_{arm}"})
    return result


def audit(root):
    root = _inside(Path(root), WRITE_ROOT)
    runtime = json.loads((root/"runtime.json").read_text()); config = runtime["config"]
    path = _inside(Path(runtime["config_path"]), PROJECT_ROOT)
    if _sha256(path) != runtime["config_sha256"] or json.loads(path.read_text()) != config:
        raise ValueError("runtime/config drift")
    selection, configs = artifacts(config)
    rows = [json.loads(v) for v in (root/"rows.jsonl").read_text().splitlines()]
    expected = jobs(selection, runtime["smoke"])
    if len(rows) != len(expected) or any(any(r[k] != j[k] for k in j) for r, j in zip(rows, expected)):
        raise ValueError("incomplete/duplicate/reordered schedule")
    details = {}; provenance = []
    for row in rows:
        if json.loads((root/row["job_id"]/"terminal.json").read_text()) != row:
            raise ValueError("terminal differs from ledger")
        details[row["model"], row["rank"], row["method"]] = inspect_row(root, row, runtime, selection, configs)
        if row["package"]:
            e = json.loads((Path(row["package"])/"evidence.json").read_text())
            provenance.append({"job_id": row["job_id"], "decision_tier": e["verdict"]["decision_tier"],
                               "reason": e["verdict"]["reason"], "tier2": e["tier2"],
                               "manifest_sha256": row["manifest_sha256"]})
    by = {(r["model"], r["rank"], r["method"]): r for r in rows}
    n = 1 if runtime["smoke"] else 10
    models = {}; equal = unavailable = 0
    for model in sorted(selection["models"]):
        for i in range(n):
            if {by[model,i,a]["status"] for a in ARMS} == {"SAFE", "UNSAFE"}:
                raise ValueError("full-model SAFE/UNSAFE contradiction")
            a, b = [details[model,i,arm]["facts"] for arm in ARMS]
            if a is not None and b is not None:
                if a != b: raise ValueError("common facts differ")
                equal += 1
            else: unavailable += 1
        methods = {}
        for arm in ARMS:
            values = [by[model,i,arm] for i in range(n)]
            methods[arm] = {"states": dict(Counter(r["status"] for r in values)),
                           "mean_observed_seconds": mean(r["wall_seconds"] for r in values),
                           "route_changing_safe_ranks": [i for i in range(n) if by[model,i,arm]["status"] == "SAFE"
                               and (details[model,i,arm]["pair_count"] or 0) > 1]}
        contrasts = {}
        for label, accepted in (("SAFE", {"SAFE"}), ("solved", {"SAFE", "UNSAFE"})):
            gain = [i for i in range(n) if by[model,i,"shared"]["status"] in accepted and by[model,i,"independent"]["status"] not in accepted]
            loss = [i for i in range(n) if by[model,i,"independent"]["status"] in accepted and by[model,i,"shared"]["status"] not in accepted]
            contrasts[label] = {"gained": gain, "lost": loss, "net": len(gain)-len(loss)}
        delta = [by[model,i,"shared"]["wall_seconds"]-by[model,i,"independent"]["wall_seconds"] for i in range(n)]
        models[model] = {"methods": methods, "contrasts": contrasts,
                         "mean_paired_seconds": mean(delta), "median_paired_seconds": median(delta)}
    return {"status": "PASS", "issues": [], "rows": len(rows),
            "packages": sum(d["package"] for d in details.values()),
            "unsafe_replayed": sum(d["replayed"] for d in details.values()),
            "common_fact_pairs_equal": equal, "common_fact_pairs_unavailable": unavailable,
            "models": models, "provenance": provenance,
            "runtime_sha256": _sha256(root/"runtime.json"), "rows_sha256": _sha256(root/"rows.jsonl"),
            "scope": "Observed-input mechanism follow-up; structural audit and witness replay, not independent SAFE reproof."}


def smoke_gate(path, config, source):
    root = _inside(Path(config["smoke_output"]), WRITE_ROOT)
    rt = json.loads((root/"runtime.json").read_text())
    if not rt["smoke"] or rt["config"] != config or rt["config_sha256"] != _sha256(path) or rt["source_sha256"] != source:
        raise ValueError("smoke identity mismatch")
    checked = audit(root)
    if checked != json.loads((root/"audit.final.json").read_text()) or checked["common_fact_pairs_equal"] == 0:
        raise ValueError("smoke audit/conformance failed")
    for arm in ARMS:
        if not any(p["job_id"].endswith("_"+arm) for p in checked["provenance"]):
            raise ValueError("smoke lacks an arm's evidence")
    return {"root": str(root), "audit_sha256": _sha256(root/"audit.final.json")}


def run(path, smoke=False):
    if _git_value("branch", "--show-current") != "feat/moe-route-verification" or _git_value("status", "--porcelain"):
        raise ValueError("clean feature checkout required")
    path = _inside(Path(path), PROJECT_ROOT); config = json.loads(path.read_text())
    selection, configs = artifacts(config, deep=True)
    if Path(sys.executable).resolve() != Path(config["python"]).resolve(): raise ValueError("act-py312 required")
    source = source_identity(); gate = None if smoke else smoke_gate(path, config, source)
    root = _inside(Path(config["smoke_output" if smoke else "output"]), WRITE_ROOT)
    root.mkdir(exist_ok=False)
    runtime = {"config": config, "config_path": str(path), "config_sha256": _sha256(path),
               "git_head": _git_value("rev-parse", "HEAD"), "source_sha256": source,
               "smoke": smoke, "smoke_gate": gate, "started_unix": time.time()}
    save(root/"runtime.json", runtime)
    env = {**os.environ, "ACT_TORCHVISION_DATA_ROOT": str(PROJECT_ROOT/"data/torchvision"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    schedule = jobs(selection, smoke)
    for number, job in enumerate(schedule, 1):
        if source_identity() != source or _git_value("status", "--porcelain"):
            raise ValueError("source/worktree drift")
        artifacts(config)
        directory = root/job["job_id"]; directory.mkdir()
        command = [config["python"], "-m", "act.pipeline.moe.staged_verifier",
                   "--checkpoint", selection["models"][job["model"]]["checkpoint"],
                   "--dataset-index", str(job["dataset_index"]), "--epsilon", repr(selection["request"]["epsilon"]),
                   "--config", config["methods"][job["method"]]["path"], "--output-dir", str(directory/"package"),
                   "--progress-path", str(directory/"progress.json"), "--common-fact-snapshot", str(directory/"common_facts.json")]
        started = time.monotonic(); expired = False; code = None
        with (directory/"worker.log").open("x") as log:
            try:
                code = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
            except subprocess.TimeoutExpired: expired = True
        elapsed = time.monotonic()-started; expired = expired or elapsed > 300
        row = {**job, "budget_seconds": 300, "wall_seconds": elapsed, "outer_timeout": expired, "return_code": code,
               "status": "TIMEOUT" if expired else "ERROR", "package": None, "snapshot_sha256": None,
               "load_average_after": list(os.getloadavg())}
        if (directory/"common_facts.json").exists(): row["snapshot_sha256"] = _sha256(directory/"common_facts.json")
        try:
            if not expired and code == 0:
                p = directory/"package"; e = json.loads((p/"evidence.json").read_text())
                row.update(status=e["verdict"]["status"], package=str(p), manifest_sha256=_sha256(p/"manifest.json"))
            if row["status"] != "ERROR": inspect_row(root, row, runtime, selection, configs)
        except Exception as exc: row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        save(directory/"terminal.json", row)
        with (root/"rows.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())
        print(f"{'smoke' if smoke else 'full'} {number}/{len(schedule)} {job['job_id']} {row['status']}", flush=True)
        if row["status"] == "ERROR": raise RuntimeError("retained worker/audit failure; no replacement")
    save(root/"audit.final.json", audit(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pipeline", action="store_true"); mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(args.audit), indent=2)); return
    with (PROJECT_ROOT/"data/moe/results/route_complexity_pairing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.pipeline: run(args.config, True); run(args.config, False)
        else: run(args.config, args.smoke)


if __name__ == "__main__": main()
