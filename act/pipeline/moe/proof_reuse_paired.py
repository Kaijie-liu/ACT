"""Frozen real-checkpoint two-arm engineering comparison; no historical times."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _git_value, _sha256
from act.pipeline.moe.paired_followup import save, source_identity
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.scoped_f0_proofs import audit_reused_property
from act.pipeline.moe.staged_verifier import _canonical_sha256

DEFAULT = PROJECT_ROOT / "act/pipeline/moe/configs/proof_reuse_paired_r1.json"


def jobs(selection, ranks):
    models = sorted(selection["models"])
    result = []
    for rank in ranks:
        for offset in range(len(models)):
            name = models[(rank + offset) % len(models)]
            for position in range(2):
                enabled = bool((rank + models.index(name) + position) % 2)
                result.append({"model": name, "rank": rank, "reuse": enabled,
                               "position": position, "dataset_index": selection["samples"][rank]["dataset_index"],
                               "job_id": f"rank{rank}_{name}_{'reuse' if enabled else 'reference'}"})
    return result


def audit(root):
    runtime = json.loads((root / "runtime.json").read_text())
    config = runtime["config"]
    if _sha256(Path(config["selection"])) != config["selection_sha256"]:
        raise ValueError("selection hash changed")
    if _sha256(Path(config["base_config"])) != config["base_config_sha256"]:
        raise ValueError("base config changed")
    selection = json.loads(Path(config["selection"]).read_text())
    base = json.loads(Path(config["base_config"]).read_text())
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    expected = jobs(selection, config["ranks"])
    if len(rows) != len(expected) or any(any(r[k] != e[k] for k in e) for r, e in zip(rows, expected)):
        raise ValueError("incomplete, duplicated or unregistered jobs")
    packages = unsafe = 0
    details = {}
    for row in rows:
        if row["status"] not in {"SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"} or row["budget_seconds"] != config["budget_seconds"]:
            raise ValueError("worker error or budget drift")
        name = "reuse" if row["reuse"] else "reference"
        registered = {**base, "scoped_proof_reuse": row["reuse"]}
        if json.loads((root / f"{name}.json").read_text()) != registered:
            raise ValueError("unregistered method config")
        row["reused_rows"] = row["f0_solved_rows"] = None
        if row["package"]:
            package = Path(row["package"])
            if _sha256(package / "manifest.json") != row["manifest_sha256"]:
                raise ValueError("package hash changed")
            checked = audit_evidence_package(package, replay_unsafe=True)
            if checked["status"] != "PASS":
                raise ValueError(f"package audit failed: {checked}")
            e = json.loads((package / "evidence.json").read_text())
            if (e["identity"]["checkpoint"]["sha256"] != selection["models"][row["model"]]["checkpoint_sha256"] or
                    e["identity"]["config_sha256"] != _canonical_sha256(registered) or
                    e["verdict"]["status"] != row["status"] or
                    e["execution"]["dataset_index"] != row["dataset_index"] or
                    e["request"]["epsilon"] != selection["request"]["epsilon"] or
                    e["numerical_safety"] != base["numerical_safety"]):
                raise ValueError("executed identity mismatch")
            key = (row["model"], row["rank"])
            literal_identity = {k: e["identity"][k] for k in ("model_state", "lower", "upper", "center", "property")}
            if key in details and literal_identity != details[key]:
                raise ValueError("paired literal task differs")
            details[key] = literal_identity
            reused = solved = 0
            for pair in e["tier2"].get("pairs", []):
                for prop in pair["property_rows"]:
                    if prop.get("solver_bound_kind") == "scoped_tier1_interval":
                        audit_reused_property(prop, pair["pair"], e)
                        reused += 1
                    else:
                        solved += 1
            if reused != e["tier2"].get("reused_property_count", 0) or (not row["reuse"] and reused):
                raise ValueError("reuse accounting mismatch")
            row.update(reused_rows=reused, f0_solved_rows=solved)
            packages += 1
            unsafe += row["status"] == "UNSAFE"
        elif not row["outer_timeout"] or row["status"] != "TIMEOUT":
            raise ValueError("missing non-timeout package")
    models = {}
    for model in sorted(selection["models"]):
        by_arm = {arm: {r["rank"]: r for r in rows if r["model"] == model and r["reuse"] == arm} for arm in (False, True)}
        contrasts = {}
        for rank in config["ranks"]:
            if {by_arm[a][rank]["status"] for a in (False, True)} == {"SAFE", "UNSAFE"}:
                raise ValueError("paired SAFE/UNSAFE conflict")
        for label, statuses in (("SAFE", {"SAFE"}), ("solved", {"SAFE", "UNSAFE"})):
            gained = [i for i in config["ranks"] if by_arm[True][i]["status"] in statuses and by_arm[False][i]["status"] not in statuses]
            lost = [i for i in config["ranks"] if by_arm[False][i]["status"] in statuses and by_arm[True][i]["status"] not in statuses]
            contrasts[label] = {"gained": gained, "lost": lost, "net": len(gained)-len(lost)}
        differences = [by_arm[True][i]["wall_seconds"] - by_arm[False][i]["wall_seconds"] for i in config["ranks"]]
        methods = {}
        for arm, values in by_arm.items():
            methods["reuse" if arm else "reference"] = {
                "statuses": dict(Counter(r["status"] for r in values.values())),
                "mean_observed_seconds": mean(r["wall_seconds"] for r in values.values()),
                "known_reused_rows": sum(r["reused_rows"] or 0 for r in values.values()),
                "known_f0_solved_rows": sum(r["f0_solved_rows"] or 0 for r in values.values()),
                "unknown_counter_requests": sum(r["reused_rows"] is None for r in values.values()),
            }
        models[model] = {"methods": methods, "contrasts": contrasts,
                         "median_observed_time_difference_seconds": median(differences),
                         "mean_observed_time_difference_seconds": mean(differences)}
    return {"status": "PASS", "issues": [], "rows": len(rows), "packages": packages, "unsafe_replayed": unsafe,
            "models": models, "rows_sha256": _sha256(root / "rows.jsonl"), "runtime_sha256": _sha256(root / "runtime.json"),
            "scope": "Observed-cohort engineering comparison; bounded cost and censored counters, no independent network proof or holdout claim."}


def run(path):
    if _git_value("branch", "--show-current") != "feat/moe-route-verification" or _git_value("status", "--porcelain"):
        raise RuntimeError("clean feature branch required")
    config = json.loads(path.read_text())
    if Path(sys.executable).resolve() != Path(config["python"]).resolve():
        raise RuntimeError("act-py312 required")
    for p, h in ((config["selection"], config["selection_sha256"]), (config["base_config"], config["base_config_sha256"])):
        if _sha256(Path(p)) != h:
            raise RuntimeError("registered artifact changed")
    selection = json.loads(Path(config["selection"]).read_text())
    base = json.loads(Path(config["base_config"]).read_text())
    for subject in selection["models"].values():
        if _sha256(_inside(Path(subject["checkpoint"]), WRITE_ROOT)) != subject["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed")
    output = _inside(Path(config["output"]), WRITE_ROOT)
    output.mkdir(exist_ok=False)
    source = source_identity()
    save(output / "runtime.json", {"config": config, "config_sha256": _sha256(path), "source_sha256": source,
                                  "git_head": _git_value("rev-parse", "HEAD")})
    for name, enabled in (("reference", False), ("reuse", True)):
        save(output / f"{name}.json", {**base, "scoped_proof_reuse": enabled})
    env = {**os.environ, "ACT_TORCHVISION_DATA_ROOT": str(PROJECT_ROOT / "data/torchvision"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    registered_jobs = jobs(selection, config["ranks"])
    for number, job in enumerate(registered_jobs, 1):
        if source_identity() != source or _git_value("status", "--porcelain"):
            raise RuntimeError("source changed during frozen run")
        directory = output / job["job_id"]
        directory.mkdir()
        name = "reuse" if job["reuse"] else "reference"
        command = [config["python"], "-m", "act.pipeline.moe.staged_verifier", "--checkpoint", selection["models"][job["model"]]["checkpoint"],
                   "--dataset-index", str(job["dataset_index"]), "--epsilon", repr(selection["request"]["epsilon"]),
                   "--config", str(output / f"{name}.json"), "--output-dir", str(directory / "package"),
                   "--progress-path", str(directory / "progress.json")]
        started = time.monotonic()
        expired = False
        code = None
        with (directory / "worker.log").open("x") as log:
            try:
                code = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=config["budget_seconds"]).returncode
            except subprocess.TimeoutExpired:
                expired = True
        row = {**job, "wall_seconds": time.monotonic()-started, "outer_timeout": expired,
               "budget_seconds": config["budget_seconds"], "package": None, "return_code": code,
               "status": "TIMEOUT" if expired else "ERROR", "load_average_after": list(os.getloadavg())}
        if code == 0 and not expired:
            package = directory / "package"
            checked = audit_evidence_package(package, replay_unsafe=True)
            save(directory / "audit.json", checked)
            e = json.loads((package / "evidence.json").read_text())
            if checked["status"] == "PASS":
                row.update(package=str(package), status=e["verdict"]["status"], manifest_sha256=_sha256(package / "manifest.json"))
        save(directory / "terminal.json", row)
        with (output / "rows.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True)+"\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"{number}/{len(registered_jobs)} {job['job_id']} {row['status']}", flush=True)
        if row["status"] == "ERROR":
            raise RuntimeError("worker/audit error retained; stop without replacement")
    save(output / "audit.final.json", audit(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(args.audit), indent=2))
    else:
        run(args.config)
