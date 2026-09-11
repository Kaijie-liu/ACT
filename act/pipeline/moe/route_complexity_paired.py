"""Frozen two-arm schedule follow-up; fail-stop smoke gate, no resume/overwrite."""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _git_value, _sha256
from act.pipeline.moe.paired_followup import save, source_identity
from act.pipeline.moe.proof_reuse_paired import jobs as reuse_jobs
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.scoped_f0_proofs import audit_reused_property
from act.pipeline.moe.staged_verifier import _canonical_sha256

DEFAULT = PROJECT_ROOT / "act/pipeline/moe/configs/route_complexity_paired_r1.json"


def jobs(selection, ranks):
    result = []
    for old in reuse_jobs(selection, ranks):
        row = {k: v for k, v in old.items() if k not in {"reuse", "job_id"}}
        method = "adaptive" if old["reuse"] else "monolithic"
        row.update(method=method, job_id=f"rank{row['rank']}_{row['model']}_{method}")
        result.append(row)
    return result


def artifacts(config):
    path = _inside(Path(config["selection"]), WRITE_ROOT)
    if _sha256(path) != config["selection_sha256"]:
        raise ValueError("selection drift")
    selection = json.loads(path.read_text())
    if set(config["methods"]) != {"adaptive", "monolithic"}:
        raise ValueError("unregistered method list")
    configs = {}
    for name, record in config["methods"].items():
        path = _inside(Path(record["path"]), PROJECT_ROOT)
        if _sha256(path) != record["sha256"]:
            raise ValueError("method config drift")
        configs[name] = json.loads(path.read_text())
        if configs[name]["route_complexity_schedule"]["total_seconds"] != config["budget_seconds"]:
            raise ValueError("inner/outer budget mismatch")
    if {k for k in configs["adaptive"] if configs["adaptive"][k] != configs["monolithic"].get(k)} != {"comparison_method"}:
        raise ValueError("methods differ beyond comparison_method")
    if set(configs["adaptive"]) != set(configs["monolithic"]):
        raise ValueError("method config fields differ")
    if configs["adaptive"]["comparison_method"] != "staged" or configs["monolithic"]["comparison_method"] != "monolithic_f0":
        raise ValueError("arm name/config mismatch")
    for name in ("ranks", "smoke_ranks"):
        ranks = config[name]
        if not ranks or ranks != list(range(len(ranks))) or len(ranks) > len(selection["samples"]):
            raise ValueError("expected frozen observed prefix, no replacements")
    return selection, configs


def common_facts(e):
    if not e["route_complexity_schedule"]["common_fact_prelude_complete"]:
        return None
    # Request and router frame IDs are deliberately local to each execution.
    # Compare actual intervals and domain memberships, not foreign factor IDs.
    return {"candidates": e["route_coverage"]["candidate_experts"],
            "pairs": e["route_coverage"]["feasible_route_sets"],
            "facts": [{"expert": b["candidate"], "bounds": b["proof_output_bounds"],
                       "source_policy": b["source_policy"]} for b in e["tier1"]["branches"]],
            "available_count": e["proof_reuse"]["available_fact_count"]}


def counters(e):
    if e["tier2"].get("partial_rows_censored"):
        return None
    reused = queries = 0
    for pair in e["tier2"].get("pairs", []):
        for prop in pair["property_rows"]:
            if prop.get("solver_bound_kind") == "scoped_tier1_interval":
                audit_reused_property(prop, pair["pair"], e)
                reused += 1
            else:
                queries += 1
    for prop in e["tier2"].get("property_rows", []):
        for item in prop.get("coverage_partition", {}).get("reused", []):
            if item["proof"]["property_index"] != prop["property_index"]:
                raise ValueError("reused monolithic property mismatch")
            audit_reused_property(item["proof"], item["pair"], e)
            reused += 1
        queries += prop.get("pair_count", 0) > 0
    return {"reused_pair_properties": reused, "recorded_weighted_query_rows": queries}


def audit(root):
    runtime = json.loads((root / "runtime.json").read_text())
    config = runtime["config"]
    if _sha256(Path(runtime["config_path"])) != runtime["config_sha256"]:
        raise ValueError("registered experiment config drift")
    if json.loads(Path(runtime["config_path"]).read_text()) != config:
        raise ValueError("runtime config mismatch")
    selection, configs = artifacts(config)
    ranks = config["smoke_ranks" if runtime["smoke"] else "ranks"]
    expected = jobs(selection, ranks)
    rows = [json.loads(x) for x in (root/"rows.jsonl").read_text().splitlines()]
    if len(rows) != len(expected) or any(any(r[k] != j[k] for k in j) for r, j in zip(rows, expected)):
        raise ValueError("incomplete, duplicate or reordered schedule")
    details = {}; packages = replayed = 0; package_counts = Counter()
    for r in rows:
        if r["status"] not in {"SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"} or r["budget_seconds"] != config["budget_seconds"]:
            raise ValueError("error terminal or budget drift")
        if not math.isfinite(r["wall_seconds"]) or r["wall_seconds"] < 0:
            raise ValueError("invalid observed time")
        r.update(counters=None, path="MISSING_PACKAGE", pair_count=None, fact_prelude_complete=False)
        if r["outer_timeout"]:
            if r["status"] != "TIMEOUT" or r["package"] is not None:
                raise ValueError("outer timeout promoted to a result")
            continue
        if not r["package"] or r["return_code"] != 0:
            raise ValueError("missing successful package")
        p = _inside(Path(r["package"]), WRITE_ROOT)
        if p != root / r["job_id"] / "package" or _sha256(p/"manifest.json") != r["manifest_sha256"]:
            raise ValueError("package identity mismatch")
        checked = audit_evidence_package(p, replay_unsafe=True)
        if checked["status"] != "PASS":
            raise ValueError(f"package audit failed: {checked}")
        e = json.loads((p/"evidence.json").read_text()); cfg = configs[r["method"]]
        if (e["identity"]["checkpoint"]["sha256"] != selection["models"][r["model"]]["checkpoint_sha256"] or
            e["identity"]["config_sha256"] != _canonical_sha256(cfg) or
            e["execution"]["git_head"] != runtime["git_head"] or
            e["execution"]["dataset_index"] != r["dataset_index"] or
            e["request"]["epsilon"] != selection["request"]["epsilon"] or
            e["verdict"]["status"] != r["status"] or e["numerical_safety"] != cfg["numerical_safety"]):
            raise ValueError("executed task/config/checkpoint mismatch")
        detail = {"identity": {k: e["identity"][k] for k in ("model_state", "center", "lower", "upper", "property")},
                  "facts": common_facts(e)}
        details[r["model"], r["rank"], r["method"]] = detail
        r.update(counters=counters(e), path=e["route_complexity_schedule"]["selected_path"],
                 pair_count=len(e["route_coverage"]["feasible_route_sets"]) if e["route_coverage"]["route_sets_exact"] else None,
                 fact_prelude_complete=detail["facts"] is not None)
        packages += 1; replayed += r["status"] == "UNSAFE"; package_counts[r["method"]] += 1
    models = {}; fact_matches = 0
    for model in sorted(selection["models"]):
        by = {arm: {r["rank"]: r for r in rows if r["model"] == model and r["method"] == arm} for arm in configs}
        matched = missing = 0
        for rank in ranks:
            if {by[a][rank]["status"] for a in configs} == {"SAFE", "UNSAFE"}:
                raise ValueError("SAFE/UNSAFE conflict")
            a, b = [details.get((model, rank, arm)) for arm in ("adaptive", "monolithic")]
            if a is not None and b is not None:
                if a["identity"] != b["identity"]: raise ValueError("paired literal task differs")
                if a["facts"] is not None and b["facts"] is not None:
                    if a["facts"] != b["facts"]: raise ValueError("common fact prelude differs")
                    matched += 1; continue
            missing += 1
        fact_matches += matched
        contrasts = {}
        for name, statuses in (("SAFE", {"SAFE"}), ("solved", {"SAFE", "UNSAFE"})):
            gained = [i for i in ranks if by["adaptive"][i]["status"] in statuses and by["monolithic"][i]["status"] not in statuses]
            lost = [i for i in ranks if by["monolithic"][i]["status"] in statuses and by["adaptive"][i]["status"] not in statuses]
            contrasts[name] = {"gained": gained, "lost": lost, "net": len(gained)-len(lost)}
        methods = {}
        for arm in configs:
            values=list(by[arm].values())
            methods[arm] = {"statuses": dict(Counter(r["status"] for r in values)),
                "mean_observed_seconds": mean(r["wall_seconds"] for r in values),
                "paths": dict(Counter(r["path"] for r in values)),
                "route_changing_safe_ranks": [r["rank"] for r in values if r["status"] == "SAFE" and r["pair_count"] is not None and r["pair_count"] > 1],
                "censored_counter_requests": sum(r["counters"] is None for r in values),
                "known_reused_pair_properties": sum(r["counters"]["reused_pair_properties"] for r in values if r["counters"] is not None),
                "known_weighted_query_rows": sum(r["counters"]["recorded_weighted_query_rows"] for r in values if r["counters"] is not None)}
        differences=[by["adaptive"][i]["wall_seconds"]-by["monolithic"][i]["wall_seconds"] for i in ranks]
        models[model] = {"methods": methods, "contrasts": contrasts, "common_fact_pairs_equal": matched,
                         "common_fact_pairs_unavailable": missing,
                         "mean_paired_observed_seconds": mean(differences), "median_paired_observed_seconds": median(differences)}
    return {"status": "PASS", "issues": [], "rows": len(rows), "packages": packages, "unsafe_replayed": replayed,
            "package_counts": dict(package_counts), "common_fact_pairs_equal": fact_matches, "models": models,
            "rows_sha256": _sha256(root/"rows.jsonl"), "runtime_sha256": _sha256(root/"runtime.json"),
            "scope": "Observed-cohort engineering; 10 shared inputs in full run, not 60 independent samples. Structural audit, not independent SAFE proof."}


def smoke_gate(path, config, source):
    root = _inside(Path(config["smoke_output"]), WRITE_ROOT)
    if not (root/"runtime.json").is_file() or not (root/"audit.final.json").is_file():
        raise ValueError("full run requires completed audited smoke")
    rt=json.loads((root/"runtime.json").read_text())
    if not rt["smoke"] or rt["config"] != config or rt["config_sha256"] != _sha256(path) or rt["source_sha256"] != source:
        raise ValueError("smoke identity mismatch")
    checked=audit(root)
    if checked != json.loads((root/"audit.final.json").read_text()):
        raise ValueError("smoke audit changed")
    if any(checked["package_counts"].get(a,0)==0 for a in config["methods"]) or checked["common_fact_pairs_equal"] == 0:
        raise ValueError("smoke supplies no paired common-fact conformance")
    return {"root": str(root), "audit_sha256": _sha256(root/"audit.final.json")}


def run(path, smoke=False):
    if _git_value("branch", "--show-current") != "feat/moe-route-verification" or _git_value("status", "--porcelain"):
        raise RuntimeError("clean feature branch required")
    path=_inside(path,PROJECT_ROOT); config=json.loads(path.read_text()); selection, configs=artifacts(config)
    if Path(sys.executable).resolve() != Path(config["python"]).resolve(): raise ValueError("act-py312 required")
    for subject in selection["models"].values():
        if _sha256(_inside(Path(subject["checkpoint"]),WRITE_ROOT)) != subject["checkpoint_sha256"]: raise ValueError("checkpoint drift")
    source=source_identity(); gate=None
    if not smoke: gate=smoke_gate(path,config,source)
    root=_inside(Path(config["smoke_output" if smoke else "output"]),WRITE_ROOT); root.mkdir(exist_ok=False)
    save(root/"runtime.json", {"config": config, "config_path": str(path), "config_sha256": _sha256(path),
                              "source_sha256":source,"git_head":_git_value("rev-parse","HEAD"),"smoke":smoke,"smoke_gate":gate})
    env={**os.environ,"ACT_TORCHVISION_DATA_ROOT":str(PROJECT_ROOT/"data/torchvision"),
         "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    schedule=jobs(selection,config["smoke_ranks" if smoke else "ranks"])
    for number, job in enumerate(schedule,1):
        if source_identity()!=source or _git_value("status","--porcelain"):raise RuntimeError("source/worktree drift")
        artifacts(config)
        directory=root/job["job_id"];directory.mkdir()
        command=[config["python"],"-m","act.pipeline.moe.staged_verifier","--checkpoint",selection["models"][job["model"]]["checkpoint"],
                 "--dataset-index",str(job["dataset_index"]),"--epsilon",repr(selection["request"]["epsilon"]),
                 "--config",config["methods"][job["method"]]["path"],"--output-dir",str(directory/"package"),
                 "--progress-path",str(directory/"progress.json")]
        started=time.monotonic();expired=False;code=None
        with (directory/"worker.log").open("x") as log:
            try: code=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=config["budget_seconds"]).returncode
            except subprocess.TimeoutExpired: expired=True
        row={**job,"wall_seconds":time.monotonic()-started,"outer_timeout":expired,"return_code":code,
             "budget_seconds":config["budget_seconds"],"status":"TIMEOUT" if expired else "ERROR","package":None,
             "load_average_after":list(os.getloadavg())}
        try:
            if code==0 and not expired:
                checked=audit_evidence_package(directory/"package",replay_unsafe=True);save(directory/"audit.json",checked)
                if checked["status"]!="PASS": raise ValueError(str(checked))
                e=json.loads((directory/"package/evidence.json").read_text())
                row.update(status=e["verdict"]["status"],package=str(directory/"package"),manifest_sha256=_sha256(directory/"package/manifest.json"))
        except Exception as exc: row["error"]=f"{type(exc).__name__}: {exc}"
        save(directory/"terminal.json",row)
        with (root/"rows.jsonl").open("a") as f:
            f.write(json.dumps(row,sort_keys=True)+"\n");f.flush();os.fsync(f.fileno())
        print(f"{'smoke' if smoke else 'full'} {number}/{len(schedule)} {job['job_id']} {row['status']}",flush=True)
        if row["status"]=="ERROR":raise RuntimeError("retained worker/audit failure; no replacement")
    save(root/"audit.final.json",audit(root))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT)
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--smoke",action="store_true");mode.add_argument("--pipeline",action="store_true");mode.add_argument("--audit",type=Path)
    args=parser.parse_args()
    if args.audit: print(json.dumps(audit(_inside(args.audit,WRITE_ROOT)),indent=2));return
    # Separate invocations cannot create concurrent timing writers.
    lock=PROJECT_ROOT/"data/moe/results/route_complexity_pairing.lock"
    with lock.open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.pipeline: run(args.config,True);run(args.config,False)
        else: run(args.config,args.smoke)


if __name__=="__main__":main()
