"""Descriptive analysis of a completed, hash-bound four-arm follow-up.

No verification queries, fitted schedule, solved-only denominator or pooled
independence assumption. Costs are observed budget-limited execution times,
not estimates of uncensored time to solve.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median

from act.pipeline.moe.experiment1 import _sha256, _inside, WRITE_ROOT
from act.pipeline.moe.paired_followup import METHODS, save, summarize


def describe_rows(rows, ranks):
    models = sorted({r["model"] for r in rows})
    if len(models) != 3:
        raise ValueError("three models required")
    expected = {(model, method, rank) for model in models for method in METHODS for rank in ranks}
    keys = [(r["model"], r["method"], r["rank"]) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("incomplete or duplicate paired task")
    if any(r["status"] not in {"SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"} or
           not math.isfinite(r["wall_seconds"]) or r["wall_seconds"] < 0 for r in rows):
        raise ValueError("invalid status or elapsed time")
    result = summarize(rows, len(expected))
    for model in models:
        part = {method: {r["rank"]: r for r in rows
                         if r["model"] == model and r["method"] == method} for method in METHODS}
        for rank in ranks:
            statuses = {part[method][rank]["status"] for method in METHODS}
            if {"SAFE", "UNSAFE"}.issubset(statuses):
                raise ValueError("conflicting SAFE and UNSAFE")
        for method, values in part.items():
            detail = result["models"][model]["methods"][method]
            detail.update(
                mean_observed_wall_seconds=mean(r["wall_seconds"] for r in values.values()),
                median_observed_wall_seconds=median(r["wall_seconds"] for r in values.values()),
                outer_timeout_count=sum(r["outer_timeout"] for r in values.values()),
                reason_counts=dict(Counter(r["reason"] for r in values.values())),
                solved_count=sum(r["status"] in {"SAFE", "UNSAFE"} for r in values.values()),
            )
        for baseline in METHODS[1:]:
            paired = result["models"][model]["paired"][baseline]
            differences = [part["staged"][i]["wall_seconds"] - part[baseline][i]["wall_seconds"] for i in ranks]
            paired["mean_observed_wall_difference_seconds"] = mean(differences)
            paired["median_observed_wall_difference_seconds"] = median(differences)
    result["cost_scope"] = "Observed bounded execution, all requests including timeouts; post-run audit excluded; shared-server timing, not time-to-solve or unconditional speedup."
    result["statistical_scope"] = "Descriptive per-model paired counts, no post-hoc significance gate; 100 shared inputs, not 1200 independent samples."
    return result


def analyze(root, audit_path):
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "PASS" or audit["issues"]:
        raise ValueError("passing independent audit required")
    for name in ("rows", "runtime"):
        suffix = "jsonl" if name == "rows" else "json"
        if _sha256(root / f"{name}.{suffix}") != audit[f"{name}_sha256"]:
            raise ValueError("audit/raw hash mismatch")
    runtime = json.loads((root / "runtime.json").read_text())
    if runtime["smoke"]:
        raise ValueError("smoke is not an effectiveness experiment")
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    if audit["rows"] != 1200 or audit["expected_rows"] != 1200:
        raise ValueError("complete four-arm audit required")
    result = describe_rows(rows, list(range(100)))
    original = json.loads((root / "summary.json").read_text())
    if summarize(rows, 1200) != original:
        raise ValueError("runner summary disagrees with raw rows")
    result.update(raw_root=str(root), experiment_git_head=runtime["git_head"],
                  experiment_source_sha256=runtime["source_sha256"],
                  config=runtime["config"], audit=audit, audit_sha256=_sha256(audit_path),
                  summary_sha256=_sha256(root / "summary.json"))
    evidence = {}
    for row in rows:
        if row["package"]:
            package = Path(row["package"])
            if _sha256(package / "manifest.json") != row["manifest_sha256"]:
                raise ValueError("package changed after audit")
            evidence[(row["model"], row["method"], row["rank"])] = json.loads((package / "evidence.json").read_text())
    for model, record in result["models"].items():
        for method, detail in record["methods"].items():
            detail["route_changing_safe_ranks"] = [
                r["rank"] for r in rows if r["model"] == model and r["method"] == method and
                r["status"] == "SAFE" and len(evidence[(model, method, r["rank"])]["route_coverage"]["feasible_route_sets"]) > 1]
        # Association only: what blocked staged on monolithic-only SAFE inputs?
        lost = record["paired"]["monolithic_f0"]["safe"]["baseline_only_ranks"]
        lookup = {r["rank"]: r for r in rows if r["model"] == model and r["method"] == "staged"}
        record["monolithic_only_safe_staged_reasons"] = dict(Counter(lookup[i]["reason"] for i in lost))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _inside(args.output, WRITE_ROOT)
    if output.exists():
        raise RuntimeError("refusing to overwrite derived result")
    result = analyze(args.root, args.audit)
    save(output, result)
    print(json.dumps({model: value["methods"] for model, value in result["models"].items()}, indent=2))
