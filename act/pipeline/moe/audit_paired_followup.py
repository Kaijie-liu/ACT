"""Recheck four-method paired evidence and per-model discordant outcomes."""
import argparse
import copy
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.experiment1 import _sha256
from act.pipeline.moe.staged_verifier import _canonical_sha256


def audit(root):
    runtime = json.loads((root / "runtime.json").read_text())
    config = runtime["config"]
    issues = []
    selection_path = Path(config["selection"])
    if _sha256(selection_path) != config["selection_sha256"]:
        issues.append("selection identity changed")
    selection = json.loads(selection_path.read_text())
    base_path = Path(config["base_config"])
    if _sha256(base_path) != config["base_config_sha256"]:
        issues.append("base config identity changed")
    base = json.loads(base_path.read_text())
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    methods = ("staged", "route_invariance", "monolithic_f0", "tier1_only")
    method_configs = {}
    for method in methods:
        expected_config = copy.deepcopy(base)
        expected_config["comparison_method"] = method
        if method == "tier1_only":
            expected_config["tier1"]["solver"]["escalation_budget_per_branch"] = config["budget_seconds"]
        if method == "monolithic_f0":
            expected_config["f0"]["solver"]["property_seconds"] = config["budget_seconds"]
        actual = json.loads((root / f"{method}.json").read_text())
        if actual != expected_config:
            issues.append(f"unregistered method options: {method}")
        method_configs[method] = actual
    models = sorted(selection["models"])
    ranks = config["smoke_ranks"] if runtime["smoke"] else list(range(100))
    expected = []
    for rank in ranks:
        for i in range(3):
            model = models[(rank + i) % 3]
            for j in range(4):
                method = methods[(rank + models.index(model) + j) % 4]
                expected.append(f"rank{rank}_{model}_{method}")
    if [r["job_id"] for r in rows] != expected:
        issues.append("incomplete, duplicate or out-of-order jobs")
    packages = witnesses = 0
    statuses = {}
    for row in rows:
        if row["dataset_index"] != selection["samples"][row["rank"]]["dataset_index"]:
            issues.append(f"input mismatch: {row['job_id']}")
        if row["budget_seconds"] != config["budget_seconds"]:
            issues.append(f"budget mismatch: {row['job_id']}")
        if row["status"] not in {"SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"}:
            issues.append(f"execution error: {row['job_id']}")
        if row["package"]:
            package = Path(row["package"])
            if _sha256(package / "manifest.json") != row["manifest_sha256"]:
                issues.append(f"manifest mismatch: {row['job_id']}")
            result = audit_evidence_package(package, replay_unsafe=True)
            issues.extend(f"{row['job_id']}: {issue}" for issue in result["issues"])
            evidence = json.loads((package / "evidence.json").read_text())
            if (evidence["identity"]["config_sha256"] != _canonical_sha256(method_configs[row["method"]]) or
                    evidence["numerical_safety"] != base["numerical_safety"] or
                    evidence["execution"]["config_sha256"] != _sha256(root / f"{row['method']}.json")):
                issues.append(f"executed config/numerical contract mismatch: {row['job_id']}")
            model = selection["models"][row["model"]]
            if evidence["identity"]["checkpoint"]["sha256"] != model["checkpoint_sha256"]:
                issues.append(f"checkpoint mismatch: {row['job_id']}")
            if (evidence["algorithm"].get("comparison_method") != row["method"] or
                evidence["request"]["epsilon"] != selection["request"]["epsilon"] or
                evidence["execution"]["dataset_index"] != row["dataset_index"] or
                evidence["verdict"]["status"] != row["status"]):
                issues.append(f"method/request/verdict mismatch: {row['job_id']}")
            packages += 1
            witnesses += row["status"] == "UNSAFE"
        elif row["status"] != "TIMEOUT" or not row["outer_timeout"]:
            issues.append(f"non-timeout without package: {row['job_id']}")
    for model in models:
        values = {m: {r["rank"]: r["status"] for r in rows if r["model"] == model and r["method"] == m}
                  for m in methods}
        contrasts = {}
        for a, b in combinations(methods, 2):
            for rank in sorted(set(values[a]) & set(values[b])):
                if {values[a][rank], values[b][rank]} == {"SAFE", "UNSAFE"}:
                    issues.append(f"SAFE/UNSAFE conflict: {model}/{rank}/{a}/{b}")
        for baseline in methods[1:]:
            common = sorted(set(values["staged"]) & set(values[baseline]))
            contrast = {}
            for name, accepted in (("SAFE", {"SAFE"}), ("solved", {"SAFE", "UNSAFE"})):
                gained = [r for r in common if values["staged"][r] in accepted and values[baseline][r] not in accepted]
                lost = [r for r in common if values["staged"][r] not in accepted and values[baseline][r] in accepted]
                contrast[name] = {"gained": gained, "lost": lost, "net": len(gained)-len(lost)}
            contrasts[baseline] = contrast
        statuses[model] = {"counts": {m: dict(Counter(v.values())) for m, v in values.items()},
                           "contrasts": contrasts}
    return {"status": "FAIL" if issues else "PASS", "issue_count": len(issues), "issues": issues,
            "rows": len(rows), "expected_rows": len(expected), "packages_audited": packages,
            "unsafe_replayed": witnesses, "models": statuses,
            "runtime_sha256": _sha256(root / "runtime.json"),
            "rows_sha256": _sha256(root / "rows.jsonl"),
            "scope": "Structural identity and coverage audit with UNSAFE replay; not independent SAFE proof checking."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        from act.pipeline.moe.experiment1 import _inside, WRITE_ROOT
        from act.pipeline.moe.paired_followup import save
        output = _inside(args.output, WRITE_ROOT)
        if output.exists():
            raise RuntimeError("refusing to overwrite audit")
        save(output, result)
    if result["issues"]:
        raise SystemExit(1)
