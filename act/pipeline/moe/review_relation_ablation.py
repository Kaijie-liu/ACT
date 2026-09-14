"""Read-only re-audit and compact archive of relation R1; no new solves."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from act.pipeline.moe.experiment1 import PROJECT_ROOT, _sha256
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.relation_ablation import audit
from act.pipeline.moe.route_complexity_paired import counters

HEAD = "5bdbd25b26dfd2227fb1b838d7f31e30a65f9c74"
BASE = PROJECT_ROOT/"data/moe/results"
OUTPUT = PROJECT_ROOT/"act/pipeline/moe/results/relation_ablation_review_20260914_r1.json"


def ref(path):
    return {"path": str(path), "sha256": _sha256(path)}


def source_identity():
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", HEAD, "act"], cwd=PROJECT_ROOT, text=True)
    digest = hashlib.sha256()
    for name in sorted(n for n in names.splitlines() if n.endswith(".py")):
        digest.update(f"{name}:{_sha256(PROJECT_ROOT/name)}\n".encode())
    return digest.hexdigest()


def weighted_rows(e):
    rows = list(e["tier2"].get("property_rows", []))
    for pair in e["tier2"].get("pairs", []): rows.extend(pair.get("property_rows", []))
    return rows


def gates(e):
    result = {}
    def add(pair, margin, lam):
        key = str(tuple(pair)); value = {"margin": margin, "lambda": lam}
        if key in result and result[key] != value: raise ValueError("inconsistent gate within request")
        result[key] = value
    for pair in e["tier2"].get("pairs", []):
        for row in pair.get("property_rows", []):
            if "margin_bounds" in row: add(pair["pair"], row["margin_bounds"], row["lambda_bounds"])
    for row in e["tier2"].get("property_rows", []):
        for p in row.get("expert_pair_factors", []): add(p["pair"], p["margin_bounds"], p["lambda_bounds"])
    return result


def build():
    result = {"execution_head": HEAD, "execution_source_sha256": source_identity(), "experiments": {}}
    for phase in ("smoke", "full"):
        root = BASE/f"relation_ablation_{phase}_20260914_r1"
        runtime = json.loads((root/"runtime.json").read_text())
        if runtime["git_head"] != HEAD or runtime["source_sha256"] != result["execution_source_sha256"]:
            raise ValueError("frozen executable sources changed; use the archived execution revision")
        checked = audit(root)
        if checked != json.loads((root/"audit.final.json").read_text()): raise ValueError("saved audit does not reconstruct")
        compact = {k: v for k, v in checked.items() if k != "provenance"}
        result["experiments"][phase] = {"runtime": ref(root/"runtime.json"), "audit": ref(root/"audit.final.json"), "summary": compact}
    root = BASE/"relation_ablation_full_20260914_r1"
    rows = [json.loads(s) for s in (root/"rows.jsonl").read_text().splitlines()]
    details = {}; missing = []; gate_records = {}; state_counts = {}
    for row in rows:
        key = (row["model"], row["rank"], row["method"])
        d = {k: row[k] for k in ("job_id", "dataset_index", "status", "wall_seconds", "outer_timeout")}
        d.update(terminal=ref(root/row["job_id"]/"terminal.json"), evidence=None, pair_count=None, counters=None)
        if row["package"]:
            path = Path(row["package"])/"evidence.json"; e = json.loads(path.read_text())
            d.update(evidence=ref(path), pair_count=len(e["route_coverage"]["feasible_route_sets"]),
                     reason=e["verdict"]["reason"], tier=e["verdict"]["decision_tier"], counters=counters(e),
                     weighted_rows=[{k: p.get(k) for k in ("property_index", "status", "reason", "solver_status", "accepted_minimum", "minimum", "expert_pair_factors")}
                                    for p in weighted_rows(e)])
            gate_records[key] = gates(e)
        else: missing.append(d)
        details[key] = d
    comparisons = []; gate_equal = gate_different = gate_one_sided = 0
    for model in ("seed0", "seed1", "seed2"):
        for rank in range(10):
            a, b = [details[model, rank, arm] for arm in ("shared", "independent")]
            if a["status"] != b["status"]: comparisons.append({"model": model, "rank": rank, "shared": a, "independent": b})
            ga, gb = [gate_records.get((model, rank, arm), {}) for arm in ("shared", "independent")]
            gate_one_sided += len(set(ga)^set(gb))
            for pair in set(ga)&set(gb):
                if ga[pair] == gb[pair]: gate_equal += 1
                else: gate_different += 1
    for arm in ("shared", "independent"):
        rr = [r for r in rows if r["method"] == arm]
        state_counts[arm] = dict(Counter(r["status"] for r in rr))
    result.update(status="PASS", issues=[], exact_reaudit_equals_saved=True,
                  full_states=state_counts, discordances=comparisons, missing_packages=missing,
                  recorded_gate_pairs={"equal": gate_equal, "different": gate_different, "one_sided": gate_one_sided,
                                       "scope": "Only gate ranges surviving in property records; unrecorded is not equality."},
                  scope="10 observed images, 30 model-input pairs, not independent confirmation; HZ/HiGHS numerical policy, not independently re-proved SAFE.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write and OUTPUT.exists(): raise ValueError("never overwrite archived results")
    value = build()
    if args.write: save(OUTPUT, value)
    elif value != json.loads(OUTPUT.read_text()): raise ValueError("archive differs")
    print(json.dumps({"status": value["status"], "recorded_gate_pairs": value["recorded_gate_pairs"], "archive": str(OUTPUT)}))


if __name__ == "__main__": main()
