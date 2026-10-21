"""Saved-only audit of frozen B1 capture controls; never reruns a solver."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from scoped_proof.io import load, sha, save
from scoped_source.native_capture import canonical, require, GRADE
from b1_native_capture_cases import NATIVE_SLOTS, FAULTS


def audit(run):
    freeze = ROOT/"docs/b1_native_capture_freeze_20261021.json"
    spec = load(freeze, limit=2**20)
    result = load(run/"result.json", limit=2**18)
    require(result["freeze_sha256"] == sha(freeze), "freeze identity")
    for name, digest in spec["files"].items():
        require(sha(ROOT/name) == digest, "execution source identity: "+name)
    for name, digest in spec["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/name).rglob("*.py"))}
        require(hashlib.sha256(canonical(members)).hexdigest() == digest, "execution tree identity")
    require(result["status"] == "PASS" and not result["complete_request"] and
            not result["checked_feasibility"] and not result["algorithm_gain"], "scope/status")
    value = load(run/"worker.json", result["worker_sha256"], 2**20)
    require(value["status"] == "PASS" and value["deadline"] == result["deadline"] and
            value["rlimit_as"] == [8*2**30]*2, "worker scope/limits")
    require(len(value["events"]) == 16 and len(value["controls"]) == len(FAULTS), "full denominator")
    for i, name in enumerate(NATIVE_SLOTS):
        a, b = value["events"][2*i:2*i+2]
        require(a["slot"] == b["slot"] == name and a["phase"] == "ENTER" and b["phase"] == "RETURN",
                "slot events")
        require(a["clock"]+b["seconds"] < value["deadline"], "call completion before deadline")
        if name.endswith("capture_deadline"):
            require(b["result"]["expected_exception"] == "CaptureDeadline", "expected cutoff")
    for i, event in enumerate(value["events"], 1):
        require(load(run/("event_%02d.json" % i), limit=4096) == event, "durable events")
    require({c["name"] for c in value["controls"]} == set(FAULTS), "fault roster")
    for i, ctrl in enumerate(value["controls"], 1):
        require(ctrl["status"] == "PASS" and load(run/("control_%02d.json" % i), limit=4096) == ctrl,
                "durable fault receipt")
    require(set(value["comparisons"]) == {"pair", "mono", "empty"} and
            all(v["equal"] for v in value["comparisons"].values()), "off/on unchanged")
    require(value["comparisons"]["pair"]["decision"]["status"] == "SAFE" and
            value["comparisons"]["mono"]["decision"]["status"] == "UNKNOWN" and
            value["comparisons"]["empty"]["decision"]["status"] == "UNKNOWN", "fixed fixture decisions")
    receipts = []
    for r in value["records"]:
        raw = (run/r["name"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == r["sha256"] and
                sha(run/"relocated"/r["name"]) == r["sha256"], "original/relocated identity")
        data = json.loads(raw)
        require(data["grade"] == GRADE and data["deadline"] == result["deadline"]
                and data["scope"] == r["scope"], "record grade/deadline/scope")
        receipts.append(r["receipt"])
    require(len(receipts) == 10 and receipts == result["online_reception"]["receipts"]
            == result["portable_reception"]["receipts"], "complete online/portable receipts")
    require(load(run/"relocated"/"receipt.json", limit=2**18) == result["portable_reception"],
            "saved portable receipt")
    counts = [value["work_counts"], result["online_reception"]["work_counts"],
              result["portable_reception"]["work_counts"]]
    require(all(all(b[k] >= a[k] for k in a) for a, b in zip(counts, counts[1:])), "continued Work")
    require(result["fds_before"] == result["fds_after"], "descriptor balance")
    require([s["name"] for s in result["sessions"]] == ["native", "portable"], "two bounded sessions")
    for item in result["sessions"]:
        r = item["receipt"]
        require(load(run/(item["name"]+"_session.json"), limit=2**18) == r and
                r["status"] == "COMPLETED" and r["returncode"] == 0
                and r["execution_error"] is None and r["cleanup_error"] is None
                and r["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
                and r["wait4_usage"] is not None, "owned completion/cleanup")
    require(result["terminal_clock"] < result["deadline"] and
            result["elapsed_before_terminal_export"] <= 300, "online budget")
    files = {str(p.relative_to(run)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(run.rglob("*")) if p.is_file()}
    size = sum(v["bytes"] for v in files.values())
    require(2*(size+2**20) <= 128*2**20, "run plus archive cap")
    return dict(status="PASS", scope="SAVED_RECORD_AND_BUDGET_AUDIT_NOT_PROOF", native_slots=8,
                ordinary_native_slots=6, expected_post_native_cutoffs=2, controls=len(FAULTS),
                structural_records=10, elapsed_before_terminal_export=result["elapsed_before_terminal_export"],
                files=files, run_bytes=size, execution_head=result["execution_head"],
                work_counts=counts, freeze_sha256=result["freeze_sha256"], real_stage_usage=[0, 0, 0])


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve()
    require(run == Path("/data1/Kane/MOE/results/b1_native_capture_20261021_r1"), "frozen result path")
    report = audit(run)
    if len(sys.argv) == 3 and sys.argv[2] == "--archive":
        # One lossless JSON archive. Script copies are hash-bound to the execution
        # commit; they need not be duplicated a third time inside the JSON file.
        records = {str(p.relative_to(run)): load(p, limit=2**20)
                   for p in sorted(run.rglob("*.json"))}
        path = ROOT/"docs/b1_native_capture_records_20261021.json"
        raw = canonical(dict(audit=report, records=records))
        require(len(raw)+report["run_bytes"] <= 128*2**20 and len(raw) <= 8*2**20, "archive total cap")
        report["archive"] = save(path, dict(audit=report, records=records))
    print(json.dumps(report, sort_keys=True))
