"""Saved-only audit; no model/solver or offline completion of online evidence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from scoped_proof.io import load, sha, save
from scoped_source.native_capture import canonical, require, GRADE
from b1_request_capture_cases import REQUESTS, FAULTS


def audit(run):
    freeze = ROOT/"docs/b1_request_capture_freeze_20261021.json"
    spec = load(freeze, limit=2**20)
    result = load(run/"result.json", limit=2**18)
    require(result["freeze_sha256"] == sha(freeze), "freeze identity")
    for name, digest in spec["files"].items():
        require(sha(ROOT/name) == digest, "source identity: "+name)
    for name, digest in spec["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/name).rglob("*.py"))}
        require(hashlib.sha256(canonical(members)).hexdigest() == digest, "tree identity")
    require(result["status"] == "PASS" and not result["trained_source"] and not result["gpu"]
            and not result["checked_feasibility"] and not result["algorithm_gain"]
            and result["real_request_slots"] == 0, "grade/status")
    value = load(run/"worker.json", result["worker_sha256"], 2**20)
    require(value["status"] == "PASS" and value["deadline"] == result["deadline"]
            and value["rlimit_as"] == [8*2**30]*2, "worker budget/identity")
    require([v["name"] for v in value["requests"]] == list(REQUESTS)
            and len(value["events"]) == 8 and value["captured_sources"] == 2
            and value["observed_complete_synthetic_requests"] == 4, "full request denominator")
    for i, name in enumerate(REQUESTS):
        enter, returned = value["events"][2*i:2*i+2]
        require(enter["name"] == returned["name"] == name and enter["phase"] == "ENTER"
                and returned["phase"] == "RETURN" and returned["status"] == "UNKNOWN"
                and enter["clock"]+returned["seconds"] < result["deadline"], "request terminal/cutoff")
        evidence = load(run/(name+"_evidence.json"), value["requests"][i]["evidence_receipt"]["sha256"], 2**20)
        require(evidence["verdict"]["status"] == "UNKNOWN"
                and evidence["verdict"]["reason"] == returned["reason"]
                and not evidence["verdict"]["full_model_witness_valid"]
                and evidence["route_coverage"]["feasible_route_sets"] == [[0,1],[0,2],[1,2]], "fixed fixture")
        if name.endswith("_off"):
            require("diagnostics" not in evidence, "default off")
            continue
        diag = evidence["diagnostics"]
        latest = [v for v in value["publications"][name] if v["kind"] == "inventory"][-1]
        require(diag == load(run/latest["filename"], latest["sha256"],2**20), "published final inventory")
        states = [v["state"] for v in diag["obligations"]]
        require(diag["terminal"]["status"] == "UNKNOWN" and diag["terminal"]["coverage"] == "COMPLETE"
                and diag["roster"]["coverage_complete"] and len(diag["roster"]["properties"]) == 9
                and len(states) == 27 and states.count("REUSED") == 25
                and states.count("SELECTED") == states.count("RETURNED") == 1, "complete obligation partition")
        require(diag["registered_f0_calls"] == (2 if name == "staged_on" else 1)
                and diag["selected"]["pair"] == [0,1] and diag["selected"]["property_index"] == 0
                and diag["selected"]["source_available"], "fixed first selection and slots")
        found = []
        for item in value["publications"][name]:
            data = load(run/item["filename"], item["sha256"], 2**20)
            if item["kind"] != "native":
                require(data["binding"] == diag["binding"] and data["roster_sha256"] == diag["roster_sha256"],
                        "inventory scope")
                continue
            require(data["grade"] == GRADE and data["deadline"] == result["deadline"]
                    and data["scope"] == diag["selected"]["receipt"]["scope"], "source scope/grade")
            require(set(data["sources"]) >= {"upstream_guarded_entry", "upstream_conditioned_router",
                                            "upstream_expert_joint"}, "all stored upstream objects")
            meta = data["source_metadata"]
            require(meta["a_rows"] == list(range(10)) and meta["b_rows"] == list(range(10,20))
                    and meta["margin_relax_binaries"] is False and meta["difference_relax_binaries"] is True,
                    "full class and range-domain semantics")
            require(item["sha256"] == diag["selected"]["receipt"]["sha256"], "selected native identity")
            found.append(item["sha256"])
        require(len(found) == 1, "at most one selected capture")
    for off, on in ((0,1),(2,3)):
        keys = ("status","reason","decision_tier","pairs","coverage","full_model_witness_valid")
        require(all(value["requests"][off][k] == value["requests"][on][k] for k in keys), "off/on equality")
    require([c["name"] for c in value["controls"]] == list(FAULTS), "fixed control roster/order")
    for kind, rows in (("event",value["events"]),("fault",value["controls"])):
        for i, row in enumerate(rows,1):
            require(load(run/(kind+"_%02d.json" % i),limit=4096) == row, "durable "+kind)
    require(all(c["status"] == "PASS" for c in value["controls"]), "controls pass")
    counts = [value["work_counts"],result["online_reception"]["work_counts"]]
    require(all(counts[1][k] >= counts[0][k] for k in counts[0]), "continued Work")
    receipts = [load(run/(n+"_evidence.json"),limit=2**20)["diagnostics"]["selected"]["receipt"]
                for n in ("staged_on","monolithic_f0_on")]
    require(receipts == result["online_reception"]["receipts"], "online structural reception")
    require(len(result["sessions"]) == 1, "one owned native worker")
    r = result["sessions"][0]["receipt"]
    require(load(run/"native_session.json",limit=2**18) == r and r["status"] == "COMPLETED"
            and r["returncode"] == 0 and r["execution_error"] is r["cleanup_error"] is None
            and r["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION" and r["wait4_usage"] is not None,
            "owned completion/cleanup")
    require(result["fds_before"] == result["fds_after"] and result["terminal_clock"] < result["deadline"],
            "final descriptor/deadline")
    files = {str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(run.rglob("*")) if p.is_file()}
    size = sum(v["bytes"] for v in files.values())
    require(2*(size+2**20) <= 128*2**20, "run plus archive cap")
    return dict(status="PASS", scope="SAVED_STRUCTURAL_AUDIT_NOT_PROOF", synthetic_requests=4,
                controls=len(FAULTS), selected_sources=2, obligation_denominator_per_request=27,
                files=files,run_bytes=size,execution_head=result["execution_head"],
                work_counts=counts,freeze_sha256=result["freeze_sha256"],real_stage_usage=[0,0,0],
                elapsed_before_terminal_export=result["elapsed_before_terminal_export"])


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve()
    require(run == Path("/data1/Kane/MOE/results/b1_request_capture_20261021_r1"), "frozen path")
    report = audit(run)
    if len(sys.argv) == 3 and sys.argv[2] == "--archive":
        records = {str(p.relative_to(run)):load(p,limit=2**20) for p in sorted(run.rglob("*.json"))}
        archive = dict(audit=report,records=records)
        raw = canonical(archive)
        require(len(raw)+report["run_bytes"] <= 128*2**20 and len(raw) <= 8*2**20, "archive cap")
        report["archive"] = save(ROOT/"docs/b1_request_capture_records_20261021.json",archive)
    print(json.dumps(report,sort_keys=True))
