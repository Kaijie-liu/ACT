"""Read-only stdlib recheck of archived M3 records; NEVER recovers online success."""
import time
START = time.monotonic()
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from scoped_source.hz_request_store import Work, compact, digest, require
from scoped_proof.io import load


def main(path, expected_hash):
    archive = load(path, expected_hash, 8*2**20)
    require(archive["schema"] == "GATE_MARGIN_SAVED_ARCHIVE_V1", "archive schema")
    terminal, worker = archive["terminal"], archive["worker"]
    members = {m["name"]: m for m in archive["members"]}
    require(len(members) == len(archive["members"]), "unique saved files")
    for name, value in (("result.json", terminal), ("worker.json", worker)):
        raw = compact(value)
        require(len(raw) == members[name]["bytes"] and digest(raw) == members[name]["sha256"],
                "lossless canonical original")
    require(terminal["worker"]["sha256"] == members["worker.json"]["sha256"]
            and terminal["freeze_sha256"] == worker["freeze_sha256"], "terminal/worker binding")
    freeze = load(ROOT/"docs/gate_margin_rows_freeze_20261020.json", terminal["freeze_sha256"], 4*2**20)
    require(worker["counts"] == freeze["expected_counts"] and worker["status"] == "PASS", "complete worker")
    spec = importlib.util.spec_from_file_location("m3_saved_cases", ROOT/"scripts/gate_margin_rows_cases.py")
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    require(digest((ROOT/"scripts/gate_margin_rows_cases.py").read_bytes())
            == freeze["files"]["scripts/gate_margin_rows_cases.py"], "frozen check adapter")
    cases.inventory(worker["records"], terminal["deadline"], worker["shared_deadline"], worker["work_counts"])
    cases.all_faults(worker["records"], worker["faults"], worker["work_counts"])
    # A new OFFLINE audit budget; never splice this into the expired online ledger.
    work = Work(START+300)
    reports = {}
    for row in worker["records"]:
        raw = compact(row); member = members[row["case"]+".json"]
        require(digest(raw) == member["sha256"] and len(raw) == member["bytes"], "saved record bytes")
        report = cases.verify_row(row, work)
        require(report == row["algebra"], "saved exact algebra")
        reports[row["case"]] = report
    work.tick()
    return dict(status="OFFLINE_SAVED_CHECKS_PASS", online_status=terminal["status"],
                online_reception_recovered=False, freeze_sha256=terminal["freeze_sha256"],
                execution_head=terminal["execution_head"], archive_sha256=expected_hash,
                record_count=len(reports), recorded_expected_refusals=len(worker["faults"]),
                independent_fault_rerun=False, audit_work_counts=work.counts,
                audit_work_total=sum(work.counts.values()), reports=reports,
                audit_seconds_before_stdout=time.monotonic()-START,
                guarantee="given-HZ/LP saved algebra, not network SAFE or recovered execution acceptance")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("archive path and mandatory SHA-256")
    print(json.dumps(main(sys.argv[1], sys.argv[2]), sort_keys=True))
