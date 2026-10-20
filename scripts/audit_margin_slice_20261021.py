"""Saved-only audit of the exact slice interface, never online status repair."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from scoped_source.hz_request_store import compact, require
from scoped_source.development_work import DevelopmentWork, POLICY
import margin_slice_cases as cases


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124)); signal.setitimer(signal.ITIMER_REAL, 60)
    path = ROOT/"docs/margin_slice_records_20261021.json"
    archive = json.loads(path.read_bytes())
    frozen_raw = (ROOT/"docs/margin_slice_freeze_20261021.json").read_bytes()
    frozen = json.loads(frozen_raw)
    for name, identity in frozen["files"].items():
        require(sha((ROOT/name).read_bytes()) == identity, "frozen file "+name)
    for name, identity in frozen["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted((ROOT/name).rglob("*.py"))}
        require(sha(compact(members)) == identity, "frozen tree "+name)
    for name, item in archive["json"].items():
        raw = compact(item)
        require(sha(raw) == archive["files"][name]["sha256"] and len(raw) == archive["files"][name]["bytes"],
                "lossless artifact "+name)
    for original, moved in frozen["portable_files"].items():
        require(archive["files"]["relocated/"+moved]["sha256"] == frozen["files"][original], "portable code identity")
    require(set(archive["files"]) == set(archive["json"]) | {"relocated/"+v for v in frozen["portable_files"].values()},
            "all retained files accounted")
    result, worker, portable = (archive["json"][k] for k in ("result.json", "worker.json", "relocated/receipt.json"))
    require(sha(frozen_raw) == result["freeze_sha256"] == worker["freeze_sha256"], "frozen execution binding")
    require(archive["outer"]["exit_code"] == 0 and archive["outer"]["through_terminal_export_seconds"] < 300
            and result["status"] == worker["status"] == portable["status"] == "PASS"
            and sha(compact(result)) == archive["outer"]["terminal_sha256"], "original complete online terminal")
    require(result["fds_before"] == result["fds_after"] and result["caps"] == frozen["caps"], "FD/limits")
    require(len(result["sessions"]) == 2 and worker["native_candidate_calls"] == 2
            and worker["full_calls_attempted"] == 2 and worker["objective_slots"] == 8, "native call counts")
    for item, name in zip(result["sessions"], ("native", "portable")):
        r = item["receipt"]
        require(item["name"] == name and r["status"] == "COMPLETED" and r["returncode"] == 0
                and r["execution_error"] is None and r["cleanup_error"] is None
                and r["remaining_session"]["live"] == [] and r["wait4_usage"] is not None
                and r["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION" and not r["rss_acceptance_gate"]
                and r["finished_at"] <= r["cleanup_deadline"] <= result["deadline"], "owned session "+name)
    require(worker["deadline"] == result["deadline"] == portable["deadline"]
            and worker["rlimit_as"] == [8*2**30]*2, "continued deadline and AS")
    work = DevelopmentWork(time.monotonic()+45, policy=POLICY)
    receipts = cases.receive(worker, work)
    require(receipts == result["online_reception"]["receipts"] == portable["receipts"], "independent receptions")
    counts = [worker["work_counts"], result["online_reception"]["work_counts"], portable["work_counts"]]
    require(all(c[k] >= p[k] for p, c in zip(counts, counts[1:]) for k in p), "no Work reset")
    totals = [sum(c.values()) for c in counts]
    require(totals[1]-totals[0] == result["online_reception"]["work"]
            and totals[2]-totals[1] == portable["reception_work"], "receiver cost accounting")
    run_bytes = sum(v["bytes"] for v in archive["files"].values())
    require(run_bytes+path.stat().st_size < frozen["caps"]["artifact_bytes"], "cumulative artifact cap")
    require(not any(n in sys.modules for n in ("act", "torch", "numpy", "scipy")), "stdlib saved audit")
    print(json.dumps(dict(status="PASS", offline_only=True, native_calls_not_repeated=True,
                         native_candidate_calls=2, objective_slots=8, work_counts=totals,
                         run_bytes=run_bytes, archive_bytes=path.stat().st_size,
                         original_lower=receipts["native"]["lower"], complete_request=False,
                         elapsed_seconds=time.monotonic()-start), sort_keys=True))


if __name__ == "__main__":
    main()
