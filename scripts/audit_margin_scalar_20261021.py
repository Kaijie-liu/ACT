"""Saved-only sparse scalar audit: no native execution or online status repair."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from scoped_source.hz_request_store import compact, require
from scoped_source.development_work import DevelopmentWork, POLICY
from audit_margin_native_20261021 import direct_lower
import margin_scalar_cases as cases


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def direct_scalar(record):
    """Second scalar formulation on saved supports, not either implementation."""
    goals = record["supports"]
    l, h = map(F, record["plan"]["anchors"])
    results = []
    for item in record["cells"]:
        x, y = F(item["x"]), F(item["y"])
        gate = (F(1, 2) if x >= 0 else max(F(0), F(1, 2)+x/4),
                F(1, 2) if y <= 0 else min(F(1), F(1, 2)+y/4))
        require(list(map(F, item["gate"])) == list(gate), "second gate check")
        ends = [max(min(F(s["lower"])-F(s["nu"])*mu for mu in (x, y))
                    for s in goals if F(s["t"]) == t) for t in (l, h)]
        lower = min((h-g)/(h-l)*ends[0]+(g-l)/(h-l)*ends[1] for g in gate)
        require(lower == F(item["lower"]), "second scalar cell arithmetic")
        results.append(lower)
    require(min(results) == F(record["lower"]), "second scalar aggregate")
    return min(results)


def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124)); signal.setitimer(signal.ITIMER_REAL, 60)
    archive_path = ROOT/"docs/margin_scalar_records_20261021.json"
    archive = json.loads(archive_path.read_bytes())
    frozen_path = ROOT/"docs/margin_scalar_freeze_20261021.json"
    frozen = json.loads(frozen_path.read_bytes())
    result, worker = archive["result"], archive["worker"]
    require(sha(frozen_path.read_bytes()) == result["freeze_sha256"], "original scalar freeze")
    for name, identity in frozen["files"].items():
        require(sha((ROOT/name).read_bytes()) == identity, "frozen file "+name)
    for name, identity in frozen["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted((ROOT/name).rglob("*.py"))}
        require(sha(compact(members)) == identity, "frozen tree "+name)
    reconstructed = {"result.json": result, "worker.json": worker,
                     "relocated/candidate.json": worker, "relocated/receipt.json": archive["portable_receipt"],
                     "relocated/continued.json": result["online_reception"]["work_counts"]}
    for name, item in archive["prefixes"].items():
        reconstructed[name] = dict(item["fields"], source=worker["source"], view=worker["view"],
                                   records={n: worker["records"][n] for n in item["records"]})
    for session in result["sessions"]:
        reconstructed[session["name"]+"_session.json"] = session["receipt"]
    for name, value in reconstructed.items():
        raw = compact(value)
        require(sha(raw) == archive["files"][name]["sha256"]
                and len(raw) == archive["files"][name]["bytes"], "lossless artifact "+name)
    for original, moved in frozen["portable_files"].items():
        require(archive["files"]["relocated/"+moved]["sha256"] == frozen["files"][original], "portable code")
    require(set(archive["files"]) == set(reconstructed) | {"relocated/"+v for v in frozen["portable_files"].values()},
            "complete retained roster")
    require(archive["outer"]["exit_code"] == 0 and result["status"] == worker["status"] == "PASS"
            and archive["outer"]["through_terminal_export_seconds"] < 300
            and sha(compact(result)) == archive["outer"]["terminal_sha256"], "original outer terminal")
    require(result["fds_before"] == result["fds_after"] and result["caps"] == frozen["caps"], "FD/limits")
    require(len(result["sessions"]) == 2 and worker["full_calls_attempted"] == 2
            and worker["native_candidate_calls"] == 2 and worker["objective_slots"] == 6, "call accounting")
    for item, name in zip(result["sessions"], ("native", "portable")):
        r = item["receipt"]
        require(item["name"] == name and r["status"] == "COMPLETED" and r["returncode"] == 0
                and r["execution_error"] is None and r["cleanup_error"] is None
                and r["remaining_session"]["live"] == []
                and r["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION" and r["wait4_usage"] is not None
                and not r["rss_acceptance_gate"]
                and r["finished_at"] <= r["cleanup_deadline"] <= result["deadline"], "owned session "+name)
    require(worker["deadline"] == result["deadline"] == archive["portable_receipt"]["deadline"]
            and worker["rlimit_as"] == [8*2**30]*2, "same deadline and AS")
    offline = DevelopmentWork(time.monotonic()+30, policy=POLICY)
    receipts = cases.receive(worker, offline)
    require(receipts == result["online_reception"]["receipts"] == archive["portable_receipt"]["receipts"],
            "all three original exact receptions")
    count = 0
    for name in cases.MODES:
        record = worker["records"][name]["record"]
        for candidate in record["supports"]:
            require(direct_lower(worker["source"]["augmented"], worker["view"], candidate)
                    == F(candidate["lower"]), "second original-coordinate residual")
            count += 1
        require(direct_scalar(record) == F(receipts[name]["lower"]), "second scalar result")
    counts = [worker["work_counts"], result["online_reception"]["work_counts"],
              archive["portable_receipt"]["work_counts"]]
    require(all(c[k] >= p[k] for p, c in zip(counts, counts[1:]) for k in p), "no reset Work")
    totals = list(map(lambda c: sum(c.values()), counts))
    require(totals[1]-totals[0] == result["online_reception"]["work"]
            and totals[2]-totals[1] == archive["portable_receipt"]["reception_work"], "continued cost")
    run_bytes = sum(v["bytes"] for v in archive["files"].values())
    archive_bytes = archive_path.stat().st_size
    require(run_bytes+archive_bytes < frozen["caps"]["artifact_bytes"], "cumulative artifacts")
    require(not any(n in sys.modules for n in ("act", "torch", "numpy", "scipy")), "stdlib saved audit")
    print(json.dumps(dict(status="PASS", offline_only=True, original_online_status=result["status"],
                          native_calls_not_repeated=True, saved_support_records=count,
                          scalar_records=len(receipts), work_counts=totals,
                          run_bytes=run_bytes, archive_bytes=archive_bytes,
                          independent_audit_seconds=time.monotonic()-start,
                          complete_request=False, scalar_aggregate_checked=True), sort_keys=True))


if __name__ == "__main__":
    main()
