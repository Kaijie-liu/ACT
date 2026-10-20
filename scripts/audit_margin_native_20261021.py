"""Saved-only A1 audit: no native import, optimizer, child launch or status repair."""
import hashlib
import importlib.util
import json
from fractions import Fraction as F
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoped_source.hz_request_store import compact, require
from scoped_source.development_work import DevelopmentWork, POLICY


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def direct_lower(hz, view, candidate):
    """Second coordinate formulation: not the b/d/m checker implementation."""
    q = [F(v) for v in view["q"]]
    t, nu = F(candidate["t"]), F(candidate["nu"])
    weights = [t*v for v in q]+[(1-t)*v for v in q]+[nu, -nu]
    nc, nb = hz["Gc"]["shape"][1], hz["Gb"]["shape"][1]
    residual = [F(0)]*(nc+nb)
    lower = F(view["constant"])
    for i, weight in enumerate(weights):
        lower += weight*F(hz["c"][i])
        for key, shift in (("Gc", 0), ("Gb", nc)):
            m = hz[key]
            for p in range(m["indptr"][i], m["indptr"][i+1]):
                residual[shift+m["indices"][p]] += weight*F(m["data"][p])
    for duals, rhs, names in ((candidate["y"], hz["ub"], ("Auc", "Aub")),
                             (candidate["z"], hz["b"], ("Ac", "Ab"))):
        require(len(duals) == len(rhs), "direct all dual rows")
        for i, v in enumerate(duals):
            d = F(v)
            require(names[0] == "Ac" or d <= 0, "direct inequality sign")
            lower += d*F(rhs[i])
            for key, shift in zip(names, (0, nc)):
                m = hz[key]
                for p in range(m["indptr"][i], m["indptr"][i+1]):
                    residual[shift+m["indices"][p]] -= d*F(m["data"][p])
    return lower-sum(map(abs, residual), F(0))


def main():
    start = time.monotonic()
    record_path = ROOT/"docs/margin_native_records_20261021.json"
    record = json.loads(record_path.read_bytes())
    freeze_path = ROOT/"docs/margin_native_freeze_20261021.json"
    freeze = json.loads(freeze_path.read_bytes())
    result, worker = record["result"], record["worker"]
    require(sha(freeze_path.read_bytes()) == result["freeze_sha256"], "original freeze")
    for name, identity in freeze["files"].items():
        require(sha((ROOT/name).read_bytes()) == identity, "frozen file "+name)
    for name, identity in freeze["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted((ROOT/name).rglob("*.py"))}
        require(sha(compact(members)) == identity, "frozen tree "+name)
    reconstructed = {"result.json": result, "worker.json": worker,
                     "relocated/candidate.json": worker, "relocated/receipt.json": record["portable_receipt"],
                     "relocated/continued.json": result["online_reception"]["work_counts"]}
    for name, item in record["prefixes"].items():
        reconstructed[name] = dict(item["fields"], source=worker["source"],
                                   supports=worker["supports"][:item["support_count"]])
    for session in result["sessions"]:
        reconstructed[session["name"]+"_session.json"] = session["receipt"]
    for name, value in reconstructed.items():
        raw = compact(value)
        require(sha(raw) == record["files"][name]["sha256"]
                and len(raw) == record["files"][name]["bytes"], "lossless archive "+name)
    for original, moved in freeze["portable_files"].items():
        require(record["files"]["relocated/"+moved]["sha256"] == freeze["files"][original], "portable code")
    require(set(record["files"]) == set(reconstructed) | {"relocated/"+v for v in freeze["portable_files"].values()},
            "complete artifact roster")
    require(record["outer"]["exit_code"] == 0 and result["status"] == worker["status"] == "PASS"
            and record["outer"]["through_terminal_export_seconds"] < 300
            and sha(compact(result)) == record["outer"]["terminal_sha256"], "online terminal and outer evidence")
    require(result["fds_before"] == result["fds_after"] and result["caps"] == freeze["caps"], "FD/limits")
    expected = [("native", "COMPLETED", 0), ("portable", "COMPLETED", 0),
                ("exit7", "ERROR", 7), ("deadline", "TIMEOUT", -9)]
    for item, (name, status, code) in zip(result["sessions"], expected):
        r = item["receipt"]
        require(item["name"] == name and r["status"] == status and r["returncode"] == code
                and r["cleanup_error"] is None and r["remaining_session"]["live"] == []
                and r["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
                and r["wait4_usage"] is not None and not r["rss_acceptance_gate"]
                and r["finished_at"] <= r["cleanup_deadline"] <= result["deadline"], "owned terminal "+name)
    require(len(result["sessions"]) == 4 and worker["native_candidate_calls"] == 2, "fixed native roster")
    spec = importlib.util.spec_from_file_location("a1_saved_cases", ROOT/"scripts/margin_native_cases.py")
    cases = importlib.util.module_from_spec(spec); spec.loader.exec_module(cases)
    offline = DevelopmentWork(time.monotonic()+60, policy=POLICY)
    receipts = cases.receive(worker, offline)
    require(receipts == result["online_reception"]["receipts"] == record["portable_receipt"]["receipts"],
            "saved exact receipt equality")
    count = 0
    for prop in worker["supports"]:
        for candidate in prop["candidates"]:
            require(direct_lower(worker["source"]["augmented"], prop["view"], candidate)
                    == F(candidate["lower"]), "independent direct-coordinate residual")
            count += 1
    before = sum(worker["work_counts"].values())
    online = sum(result["online_reception"]["work_counts"].values())
    portable = sum(record["portable_receipt"]["work_counts"].values())
    require(online-before == result["online_reception"]["work"]
            and portable-online == record["portable_receipt"]["reception_work"], "continued accounting")
    run_bytes = sum(v["bytes"] for v in record["files"].values())
    archive_bytes = record_path.stat().st_size
    require(run_bytes+archive_bytes < freeze["caps"]["artifact_bytes"], "cumulative archive budget")
    print(json.dumps(dict(status="PASS", saved_supports=count, offline_only=True,
                          native_calls_not_repeated=True, original_online_status=result["status"],
                          work_counts=[before, online, portable], run_bytes=run_bytes,
                          archive_bytes=archive_bytes, independent_audit_seconds=time.monotonic()-start,
                          grade=receipts[0]["grade"], scalar_aggregate_checked=False,
                          complete_request=False), sort_keys=True))


if __name__ == "__main__":
    main()
