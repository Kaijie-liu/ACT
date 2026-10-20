"""Saved-only A2 audit. Never repairs an online failure or reruns a solver."""
import base64
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from scoped_source.hz_request_store import compact, require
from scoped_source.development_work import DevelopmentWork, POLICY
import margin_a2_cases as cases


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124))
    signal.setitimer(signal.ITIMER_REAL, 60)
    archive_path = ROOT/"docs/margin_a2_records_20261021.json"
    archive = json.loads(archive_path.read_bytes())
    frozen_raw = (ROOT/"docs/margin_a2_freeze_20261021.json").read_bytes()
    frozen = json.loads(frozen_raw)
    for name, identity in frozen["files"].items():
        require(sha((ROOT/name).read_bytes()) == identity, "frozen code "+name)
    for name, identity in frozen["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted((ROOT/name).rglob("*.py"))}
        require(sha(compact(members)) == identity, "frozen tree "+name)
    require(archive["schema"] == "A2_DEDUPLICATED_LOSSLESS_ARCHIVE_V1", "archive schema")
    pool = {}
    for identity, item in archive["objects"].items():
        require(item["bytes"] <= 2*2**20, "bounded archived object")
        dec = zlib.decompressobj()
        raw = dec.decompress(base64.b64decode(item["zlib_base64"], validate=True), item["bytes"]+1)
        require(dec.eof and not dec.unused_data and not dec.unconsumed_tail
                and len(raw) == item["bytes"] and sha(raw) == identity, "lossless bounded object")
        pool[identity] = raw
    total = 0
    for name, item in archive["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and item["sha256"] in pool and len(pool[item["sha256"]]) == item["bytes"], "archive file roster")
        total += item["bytes"]
    require(total+archive_path.stat().st_size <= frozen["caps"]["artifact_bytes"], "cumulative archive capacity")
    def read(name):
        return json.loads(pool[archive["files"][name]["sha256"]])
    result = read("result.json")
    require(result["freeze_sha256"] == sha(frozen_raw)
            and archive["outer"]["terminal_sha256"] == archive["files"]["result.json"]["sha256"], "terminal identity")
    require(archive["outer"]["exit_code"] == (0 if result["status"] == "PASS" else 1)
            and archive["outer"]["through_terminal_export_seconds"] < 300, "outer exit and deadline")
    if result["status"] != "PASS":
        require(result["decision"] == "INCONCLUSIVE_MISSING_REQUIRED_ARM", "no failed-prefix victory")
        print(json.dumps(dict(status="AUDITED_FAILURE_NOT_REPAIRED", reason=result.get("error"),
                             complete_cost_and_Work=False, run_bytes=total, offline_only=True)))
        return
    worker, portable = read("worker.json"), read("relocated/receipt.json")
    require(worker["status"] == portable["status"] == "PASS" and worker["freeze_sha256"] == sha(frozen_raw)
            and worker["deadline"] == portable["deadline"] == result["deadline"]
            and worker["rlimit_as"] == [8*2**30]*2 and result["fds_before"] == result["fds_after"], "complete online envelope")
    cases.duration(result["entry"], result["terminal_clock"], result["deadline"], "saved terminal")
    require(result["elapsed_before_terminal_export"] == result["terminal_clock"]-result["entry"], "terminal duration")
    cases.duration(worker["worker_entry"], worker["worker_finished_before_export"], worker["deadline"], "saved worker")
    require(worker["elapsed_before_worker_export"] == worker["worker_finished_before_export"]-worker["worker_entry"], "worker duration")
    for item, name in zip(result["sessions"], ("native", "portable")):
        r = item["receipt"]
        require(item["name"] == name and r["status"] == "COMPLETED" and r["returncode"] == 0
                and r["execution_error"] is None and r["cleanup_error"] is None
                and r["remaining_session"]["live"] == [] and r["wait4_usage"] is not None
                and r["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION" and not r["rss_acceptance_gate"]
                and r["finished_at"] <= r["cleanup_deadline"] <= result["deadline"], "owned session "+name)
    require(len(result["sessions"]) == 2, "complete session roster")
    for original, target in frozen["portable_files"].items():
        require(archive["files"]["relocated/"+target]["sha256"] == frozen["files"][original], "portable code hash")
    for n in ["common.json", "worker.json"]+["arm_"+a+".json" for a in cases.ARMS]:
        require(archive["files"][n] == archive["files"]["relocated/"+n], "portable data hash")
    require(set(worker["faults"]) == set(cases.FAULTS) and all(
        worker["faults"][k]["type"] == "ValueError" and t in worker["faults"][k]["reason"]
        for k,t in cases.FAULTS.items()), "accounting faults")
    counts = cases.check_events(worker)
    for i, event in enumerate(worker["events"], 1):
        require(read("event_%03d.json" % i) == event, "durable original event")
    work = DevelopmentWork(time.monotonic()+50, policy=POLICY)
    common = read("common.json"); raw, view = compact(common["source"]), compact(common["view"])
    require(sha(raw) == common["source_sha256"] and sha(view) == common["property_sha256"], "given source/view")
    require(common["source"]["scope"] == cases.SCOPE and
            all(common["source"]["base"]["sources"][k] == v for k,v in cases.fixtures().items())
            and common["view"]["q"] == cases.PROPERTY["q"]
            and common["view"]["constant"] == cases.PROPERTY["constant"], "fixed fixture/property")
    artifacts, receipts = {}, {}
    for name in cases.ARMS:
        item = worker["arms"][name]
        require(item["file"] == "arm_"+name+".json"
                and {k: item[k] for k in ("sha256","bytes")} == archive["files"][item["file"]], "arm file binding")
        cases.duration(item["started_at"], item["finished_at"], worker["deadline"], name)
        require(item["seconds"] == item["finished_at"]-item["started_at"], "arm duration")
        artifact = read(item["file"])
        receipt = cases.check_arm(raw, view, common, name, artifact, work, artifacts.get("finite"))
        cases.check_queries(raw, view, common, name, artifact, worker["events"], work)
        require(receipt == artifact["receipt"] == result["online_reception"]["receipts"][name]
                == portable["receipts"][name], "saved independent arithmetic "+name)
        artifacts[name], receipts[name] = artifact, receipt
    chain = [worker["work_counts"], result["online_reception"]["work_counts"], portable["work_counts"]]
    require(all(c[k] >= p[k] for p,c in zip(chain,chain[1:]) for k in p), "continued Work")
    totals = [sum(c.values()) for c in chain]
    require(totals[1]-totals[0] == result["online_reception"]["work"]
            and totals[2]-totals[1] == portable["reception_work"], "reception Work increments")
    cost = result["costs"]
    variable = {n: worker["arms"][n]["seconds"]+result["online_reception"]["arm_seconds"][n]+portable["arm_seconds"][n]
                for n in cases.ARMS}
    measured = cases.duration(result["entry"],cost["cut"],result["deadline"],"saved cost")
    shared = measured-sum(variable.values())
    price = {n: shared+v for n,v in variable.items()}; price["transfer"] += variable["finite"]
    require(shared >= 0 and variable == cost["variable_seconds"] and price == cost["standalone_through_cut"]
            and measured == cost["measured_batch_seconds"] and shared == cost["shared_seconds"], "full cost reconstruction")
    conservative = {n: v+result["deadline"]-cost["cut"] for n,v in price.items()}
    # Match the producer's grouped subtraction, preserving binary64 evaluation order.
    conservative = {n: v+(result["deadline"]-cost["cut"]) for n,v in price.items()}
    require(conservative == cost["decision_costs"], "tail-bound costs")
    verdict = cases.decision(receipts, conservative)
    descriptive = cases.decision(receipts, price)
    if verdict["action"] == "GRAY_NO_AUTOMATIC_EXPANSION" and descriptive["action"].startswith("LIMITED_MECHANISM_SIGNAL"):
        verdict["cost_qualification"] = "UNRESOLVED_BY_CONSERVATIVE_TERMINAL_TAIL_NOT_COST_FAILURE"
    require(verdict == result["decision"] and descriptive == result["through_cut_decision"], "frozen decision")
    require(not any(k in sys.modules for k in ("act", "numpy", "scipy", "torch")), "stdlib saved audit")
    print(json.dumps(dict(status="PASS", offline_only=True, decision=verdict, native_calls=counts,
                         lowers={k:v["lower"] for k,v in receipts.items()}, work_counts=totals,
                         run_bytes=total, archive_bytes=archive_path.stat().st_size,
                         elapsed=time.monotonic()-start, complete_request=False), sort_keys=True))


if __name__ == "__main__":
    main()
