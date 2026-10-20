"""Frozen A2 scientific comparison; reuse the existing owned-session supervisor."""
import time
ENTRY = time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from margin_native_control import limits, successful, ENV, PYTHON, POLICY, CAPS
from scoped_proof.io import load, save, sha
from scoped_source.hz_request_store import compact, require

FREEZE = ROOT/"docs/margin_a2_freeze_20261021.json"
RUN = Path("/data1/Kane/MOE/results/margin_a2_strong_20261021_r1")


def binding(anchor):
    import margin_a2_cases as cases
    spec = load(FREEZE, anchor, 2**20)
    require(spec["policy"] == POLICY and spec["caps"] == CAPS and spec["attempts"] == 1
            and spec["result_directory"] == str(RUN) and spec["arms"] == list(cases.ARMS)
            and spec["query_slots"] == cases.SLOTS and spec["native_call_limit"] == cases.TOTAL_CALLS
            and spec["query_seconds"] == cases.QUERY_SECONDS and spec["global_cells"] == cases.GLOBAL_CELLS
            and spec["minimum_effect"] == str(cases.EFFECT) and spec["cost_ratio"] == cases.COST_RATIO
            and spec["faults"] == cases.FAULTS
            and spec["cuts"] == {str(k): list(map(str, cases.cuts_for(k))) for k in (2, 4)},
            "frozen A2 scope and decision")
    require(sum((ROOT/p).stat().st_size for p in spec["portable_files"]) <= 2**20,
            "portable source capacity")
    for name, identity in spec["files"].items():
        require(sha(ROOT/name) == identity, "frozen code drift: "+name)
    for name, identity in spec["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/name).rglob("*.py"))}
        require(hashlib.sha256(compact(members)).hexdigest() == identity, "frozen tree: "+name)
    return spec


def size():
    return sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())


def bounded_save(path, value, cap=2**18):
    raw = compact(value)
    # At most one extra lossless archive copy, plus reserved terminal output.
    require(len(raw) <= cap and 2*(size()+2*len(raw)+2**20) <= CAPS["artifact_bytes"],
            "A2 publication and archive capacity")
    return save(path, value)


def worker(anchor, deadline, phase_deadline):
    limits(phase_deadline)
    binding(anchor)
    import margin_a2_cases as cases
    def publish(name, value, cap):
        permitted = ({"common.json": 2**20}
                     | {"arm_"+n+".json": cases.ARM_LIMIT for n in cases.ARMS}
                     | {"event_%03d.json" % n: 4096 for n in range(1, 2*cases.TOTAL_CALLS+1)}
                     | {"progress_%02d.json" % n: 65536 for n in range(1, len(cases.ARMS)+1)})
        require(permitted.get(name) == cap, "fixed bounded A2 file roster")
        return bounded_save(RUN/name, value, cap)
    value = cases.run(deadline, publish)
    value.update(freeze_sha256=anchor, phase_deadline=phase_deadline,
                 rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)))
    receipt = bounded_save(RUN/"worker.json", value)
    require(time.monotonic() < phase_deadline, "worker export deadline")
    print(json.dumps(dict(status=value["status"], worker=receipt)), flush=True)
    return 0 if value["status"] == "PASS" else 1


def main(anchor):
    deadline = ENTRY+300
    limits(deadline)
    from scoped_proof import owned_session as owned
    from scoped_source.development_work import DevelopmentWork
    import margin_a2_cases as cases
    result = dict(schema="MARGIN_A2_CONTROL_RESULT_V1", status="FAIL", policy=POLICY,
                  freeze_sha256=anchor, entry=ENTRY, deadline=deadline, caps=CAPS,
                  full_cold_start_seconds=None, trained_source=False, gpu=False,
                  complete_request=False, sessions=[], decision="INCONCLUSIVE_MISSING_REQUIRED_ARM")
    created = False
    try:
        require(sys.executable == PYTHON, "existing act-py312")
        frozen = binding(anchor)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           timeout=min(5, max(.001, deadline-time.monotonic()))).strip()
        require(git("branch", "--show-current") == "feat/moe-route-verification"
                and not git("status", "--porcelain"), "clean frozen branch")
        result["execution_head"] = git("rev-parse", "HEAD")
        available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        disk = shutil.disk_usage(RUN.parent).free
        require(time.monotonic() < ENTRY+60 and available >= 4*2**30 and disk >= CAPS["artifact_bytes"],
                "one fresh bounded admission")
        result["admission"] = dict(seconds=time.monotonic()-ENTRY, available_bytes=available, disk_bytes=disk)
        RUN.mkdir(); created = True
        result["fds_before"] = len(os.listdir("/proc/self/fd"))
        def launch(name, args, run_end, clean_end):
            receipt = owned.execute(args, run_deadline=run_end, cleanup_deadline=clean_end,
                                    env=ENV, rss_limit=None, development_policy=POLICY)
            result["sessions"].append(dict(name=name, receipt=receipt))
            bounded_save(RUN/(name+"_session.json"), receipt)
            return receipt
        receipt = launch("native", [PYTHON, "-I", "-B", __file__, "--worker", anchor,
                                     str(deadline), str(ENTRY+200)], ENTRY+210, ENTRY+220)
        require(successful(receipt), "native A2 batch incomplete")
        result["worker_sha256"] = sha(RUN/"worker.json")
        value = load(RUN/"worker.json", result["worker_sha256"], 2**18)
        require(value["freeze_sha256"] == anchor and value["deadline"] == deadline
                and value["rlimit_as"] == [CAPS["as_bytes"]]*2, "worker identity/deadline/AS")
        # Check every durable event against the completed manifest, not only totals.
        for i, event in enumerate(value["events"], 1):
            require(load(RUN/("event_%03d.json" % i), limit=4096) == event, "saved native event identity")
        work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(value["work_counts"])
        before = sum(work.counts.values()); started = time.monotonic()
        online = cases.receive(RUN, value, work)
        result["online_reception"] = dict(online, work=sum(work.counts.values())-before,
                                         seconds=cases.duration(started, time.monotonic(), deadline, "online"))
        relocated = RUN/"relocated"; relocated.mkdir()
        started = time.monotonic()
        data_names = ["common.json", "worker.json"]+["arm_"+n+".json" for n in cases.ARMS]
        total = sum((ROOT/src).stat().st_size for src in frozen["portable_files"])
        total += sum((RUN/n).stat().st_size for n in data_names)
        require(2*(size()+total+2**20) <= CAPS["artifact_bytes"], "portable/archive capacity")
        for src, dest in frozen["portable_files"].items():
            target = relocated/dest; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/src, target)
            require(sha(target) == frozen["files"][src], "portable code identity")
        for name in data_names:
            shutil.copyfile(RUN/name, relocated/name)
            require(sha(RUN/name) == sha(relocated/name), "portable data identity")
        continuation = bounded_save(relocated/"continued.json", dict(work.counts), 4096)
        result["relocation_seconds"] = cases.duration(started, time.monotonic(), deadline, "relocation")
        receipt = launch("portable", [PYTHON, "-I", "-S", "-B", str(relocated/"margin_a2_recheck.py"),
                                       result["worker_sha256"], continuation["sha256"], str(deadline)],
                         ENTRY+260, ENTRY+280)
        require(successful(receipt), "portable A2 batch incomplete")
        portable = load(relocated/"receipt.json", limit=2**18)
        require(portable["status"] == "PASS" and portable["receipts"] == online["receipts"]
                and portable["native_calls"] == online["native_calls"] and portable["deadline"] == deadline
                and all(portable["work_counts"][k] >= work.counts[k] for k in work.counts),
                "portable identity/results/continued Work")
        result["portable_reception"] = portable
        result["fds_after"] = len(os.listdir("/proc/self/fd"))
        require(result["fds_before"] == result["fds_after"], "final FD balance")
        accounting_cut = time.monotonic()
        variable = {n: value["arms"][n]["seconds"]+online["arm_seconds"][n]+portable["arm_seconds"][n]
                    for n in cases.ARMS}
        measured = cases.duration(ENTRY, accounting_cut, deadline, "decision accounting")
        shared = measured-sum(variable.values())
        require(shared >= 0, "disjoint measured variable scopes")
        # Every standalone comparison pays ALL shared costs, not one ninth.
        costs = {n: shared+v for n, v in variable.items()}
        costs["transfer"] += variable["finite"]
        result["costs"] = dict(cut=accounting_cut, measured_batch_seconds=measured,
            shared_seconds=shared, variable_seconds=variable, standalone_through_cut=costs,
            transfer_origin_seconds=variable["finite"], shared_fully_charged_to_each=True,
            independent_cold_starts_measured=False, terminal_tail_in_cost_gate=False)
        # The terminal-export tail is measured separately. A positive cost gate
        # must survive the maximum remaining budget charged equally to all arms.
        tail_bound = deadline-accounting_cut
        conservative = {n: v+tail_bound for n, v in costs.items()}
        result["costs"].update(unmeasured_tail_upper_seconds=tail_bound,
                               decision_costs=conservative)
        result["decision"] = cases.decision(online["receipts"], conservative)
        result["through_cut_decision"] = cases.decision(online["receipts"], costs)
        if (result["decision"]["action"] == "GRAY_NO_AUTOMATIC_EXPANSION"
                and result["through_cut_decision"]["action"].startswith("LIMITED_MECHANISM_SIGNAL")):
            result["decision"]["cost_qualification"] = "UNRESOLVED_BY_CONSERVATIVE_TERMINAL_TAIL_NOT_COST_FAILURE"
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    if created:
        result["artifact_bytes_before_terminal"] = size()
        result["archive_reserve"] = "ONE_ADDITIONAL_COPY_OF_ALL_RUN_ARTIFACTS"
        result["terminal_clock"] = time.monotonic()
        result["elapsed_before_terminal_export"] = cases.duration(ENTRY, result["terminal_clock"], deadline, "terminal")
        receipt = bounded_save(RUN/"result.json", result)
        require(time.monotonic() < deadline, "late terminal rejected")
        print(json.dumps(dict(status=result["status"], receipt=receipt,
                              through_export_seconds=time.monotonic()-ENTRY,
                              stdout_flush_and_exit_in_duration=False)), flush=True)
    else:
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
    if len(sys.argv) == 3 and sys.argv[1] == "--execute-approved-once":
        raise SystemExit(main(sys.argv[2]))
    raise SystemExit("explicit frozen A2 invocation required")
