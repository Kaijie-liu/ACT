"""One frozen synthetic integration batch on the existing owned supervisor."""
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
from scoped_source.native_capture import canonical, require

FREEZE = ROOT/"docs/b1_native_capture_freeze_20261021.json"
RUN = Path("/data1/Kane/MOE/results/b1_native_capture_20261021_r1")


def binding(anchor):
    import b1_native_capture_cases as cases
    spec = load(FREEZE, anchor, 2**20)
    require(spec["policy"] == POLICY and spec["caps"] == CAPS and spec["attempts"] == 1
            and spec["result_directory"] == str(RUN) and spec["native_slots"] == list(cases.NATIVE_SLOTS)
            and spec["controls"] == list(cases.FAULTS), "frozen capture batch")
    for name, digest in spec["files"].items():
        require(sha(ROOT/name) == digest, "frozen code: "+name)
    for name, digest in spec["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/name).rglob("*.py"))}
        require(hashlib.sha256(canonical(members)).hexdigest() == digest, "frozen tree: "+name)
    return spec


def size():
    return sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())


def publish(name, value, cap=2**18):
    require(Path(name).name == name and name.endswith(".json"), "bounded output name")
    raw = canonical(value)
    require(len(raw) <= cap and 2*(size()+2*len(raw)+2**20) <= CAPS["artifact_bytes"],
            "cumulative publication/archive capacity")
    return save(RUN/name, value)


def worker(anchor, deadline, phase_deadline):
    limits(phase_deadline); binding(anchor)
    import b1_native_capture_cases as cases
    value = cases.run(deadline, publish)
    value.update(freeze_sha256=anchor, phase_deadline=phase_deadline,
                 rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)))
    receipt = publish("worker.json", value, 2**20)
    require(time.monotonic() < phase_deadline, "worker export cutoff")
    print(json.dumps(dict(status=value["status"], receipt=receipt)), flush=True)
    return 0 if value["status"] == "PASS" else 1


def main(anchor):
    deadline = ENTRY+300
    limits(deadline)
    from scoped_proof import owned_session as owned
    from scoped_source.development_work import DevelopmentWork
    import b1_native_capture_cases as cases
    result = dict(schema="B1_NATIVE_CAPTURE_CONTROL_V1", status="FAIL", policy=POLICY,
                  freeze_sha256=anchor, entry=ENTRY, deadline=deadline, caps=CAPS, sessions=[],
                  full_cold_start_seconds=None, trained_source=False, gpu=False,
                  complete_request=False, checked_feasibility=False, algorithm_gain=False)
    created = False
    try:
        require(sys.executable == PYTHON, "existing ACT interpreter")
        frozen = binding(anchor)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                timeout=min(5, max(.001, deadline-time.monotonic()))).strip()
        require(git("branch", "--show-current") == "feat/moe-route-verification"
                and not git("status", "--porcelain"), "clean frozen branch")
        result["execution_head"] = git("rev-parse", "HEAD")
        available = next(int(s.split()[1])*1024 for s in Path("/proc/meminfo").read_text().splitlines()
                         if s.startswith("MemAvailable:"))
        disk = shutil.disk_usage(RUN.parent).free
        require(time.monotonic() < ENTRY+60 and available >= 4*2**30
                and disk >= CAPS["artifact_bytes"], "one fresh bounded admission")
        result["admission"] = dict(seconds=time.monotonic()-ENTRY, available_bytes=available, disk_bytes=disk)
        RUN.mkdir(); created = True
        result["fds_before"] = len(os.listdir("/proc/self/fd"))
        def launch(name, args, run_end, clean_end):
            receipt = owned.execute(args, run_deadline=run_end, cleanup_deadline=clean_end,
                                    env=ENV, rss_limit=None, development_policy=POLICY)
            result["sessions"].append(dict(name=name, receipt=receipt))
            publish(name+"_session.json", receipt)
            return receipt
        receipt = launch("native", [PYTHON, "-I", "-B", __file__, "--worker", anchor,
                         str(deadline), str(ENTRY+200)], ENTRY+210, ENTRY+220)
        require(successful(receipt), "native capture controls incomplete")
        result["worker_sha256"] = sha(RUN/"worker.json")
        value = load(RUN/"worker.json", result["worker_sha256"], 2**20)
        require(value["freeze_sha256"] == anchor and value["deadline"] == deadline
                and value["rlimit_as"] == [CAPS["as_bytes"]]*2, "worker identity/AS/deadline")
        for i, event in enumerate(value["events"], 1):
            require(load(RUN/("event_%02d.json" % i), limit=4096) == event, "durable event identity")
        for i, ctrl in enumerate(value["controls"], 1):
            require(load(RUN/("control_%02d.json" % i), limit=4096) == ctrl, "durable control identity")
        work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(value["work_counts"])
        started = time.monotonic(); before = sum(work.counts.values())
        online = cases.replay(RUN, value, work)
        result["online_reception"] = dict(receipts=online, seconds=time.monotonic()-started,
            work=sum(work.counts.values())-before, work_counts=dict(work.counts))
        relocated = RUN/"relocated"; relocated.mkdir()
        data_names = ["worker.json"]+[r["name"] for r in value["records"]]
        total = sum((ROOT/p).stat().st_size for p in frozen["portable_files"])
        total += sum((RUN/n).stat().st_size for n in data_names)
        require(2*(size()+total+2**20) <= CAPS["artifact_bytes"], "portable/archive capacity")
        started = time.monotonic()
        for src, dest in frozen["portable_files"].items():
            target = relocated/dest; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/src, target)
            require(sha(target) == frozen["files"][src], "portable code hash")
        for name in data_names:
            shutil.copyfile(RUN/name, relocated/name)
            require(sha(RUN/name) == sha(relocated/name), "portable data hash")
        counts = save(relocated/"continued.json", dict(work.counts))
        result["relocation_seconds"] = time.monotonic()-started
        receipt = launch("portable", [PYTHON, "-I", "-S", "-B", str(relocated/"b1_native_capture_recheck.py"),
                        result["worker_sha256"], counts["sha256"], str(deadline)], ENTRY+260, ENTRY+280)
        require(successful(receipt), "portable reception incomplete")
        portable = load(relocated/"receipt.json", limit=2**18)
        require(portable["status"] == "PASS" and portable["deadline"] == deadline
                and portable["receipts"] == online
                and all(portable["work_counts"][k] >= work.counts[k] for k in work.counts),
                "same portable receipts/continued Work")
        result["portable_reception"] = portable
        result["fds_after"] = len(os.listdir("/proc/self/fd"))
        require(result["fds_before"] == result["fds_after"], "final descriptor balance")
        result["status"] = "PASS"
    except BaseException as exc:
        result["error"] = repr(exc)
    if created:
        result["artifact_bytes_before_terminal"] = size()
        result["archive_reserve"] = "ONE_ADDITIONAL_COPY_OF_ALL_RUN_ARTIFACTS"
        result["terminal_clock"] = time.monotonic()
        result["elapsed_before_terminal_export"] = result["terminal_clock"]-ENTRY
        receipt = publish("result.json", result)
        require(time.monotonic() < deadline, "late terminal rejected")
        print(json.dumps(dict(status=result["status"], receipt=receipt,
            through_export_seconds=time.monotonic()-ENTRY, stdout_flush_and_exit_in_duration=False)), flush=True)
    else:
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
    if len(sys.argv) == 3 and sys.argv[1] == "--execute-frozen-once":
        raise SystemExit(main(sys.argv[2]))
    raise SystemExit("explicit frozen B1 capture invocation required")
